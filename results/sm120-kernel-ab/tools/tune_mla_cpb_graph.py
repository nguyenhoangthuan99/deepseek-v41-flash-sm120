import os, sys, json, statistics, torch
sys.path.insert(0, "/DI"); sys.path.insert(0, "/DI/benchmarks/sm120")
from benchmarks.sm120.bench_ab_flashinfer_sparse_mla import DIM, SCALE, make_latent, pack_fi, make_inputs
from flashinfer.mla._sparse_mla_sm120 import sparse_mla_sm120_decode_dsv4
from flashinfer.decode import trtllm_batch_decode_sparse_mla_dsv4 as fi_public

CTX = int(os.environ.get("CTX", "32768"))
TOPK = int(os.environ.get("TOPK", "512"))
HEADS = [int(x) for x in os.environ.get("HEADS", "16").split(",")]
TOKENS = [int(x) for x in os.environ.get("TOKENS", "1,2,4,8,16,32,64").split(",")]
CPBS = [int(x) for x in os.environ.get("CPBS", "0,1,2,3,4,5,6,8").split(",")]

def capture(fn, warm=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g

def time_graph(g, iters=100):
    for _ in range(5): g.replay()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    samples = []
    for _ in range(3):
        st.record()
        for _ in range(iters): g.replay()
        en.record(); torch.cuda.synchronize()
        samples.append(st.elapsed_time(en)/iters*1000)
    return statistics.median(samples)

def main():
    torch.cuda.set_device(0)
    pool = pack_fi(make_latent(CTX, seed=7))
    pool_flat = pool.reshape(pool.shape[0], -1)
    rows = []
    for H in HEADS:
        for T in TOKENS:
            q, idx, lens, cu, seq = make_inputs(T, H, TOPK, CTX)
            out_pub = torch.empty((T, H, DIM), dtype=torch.bfloat16, device="cuda")
            ws = torch.zeros(128*1024*1024, dtype=torch.uint8, device="cuda")
            def pub():
                fi_public(query=q, swa_kv_cache=pool, workspace_buffer=ws, sparse_indices=idx,
                          swa_topk_lens=lens, seq_lens=seq, out=out_pub, bmm1_scale=SCALE,
                          bmm2_scale=1.0, sinks=None, cum_seq_lens_q=cu, max_q_len=1)
            try:
                g_pub = capture(pub); base = time_graph(g_pub)
            except Exception as exc:
                print(f"T={T} H={H} public capture failed: {type(exc).__name__}: {str(exc)[:120]}"); continue
            num_splits = (TOPK + 63)//64
            mid_out = torch.empty((T,H,num_splits,DIM), dtype=torch.bfloat16, device="cuda")
            mid_lse = torch.empty((T,H,num_splits), dtype=torch.float32, device="cuda")
            out = torch.empty_like(out_pub)
            out_lse = torch.empty((T,H), dtype=torch.float32, device="cuda")
            best = None
            for cpb in CPBS:
                def run(cpb=cpb):
                    sparse_mla_sm120_decode_dsv4(q, pool_flat, idx, mid_out, mid_lse, out, out_lse, SCALE,
                                                 topk_length=lens, chunks_per_block=(None if cpb==0 else cpb))
                try:
                    g = capture(run); us = time_graph(g)
                except Exception as exc:
                    print(f"  cpb={cpb} capture failed: {type(exc).__name__}: {str(exc)[:100]}"); continue
                # correctness vs public
                run(); torch.cuda.synchronize()
                rel = ((out.float()-out_pub.float()).pow(2).mean().sqrt()/out_pub.float().pow(2).mean().sqrt().clamp_min(1e-12)).item()
                print(f"T={T:3d} H={H:2d} cpb={cpb}: kernel={us:7.2f} us  public_capture={base:7.2f}  ratio={us/base:.3f}  rel_l2={rel:.1e}")
                if best is None or us < best[1]: best=(cpb,us,rel)
            if best:
                rows.append({"T":T,"H":H,"topk":TOPK,"ctx":CTX,"public_us":base,"best_cpb":best[0],
                             "best_us":best[1],"ratio":best[1]/base,"rel_l2":best[2]})
                print(f"BEST T={T} H={H}: cpb={best[0]} {best[1]:.2f}us vs public {base:.2f}us ({best[1]/base:.3f})")
    out_path = "/DI/benchmarks/sm120/results/mla_cpb_graph_sweep.json"
    json.dump(rows, open(out_path,"w"), indent=2)
    print("wrote", out_path)

if __name__ == "__main__":
    main()
