import os, sys, statistics, torch
sys.path.insert(0, "/DI"); sys.path.insert(0, "/DI/benchmarks/sm120")
from benchmarks.sm120.bench_ab_flashinfer_sparse_mla import DIM, SCALE, make_latent, pack_fi
from flashinfer.decode import trtllm_batch_decode_sparse_mla_dsv4 as fi_public
CTX, SWA_W, EXTRA_W = 32768, 128, 512
SHAPES = [(T, int(os.environ.get("H","16"))) for T in (2,4,8,16,32,64)]
def capture(fn, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize(); return g
def tg(g, iters=200):
    for _ in range(5): g.replay()
    torch.cuda.synchronize(); st,en=torch.cuda.Event(True),torch.cuda.Event(True); vals=[]
    for _ in range(3):
        st.record()
        for _ in range(iters): g.replay()
        en.record(); torch.cuda.synchronize(); vals.append(st.elapsed_time(en)/iters*1000)
    return statistics.median(vals)
def main():
    torch.cuda.set_device(0)
    swa_pool = pack_fi(make_latent(CTX, seed=7)).permute(0, 2, 1, 3).contiguous(); comp_pool = pack_fi(make_latent(CTX, seed=99)).permute(0, 2, 1, 3).contiguous()
    ws = torch.zeros(128*1024*1024, dtype=torch.uint8, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(3)
    for T,H in SHAPES:
        q=(torch.randn((T,H,DIM),generator=g,device="cuda")*0.5).to(torch.bfloat16)
        swa_idx=torch.randint(0,CTX,(T,SWA_W),generator=g,device="cuda",dtype=torch.int32)
        extra_idx=torch.randint(0,CTX,(T,1,EXTRA_W),generator=g,device="cuda",dtype=torch.int32)
        swa_lens=torch.full((T,),SWA_W,dtype=torch.int32,device="cuda")
        extra_lens=torch.full((T,),EXTRA_W,dtype=torch.int32,device="cuda")
        out=torch.empty((T,H,DIM),dtype=torch.bfloat16,device="cuda")
        def run():
            fi_public(query=q, swa_kv_cache=swa_pool, workspace_buffer=ws, sparse_indices=swa_idx,
                      compressed_kv_cache=comp_pool, out=out, bmm1_scale=SCALE, sinks=None,
                      kv_layout="NHD", swa_topk_lens=swa_lens, extra_sparse_indices=extra_idx,
                      extra_sparse_topk_lens=extra_lens)
        print(f"T={T:3d} H={H:2d} prod: {tg(capture(run)):7.2f} us", flush=True)
if __name__=="__main__": main()
