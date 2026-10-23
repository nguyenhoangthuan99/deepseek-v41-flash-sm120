import os, sys, torch
sys.path.insert(0, "/DI")
sys.path.insert(0, "/DI/benchmarks/sm120")

from benchmarks.sm120.bench_ab_flashinfer_sparse_mla import (  # noqa: E402
    DIM, SCALE, make_latent, pack_fi, make_inputs,
)
from flashinfer.decode import trtllm_batch_decode_sparse_mla_dsv4 as fi_fn  # noqa: E402

PHASE = os.environ.get("PHASE", "decode")
T = int(os.environ.get("T", "16"))
H = int(os.environ.get("H", "16"))
TOPK = int(os.environ.get("TOPK", "512"))
CTX = int(os.environ.get("CTX", "32768"))

def main():
    torch.cuda.set_device(0)
    pool = pack_fi(make_latent(CTX, seed=7))
    q, idx, lens, cu, seq = make_inputs(T, H, TOPK, CTX)
    out = torch.empty((T, H, DIM), dtype=torch.bfloat16, device="cuda")
    ws = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    cu_use = torch.arange(T + 1, dtype=torch.int32, device="cuda") if PHASE == "prefill" else cu
    def run():
        fi_fn(query=q, swa_kv_cache=pool, workspace_buffer=ws, sparse_indices=idx,
              swa_topk_lens=lens, seq_lens=seq, out=out, bmm1_scale=SCALE,
              bmm2_scale=1.0, sinks=None, cum_seq_lens_q=cu_use, max_q_len=1)
    for _ in range(int(os.environ.get("WARM", "4"))):
        run()
    torch.cuda.synchronize()
    run(); torch.cuda.synchronize()
    print(f"done {PHASE} T={T} H={H} topk={TOPK} ctx={CTX} out0={float(out[0,0,0])}")

if __name__ == "__main__":
    main()
