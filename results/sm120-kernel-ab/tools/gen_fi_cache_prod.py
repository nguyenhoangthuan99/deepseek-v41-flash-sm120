import os, sys, torch
AUTODIR = os.environ.get("AUTODIR", "/tmp/fi_autotune_prod")
os.environ["FLASHINFER_AUTOTUNE_DIR"] = AUTODIR
os.makedirs(AUTODIR, exist_ok=True)
sys.path.insert(0, "/DI"); sys.path.insert(0, "/DI/benchmarks/sm120")
from benchmarks.sm120.bench_ab_flashinfer_sparse_mla import DIM, SCALE, make_latent, pack_fi
from flashinfer.decode import trtllm_batch_decode_sparse_mla_dsv4 as fi_public
from flashinfer.autotuner import autotune

CTX = 32768
SWA_W, EXTRA_W = 128, 512
SHAPES = [(T, H) for H in (16, 64) for T in (2, 4, 8, 16, 32, 64)]

def main():
    torch.cuda.set_device(0)
    swa_pool = pack_fi(make_latent(CTX, seed=7)).permute(0, 2, 1, 3).contiguous()
    comp_pool = pack_fi(make_latent(CTX, seed=99)).permute(0, 2, 1, 3).contiguous()
    ws = torch.zeros(128*1024*1024, dtype=torch.uint8, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(3)
    cache_path = os.path.join(AUTODIR, "sparse_mla_sm120_decode_dsv4.json")
    with autotune(True, cache=cache_path):
        for T, H in SHAPES:
            q = (torch.randn((T, H, DIM), generator=g, device="cuda") * 0.5).to(torch.bfloat16)
            swa_idx = torch.randint(0, CTX, (T, SWA_W), generator=g, device="cuda", dtype=torch.int32)
            extra_idx = torch.randint(0, CTX, (T, 1, EXTRA_W), generator=g, device="cuda", dtype=torch.int32)
            swa_lens = torch.full((T,), SWA_W, dtype=torch.int32, device="cuda")
            extra_lens = torch.full((T,), EXTRA_W, dtype=torch.int32, device="cuda")
            out = torch.empty((T, H, DIM), dtype=torch.bfloat16, device="cuda")
            try:
                for _ in range(3):
                    fi_public(query=q, swa_kv_cache=swa_pool, workspace_buffer=ws,
                              sparse_indices=swa_idx, compressed_kv_cache=comp_pool,
                              out=out, bmm1_scale=SCALE, sinks=None, kv_layout="NHD",
                              swa_topk_lens=swa_lens, extra_sparse_indices=extra_idx,
                              extra_sparse_topk_lens=extra_lens)
                torch.cuda.synchronize()
                print(f"tuned prod T={T} H={H}", flush=True)
            except Exception as exc:
                print(f"skip prod T={T} H={H}: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
    print("cache:", os.path.exists(cache_path), os.path.getsize(cache_path) if os.path.exists(cache_path) else 0)

if __name__ == "__main__":
    main()
