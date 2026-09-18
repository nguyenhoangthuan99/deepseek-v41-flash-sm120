#!/usr/bin/env python3
"""Direct output comparison of every engine against the Triton kernels.

Same quantized inputs go through:
  triton_tuned (serving kernel + config), triton_generic (upstream fallback),
  cuda_custom (all autotune-winning variants), cublas_mxfp8 (po2 scales only).
Reports, per (N,K,M): max abs diff, rel L2 diff, and bitwise-equal fraction
against triton_tuned bf16 output. Ragged M values exercise tail guards.

Split-K and tile order change fp32 summation order, so bitwise equality is NOT
expected; the pass bar is rel L2 <= 2e-3 and max_abs <= 4x the generic/tuned
disagreement on the same inputs (both bf16 outputs of fp32 accumulations).
"""
import ctypes
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, "/opt/dsv41/runtime")
import sm120_fp8_patch  # noqa: E402
import sglang.kernels.ops.quantization.fp8_kernel as fp8k  # noqa: E402

spec = importlib.util.spec_from_file_location("bm", ROOT / "bench_mxfp8.py")
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)

KG = 32
SHAPES = [(1792, 5120), (4096, 1280), (5120, 1024), (576, 5120), (5120, 288),
          (25600, 6144)]
# Full M grid incl. ragged tails and every threshold-relevant point, so
# correctness is verified wherever the dispatch table can route to MXFP8.
MS = [1, 2, 4, 8, 16, 32, 48, 64, 100, 128, 160, 256, 384, 512, 768,
      777, 1024, 1536, 2048, 3072, 4096]
CUDA_VARIANTS = [(64, 64, 2, 2, 2), (64, 128, 2, 4, 2), (128, 64, 4, 2, 2),
                 (32, 64, 1, 4, 2)]
SPLITKS = [1, 4, 8]

table = json.loads(Path("/opt/dsv41/runtime/fp8_sm120_configs.json").read_text())
tuned = {(e["N"], e["K"]): {int(m): c for m, c in e["configs"].items() if c}
         for e in table["shapes"]}

mx = ctypes.CDLL(str(ROOT / "build/cublaslt_mxfp8.so"))
mx.mxfp8_gemm.restype = ctypes.c_int
mx.mxfp8_gemm.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]

cuda_libs = {}
for v in CUDA_VARIANTS:
    bmv, bnv, wmv, wnv, sv = v
    lib = ROOT / "build" / f"w8a8_bm{bmv}_bn{bnv}_w{wmv}x{wnv}_s{sv}.so"
    dll = ctypes.CDLL(str(lib))
    dll.launch_w8a8.restype = ctypes.c_int
    dll.launch_w8a8.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    cuda_libs[v] = dll

torch.manual_seed(1234)
stream = torch.cuda.current_stream().cuda_stream
failures = 0
rows = []
out = open(ROOT / "check-vs-triton-extended.jsonl", "a", buffering=1)

for (N, K) in SHAPES:
    for M in MS:
        A = (torch.randn(M, K, device="cuda") / 8).to(torch.float8_e4m3fn)
        B = (torch.randn(N, K, device="cuda") / 8).to(torch.float8_e4m3fn)
        # po2 scales: realistic (checkpoint ue8m0) AND makes mxfp8 comparable
        As = torch.pow(2.0, torch.randint(-9, -5, (M, K // KG), device="cuda").float())
        Bs = torch.pow(2.0, torch.randint(-9, -5, (N // KG, K // KG), device="cuda").float())

        cfgs = tuned.get((N, K))
        cfg = cfgs[min(cfgs, key=lambda m: abs(m - M))] if cfgs else None
        # Use the Triton kernel directly: w8a8_block_fp8_matmul routes to
        # DeepGEMM on bf16, which asserts on [32,32] blocks (wants 1x128).
        ref_t = (sm120_fp8_patch.matmul(A, B, As, Bs, [KG, KG],
                                        output_dtype=torch.bfloat16, config=cfg)
                 if cfg else
                 fp8k.w8a8_block_fp8_matmul_triton(A, B, As, Bs, [KG, KG],
                                                   output_dtype=torch.bfloat16))
        gen = fp8k.w8a8_block_fp8_matmul_triton(A, B, As, Bs, [KG, KG],
                                                output_dtype=torch.bfloat16)
        torch.cuda.synchronize()

        def stats(o, name):
            d = (o.float() - ref_t.float())
            rel = float(d.norm() / ref_t.float().norm())
            mab = float(d.abs().max())
            bit = float((o == ref_t).float().mean())
            return {"engine": name, "rel_l2_vs_tuned": round(rel, 8),
                    "max_abs_vs_tuned": round(mab, 6),
                    "bitwise_equal_frac": round(bit, 4)}

        base = stats(gen, "triton_generic")   # engine-noise yardstick
        row = {"M": M, "N": N, "K": K, "checks": [base]}
        bar_rel, bar_abs = 2e-3, max(4 * base["max_abs_vs_tuned"], 1e-4)

        for v, dll in cuda_libs.items():
            for sk in SPLITKS:
                if K // KG < sk:
                    continue
                part = torch.empty(sk, M, N, device="cuda", dtype=torch.float32)
                rc = dll.launch_w8a8(A.data_ptr(), B.data_ptr(), As.data_ptr(),
                                     Bs.data_ptr(), part.data_ptr(), M, N, K,
                                     sk, ctypes.c_void_p(stream))
                assert rc == 0, rc
                torch.cuda.synchronize()
                o = part.sum(0).bfloat16() if sk > 1 else part[0].bfloat16()
                s = stats(o, f"cuda_{v}_sk{sk}")
                s["pass"] = s["rel_l2_vs_tuned"] <= bar_rel and s["max_abs_vs_tuned"] <= bar_abs
                failures += not s["pass"]
                row["checks"].append(s)

        a_deq = (A.float().view(M, K // KG, KG) * As.view(M, K // KG, 1)).view(M, K)
        b_deq = (B.float().view(N // KG, KG, K // KG, KG)
                 * Bs.view(N // KG, 1, K // KG, 1)).view(N, K)
        aq = (a_deq / As.repeat_interleave(KG, 1)).to(torch.float8_e4m3fn)
        sa = bm.to_blocked((As.log2() + 127).to(torch.uint8))
        sb = bm.to_blocked((Bs.repeat_interleave(KG, 0).log2() + 127).to(torch.uint8))
        D = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        rc = mx.mxfp8_gemm(aq.data_ptr(), B.data_ptr(), sa.data_ptr(),
                           sb.data_ptr(), D.data_ptr(), M, N, K,
                           ctypes.c_void_p(stream))
        assert rc == 0, rc
        torch.cuda.synchronize()
        s = stats(D, "cublas_mxfp8_po2")
        s["pass"] = s["rel_l2_vs_tuned"] <= bar_rel and s["max_abs_vs_tuned"] <= bar_abs
        failures += not s["pass"]
        row["checks"].append(s)

        worst = max(row["checks"][1:], key=lambda c: c["rel_l2_vs_tuned"])
        print(f"({N:>5},{K:>4}) M={M:<5} generic_vs_tuned rel={base['rel_l2_vs_tuned']:.1e} "
              f"| worst engine {worst['engine']}: rel={worst['rel_l2_vs_tuned']:.1e} "
              f"max_abs={worst['max_abs_vs_tuned']:.4f} "
              f"{'ALL PASS' if all(c.get('pass', True) for c in row['checks']) else 'FAIL'}",
              flush=True)
        out.write(json.dumps(row) + "\n")
        rows.append(row)

total = sum(len(r["checks"]) - 1 for r in rows)
print(f"\ncomparisons={total} failures={failures}", flush=True)
sys.exit(1 if failures else 0)
