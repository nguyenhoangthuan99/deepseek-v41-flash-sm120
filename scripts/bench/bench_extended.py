#!/usr/bin/env python3
"""Extended engine comparison for the DeepSeek-V4.1-Flash DENSE block-FP8 linears.

Runs every serving shape (the six [32,32] block-FP8 weight shapes) across a wide M
grid and reports, per cell, the time and numeric error of every candidate engine:

  triton_tuned    serving-stack split-K kernel + measured config (kit adapter)
  triton_generic  upstream fallback kernel (what you get with no config table)
  cuda_custom     hand-written mma.sync m16n8k32 e4m3 kernel, autotuned
  cublas_mxfp8    cuBLASLt VEC32_UE8M0 MXFP8 (checkpoint-native ue8m0 scales)
  cublas_bf16     dequantized bf16 cuBLAS (dequant excluded) upper bound

Then emits a per-shape CROSSOVER table: the smallest M at which cuBLASLt MXFP8
beats the tuned Triton kernel, and the chosen dispatch threshold the serving
adapter should use (mirrors scripts/../mxfp8_crossover.json).

Run inside the serving container (needs the sglang tree + nvcc + cuBLASLt):
  docker exec -e PYTHONPATH=/opt/dsv41/runtime:/s-s glang/python \
      dsv41 python3 bench_extended.py --jsonl /work/extended.jsonl
"""
from __future__ import annotations

import argparse
import ctypes
import itertools
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
KG = 32

# All six dense block-FP8 serving shapes: (N, K). These are the [32,32] weight
# matrices the cuBLASLt MXFP8 engine would replace in the Fp8LinearMethod path.
SHAPES = [(1792, 5120), (4096, 1280), (5120, 1024), (576, 5120), (5120, 288),
          (25600, 6144)]
if os.environ.get("W8A8_SHAPES"):
    SHAPES = [tuple(map(int, s.split("x"))) for s in os.environ["W8A8_SHAPES"].split(",")]

# Decode M is ~concurrency; prefill chunks are 512..4096. Grid straddles both.
DEFAULT_MS = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 256, 384, 512, 768,
              1024, 1536, 2048, 3072, 4096]

VARIANTS = [  # (BM, BN, WM, WN, STAGES)
    (32, 32, 1, 2, 2), (32, 64, 1, 4, 2), (64, 32, 2, 2, 2),
    (64, 64, 2, 2, 2), (64, 64, 2, 2, 3), (64, 128, 2, 4, 2),
    (128, 64, 4, 2, 2), (128, 128, 4, 4, 2), (64, 64, 1, 4, 2),
    (128, 32, 4, 1, 2),
]
SPLITKS = [int(x) for x in os.environ.get("W8A8_SPLITKS", "1,2,4,8").split(",")]


def build_cuda(bm, bn, wm, wn, stages):
    name = f"w8a8_bm{bm}_bn{bn}_w{wm}x{wn}_s{stages}"
    lib = ROOT / "build" / f"{name}.so"
    lib.parent.mkdir(exist_ok=True)
    if not lib.exists():
        cmd = ["nvcc", "-O3", "-std=c++17", "--shared", "-Xcompiler", "-fPIC",
               "-gencode", "arch=compute_120a,code=sm_120a",
               f"-DBM={bm}", f"-DBN={bn}", f"-DWM={wm}", f"-DWN={wn}",
               f"-DSTAGES={stages}", str(ROOT / "w8a8_sm120.cu"), "-o", str(lib)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return None, r.stderr[-400:]
    dll = ctypes.CDLL(str(lib))
    dll.launch_w8a8.restype = ctypes.c_int
    dll.launch_w8a8.argtypes = ([ctypes.c_void_p] * 5 + [ctypes.c_int] * 4
                                + [ctypes.c_void_p])
    return dll, None


def build_mxfp8():
    lib = ROOT / "build" / "cublaslt_mxfp8.so"
    lib.parent.mkdir(exist_ok=True)
    cmd = ["nvcc", "-O3", "-std=c++17", "--shared", "-Xcompiler", "-fPIC",
           "-gencode", "arch=compute_120a,code=sm_120a",
           str(ROOT / "cublaslt_mxfp8.cu"), "-lcublasLt", "-o", str(lib)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("nvcc failed:\n" + r.stderr[-1500:])
    dll = ctypes.CDLL(str(lib))
    dll.mxfp8_gemm.restype = ctypes.c_int
    dll.mxfp8_gemm.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    dll.last_error.restype = ctypes.c_char_p
    return dll


def to_blocked(x):
    """cuBLASLt/CUTLASS MX scale layout: 128x4 tiles, inner 32x16 swizzle."""
    rows, cols = x.shape
    rb, cb = -(-rows // 128), -(-cols // 4)
    padded = torch.zeros(rb * 128, cb * 4, dtype=x.dtype, device=x.device)
    padded[:rows, :cols] = x
    blocks = padded.view(rb, 128, cb, 4).permute(0, 2, 1, 3)
    return (blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1).contiguous())


def median_time(run, iters, samples):
    for _ in range(10):
        run()
    times = []
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(samples):
        torch.cuda.synchronize()
        s.record()
        for _ in range(iters):
            run()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000 / iters)
    return statistics.median(times)


def load_candidates():
    """Upstream fp8 kernel, kit-tuned config table, optional DeepGEMM."""
    cands = {}
    try:
        sys.path.insert(0, "/opt/dsv41/runtime")
        import sglang.kernels.ops.quantization.fp8_kernel as fp8k  # noqa: F401
        cands["fp8k"] = fp8k
        import sm120_fp8_patch  # noqa: F401
        cands["patch"] = sm120_fp8_patch
        table = json.loads(
            Path("/opt/dsv41/runtime/fp8_sm120_configs.json").read_text())
        cands["tuned"] = {
            (e["N"], e["K"]): {int(m): c for m, c in e["configs"].items()
                               if c is not None}
            for e in table["shapes"]}
    except Exception as exc:  # noqa: BLE001
        cands["error"] = str(exc)
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--ms", default=None)
    ap.add_argument("--jsonl", default=str(ROOT / "extended-results.jsonl"))
    ap.add_argument("--crossover-out", default=str(ROOT / "mxfp8_crossover.json"))
    ap.add_argument("--no-cuda", action="store_true", help="skip the custom CUDA engine")
    args = ap.parse_args()

    torch.manual_seed(20260917)
    out = open(args.jsonl, "a", buffering=1)

    def emit(**row):
        out.write(json.dumps(row) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "candidates"}),
              flush=True)

    ms = DEFAULT_MS if not args.ms else [int(x) for x in args.ms.split(",")]
    cands = load_candidates()
    if "error" in cands:
        emit(kind="candidate_load_error", err=cands["error"])
    fp8k = cands.get("fp8k")
    patch = cands.get("patch")
    tuned = cands.get("tuned", {})

    mxfp8 = build_mxfp8()
    stream = torch.cuda.current_stream().cuda_stream

    libs = {}
    if not args.no_cuda:
        for v in VARIANTS:
            dll, err = build_cuda(*v)
            if dll is not None:
                libs[v] = dll
            else:
                emit(kind="build_failed", variant=list(v), err=err)
        emit(kind="built", variants=[list(v) for v in libs])

    crossover = {}

    for (N, K), M in itertools.product(SHAPES, ms):
        if K % KG or N % KG:
            emit(kind="skip", reason="K/N not divisible by 32", M=M, N=N, K=K)
            continue
        A = (torch.randn(M, K, device="cuda") / 8).to(torch.float8_e4m3fn)
        B = (torch.randn(N, K, device="cuda") / 8).to(torch.float8_e4m3fn)
        # Checkpoint-native UE8M0 order-of-magnitude scales (2^-k) == po2.
        As = torch.pow(2.0, -(torch.randint(2, 10, (M, K // KG),
                                            device="cuda").float()))
        Bs = torch.pow(2.0, -(torch.randint(2, 10, (N // KG, K // KG),
                                            device="cuda").float()))
        a_deq = (A.float().view(M, K // KG, KG) * As.view(M, K // KG, 1)).view(M, K)
        b_deq = (B.float().view(N // KG, KG, K // KG, KG)
                 * Bs.view(N // KG, 1, K // KG, 1)).view(N, K)
        ref = a_deq @ b_deq.T
        refn = ref.norm()

        def rel(o):
            return round(float((o.float() - ref).norm() / refn), 8)

        row = {"M": M, "N": N, "K": K}

        if patch is not None and (N, K) in tuned:
            cfgs = tuned[(N, K)]
            cfg = cfgs[min(cfgs, key=lambda m: abs(m - M))]
            def run_tuned(cfg=cfg):
                return patch.matmul(A, B, As, Bs, [KG, KG],
                                    output_dtype=torch.bfloat16, config=cfg)
            try:
                row["triton_tuned_us"] = round(
                    median_time(run_tuned, args.iters, args.samples), 2)
                row["triton_tuned_rel"] = rel(run_tuned())
            except Exception as exc:  # noqa: BLE001
                row["triton_tuned_err"] = str(exc)[:160]

        if fp8k is not None:
            # Use the Triton kernel directly: w8a8_block_fp8_matmul dispatches to
            # DeepGEMM on bf16, which asserts on [32,32] blocks (needs 1x128).
            def run_generic():
                return fp8k.w8a8_block_fp8_matmul_triton(A, B, As, Bs, [KG, KG],
                                                         output_dtype=torch.bfloat16)
            try:
                row["triton_generic_us"] = round(
                    median_time(run_generic, args.iters, args.samples), 2)
                row["triton_generic_rel"] = rel(run_generic())
            except Exception as exc:  # noqa: BLE001
                row["triton_generic_err"] = str(exc)[:160]

        # cuBLASLt MXFP8: 32x32 blocks expand losslessly to 1x32 UE8M0 rows.
        try:
            bs_row = Bs.repeat_interleave(KG, dim=0)
            sa = to_blocked((As.log2().to(torch.int32) + 127).clamp(0, 255)
                            .to(torch.uint8))
            sb = to_blocked((bs_row.log2().to(torch.int32) + 127).clamp(0, 255)
                            .to(torch.uint8))
            aq = (a_deq / torch.pow(2.0, As.log2())
                  .repeat_interleave(KG, 1)).to(torch.float8_e4m3fn).contiguous()
            bq = (b_deq / torch.pow(2.0, bs_row.log2())
                  .repeat_interleave(KG, 1)).to(torch.float8_e4m3fn).contiguous()
            D = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

            def run_mx():
                rc = mxfp8.mxfp8_gemm(aq.data_ptr(), bq.data_ptr(),
                                      sa.data_ptr(), sb.data_ptr(),
                                      D.data_ptr(), M, N, K,
                                      ctypes.c_void_p(stream))
                if rc:
                    raise RuntimeError(f"rc={rc}: {mxfp8.last_error().decode()}")
                return D
            o = run_mx()
            row["cublas_mxfp8_us"] = round(
                median_time(run_mx, args.iters, args.samples), 2)
            row["cublas_mxfp8_rel"] = rel(o)
        except Exception as exc:  # noqa: BLE001
            row["cublas_mxfp8_err"] = str(exc)[:200]

        a16, b16 = a_deq.bfloat16(), b_deq.bfloat16()
        def run_bf16():
            return a16 @ b16.T
        row["cublas_bf16_dequant_us"] = round(
            median_time(run_bf16, args.iters, args.samples), 2)

        if libs:
            best = None
            cand_list = []
            for v, dll in libs.items():
                for sk in SPLITKS:
                    if (K // KG) < sk:
                        continue
                    part = torch.empty(sk, M, N, device="cuda", dtype=torch.float32)
                    def run_cuda(dll=dll, sk=sk, part=part):
                        err = dll.launch_w8a8(A.data_ptr(), B.data_ptr(),
                                              As.data_ptr(), Bs.data_ptr(),
                                              part.data_ptr(), M, N, K, sk,
                                              ctypes.c_void_p(stream))
                        if err:
                            raise RuntimeError(f"cuda error {err}")
                        return part.sum(0).bfloat16() if sk > 1 else part[0].bfloat16()
                    try:
                        e_ = rel(run_cuda())
                        if e_ > 5e-3:
                            cand_list.append({"v": list(v), "sk": sk, "bad_numerics": e_})
                            continue
                        us = round(median_time(run_cuda, args.iters, args.samples), 2)
                        c = {"v": list(v), "sk": sk, "us": us, "rel_err": e_}
                        cand_list.append(c)
                        if best is None or us < best["us"]:
                            best = c
                    except Exception as exc:  # noqa: BLE001
                        cand_list.append({"v": list(v), "sk": sk, "err": str(exc)[:100]})
            if best:
                row["cuda_custom_us"] = best["us"]
                row["cuda_cfg"] = best
            row["candidates"] = cand_list

        # name the winner among the engines actually wired/considered
        engines = {"triton_tuned": row.get("triton_tuned_us"),
                   "cuda_custom": row.get("cuda_custom_us"),
                   "cublas_mxfp8": row.get("cublas_mxfp8_us")}
        engines = {k: v for k, v in engines.items() if v}
        if engines:
            win = min(engines, key=engines.get)
            row["best_engine"] = win
            row["best_us"] = engines[win]
            tt = row.get("triton_tuned_us")
            if tt:
                row["best_speedup_vs_tuned"] = round(tt / engines[win], 3)
                row["mxfp8_vs_tuned"] = (round(tt / row["cublas_mxfp8_us"], 3)
                                         if row.get("cublas_mxfp8_us") else None)
        emit(**row)

        # accumulate crossover: smallest M where mxfp8 beats tuned triton by >1.05x
        if row.get("triton_tuned_us") and row.get("cublas_mxfp8_us"):
            key = f"{N}x{K}"
            ratio = row["triton_tuned_us"] / row["cublas_mxfp8_us"]
            crossover.setdefault(key, {"N": N, "K": K, "points": []})
            crossover[key]["points"].append(
                {"M": M, "mxfp8_over_tuned": round(ratio, 3)})

    # summarize crossover: smallest M with ratio >= 1.05, else null (never wins)
    for key, entry in crossover.items():
        pts = sorted(entry["points"], key=lambda p: p["M"])
        winners = [p["M"] for p in pts if p["mxfp8_over_tuned"] >= 1.05]
        entry["threshold_m"] = winners[0] if winners else None
        entry["always_wins"] = bool(winners and winners[0] == pts[0]["M"])
        entry["never_wins"] = not winners

    emit(kind="done")
    if crossover:
        Path(args.crossover_out).write_text(json.dumps(
            {"kind": "mxfp8_crossover", "shapes": crossover}, indent=2))
        print("crossover written to", args.crossover_out)


if __name__ == "__main__":
    main()