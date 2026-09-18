#!/usr/bin/env python3
"""cuBLASLt MXFP8 (vec32 UE8M0) engine: numerics + timing on the six shapes.

The 32x32 weight-block scales expand to per-row 1x32 groups; payloads are
requantized against the UE8M0-rounded scales so scale*payload reproduces the
original dequantized operand (bounded by e4m3 rounding).
"""
import argparse
import ctypes
import itertools
import json
import statistics
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
SHAPES = [(1792, 5120), (4096, 1280), (5120, 1024), (576, 5120), (5120, 288),
          (25600, 6144)]
KG = 32



def to_blocked(x):
    """cuBLASLt/CUTLASS MX scale layout: 128x4 tiles, inner 32x16 swizzle."""
    rows, cols = x.shape
    rb, cb = -(-rows // 128), -(-cols // 4)
    padded = torch.zeros(rb * 128, cb * 4, dtype=x.dtype, device=x.device)
    padded[:rows, :cols] = x
    blocks = padded.view(rb, 128, cb, 4).permute(0, 2, 1, 3)
    return (blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1)
            .contiguous())
def build():
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


def median_time(run, iters, samples):
    for _ in range(10):
        run()
    times = []
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    for _ in range(samples):
        torch.cuda.synchronize(); s.record()
        for _ in range(iters):
            run()
        e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000 / iters)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--ms", default="64,128,512,2048")
    ap.add_argument("--jsonl", default=str(ROOT / "mxfp8-results.jsonl"))
    args = ap.parse_args()
    dll = build()
    torch.manual_seed(20260917)
    out = open(args.jsonl, "a", buffering=1)
    stream = torch.cuda.current_stream().cuda_stream

    for (N, K), M in itertools.product(SHAPES, [int(m) for m in args.ms.split(",")]):
        A = (torch.randn(M, K, device="cuda") / 8).to(torch.float8_e4m3fn)
        B = (torch.randn(N, K, device="cuda") / 8).to(torch.float8_e4m3fn)
        As = (torch.rand(M, K // KG, device="cuda") + 0.5) * 0.01
        Bs = (torch.rand(N // KG, K // KG, device="cuda") + 0.5) * 0.01
        a_deq = (A.float().view(M, K // KG, KG) * As.view(M, K // KG, 1)).view(M, K)
        b_deq = (B.float().view(N // KG, KG, K // KG, KG)
                 * Bs.view(N // KG, 1, K // KG, 1)).view(N, K)
        ref = a_deq @ b_deq.T

        # UE8M0 scales (power-of-two ceil) + payload requantization
        as_e = As.log2().ceil().clamp(-127, 127)
        bs_row = Bs.repeat_interleave(KG, dim=0)
        bs_e = bs_row.log2().ceil().clamp(-127, 127)
        aq = (a_deq / torch.pow(2.0, as_e).repeat_interleave(KG, 1)).to(torch.float8_e4m3fn)
        bq = (b_deq / torch.pow(2.0, bs_e).repeat_interleave(KG, 1)).to(torch.float8_e4m3fn)
        # Column-major problem: A' = our B (N,K), B' = our A (M,K); each
        # operand's scale is blocked over (rows, K/32).
        sa = to_blocked((as_e + 127).to(torch.uint8))
        sb = to_blocked((bs_e + 127).to(torch.uint8))
        D = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

        def run():
            rc = dll.mxfp8_gemm(aq.data_ptr(), bq.data_ptr(), sa.data_ptr(),
                                sb.data_ptr(), D.data_ptr(), M, N, K,
                                ctypes.c_void_p(stream))
            if rc:
                raise RuntimeError(f"rc={rc}: {dll.last_error().decode()}")
            return D

        row = {"M": M, "N": N, "K": K}
        try:
            o = run()
            row["rel_err"] = round(float((o.float() - ref).norm() / ref.norm()), 8)
            row["us"] = round(median_time(run, args.iters, args.samples), 2)
        except Exception as exc:  # noqa: BLE001
            row["unsupported"] = str(exc)[:200]
        out.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
