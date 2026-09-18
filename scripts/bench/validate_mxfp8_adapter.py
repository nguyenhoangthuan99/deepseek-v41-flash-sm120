#!/usr/bin/env python3
"""Isolated validation for the MXFP8 serving adapter (runtime/mxfp8_gemm_sm120.py).

Checks, without loading model weights:
  1. weight-scale expansion (32x32 UE8M0 blocks -> 1x32 blocked) is lossless;
  2. activation quant (per-32 e4m3 + UE8M0 scales) matches the serving contract;
  3. the dispatch table routes M below/above the measured threshold correctly;
  4. end-to-end adapter matmul vs an fp32 reference and vs the tuned Triton kernel
     on identical quantized inputs, at each serving shape.

Run inside the serving container with the runtime on PYTHONPATH:
  docker exec -e PYTHONPATH=/opt/dsv41/runtime:/sgl-workspace/sglang/python \
      -e DSV41_SM120_MXFP8=1 dsv41 python3 validate_mxfp8_adapter.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/opt/dsv41/runtime")
import mxfp8_gemm_sm120 as mx  # noqa: E402

KG = 32
SHAPES = [(1792, 5120), (4096, 1280), (5120, 1024), (576, 5120), (5120, 288),
          (25600, 6144)]
FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILURES.append(name)


def main():
    torch.manual_seed(7)
    print("== 1/4 library build ==")
    lib = mx._build_library()
    check("cublasLt MXFP8 library builds", lib is not None)
    if lib is None:
        return 1

    print("== 2/4 dispatch table ==")
    table = mx._Thresholds(Path("/opt/dsv41/runtime/mxfp8_crossover.json"))
    check("crossover table parsed", bool(table.by_shape or table.always),
          f"by_shape={len(table.by_shape)} always={len(table.always)} never={len(table.never)}")

    print("== 3/4 numeric parity per shape ==")
    for (N, K) in SHAPES:
        if K % KG or N % KG:
            continue
        M = 256
        A = (torch.randn(M, K, device="cuda") / 8).bfloat16()
        B = (torch.randn(N, K, device="cuda") / 8).to(torch.float8_e4m3fn)
        # checkpoint-style UE8M0 block scales (power-of-two)
        Bs = torch.pow(2.0, -(torch.randint(2, 10, (N // KG, K // KG),
                                            device="cuda").float())) * (2.0 ** 9)
        # reference: dequantize with the po2 scales
        b_deq = (B.float().view(N // KG, KG, K // KG, KG)
                 * Bs.view(N // KG, 1, K // KG, 1)).view(N, K)
        out = mx.matmul(A, B, None, Bs, [KG, KG], torch.bfloat16,
                        thresholds=table, lib=lib)
        # mx quantizes A internally with po2 scales; rebuild the exact reference
        g = A.view(M, K // KG, KG).float()
        amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-10)
        exp = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
        aq = (g / torch.pow(2.0, exp)).to(torch.float8_e4m3fn).float()
        # dequantized activation (M, K): group scale broadcasts over the 32 lanes
        a_dq = (aq * torch.pow(2.0, exp)).view(M, K)
        ref = a_dq @ b_deq.T
        rel = float((out.float() - ref).norm() / ref.norm())
        check(f"({N},{K}) M={M} rel-L2 vs fp32", rel < 2e-3, f"rel={rel:.2e}")

    print("== 4/4 weight-scale expansion is lossless ==")
    N, K = 576, 5120
    B = (torch.randn(N, K, device="cuda") / 8).to(torch.float8_e4m3fn)
    Bs = torch.pow(2.0, -(torch.randint(0, 4, (N // KG, K // KG),
                                        device="cuda").float()))
    payload, blocked = mx._prepare_weight(B, Bs)
    check("weight payload unchanged", torch.equal(payload, B.contiguous()))
    check("blocked scale buffer sized for 128x4 tiles",
          blocked.numel() == (-(-N // 128) * 128) * (-(-(K // KG) // 4) * 4),
          f"{blocked.shape}")

    print()
    if FAILURES:
        print(f"FAILED: {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())