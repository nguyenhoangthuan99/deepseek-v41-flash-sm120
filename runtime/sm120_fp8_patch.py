"""SM120 small-batch block-FP8 GEMM using SGLang's existing split-K kernel.

The deployed kernel's historical ``_hopper`` name is not an ISA restriction:
it uses ordinary Triton loads and tl.dot, not Hopper TMA/WGMMA. This adapter
selects measured SM120 launch configurations, accumulates split results in
FP32, and retains the original implementation outside the tuned domain.
"""

import functools
import json
import logging
import os
from pathlib import Path

import torch
import triton

from sglang.kernels.ops.quantization import fp8_kernel as kernels

_LOG = logging.getLogger("sm120_fp8_patch")
_INSTALLED = False


def matmul(A, B, As, Bs, block_size, output_dtype=torch.bfloat16, *, config):
    """Block-scaled A @ B.T; FP32 partials are reduced before output conversion."""
    if As.dtype != torch.float32 or Bs.dtype != torch.float32:
        raise TypeError("SM120 split-K GEMM requires FP32 block scales")
    if output_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("SM120 split-K GEMM supports BF16, FP16, and FP32 outputs")
    bk = config["BLOCK_SIZE_K"]
    if bk > block_size[1] or block_size[1] % bk:
        raise ValueError(
            "K tiles must divide the quantization group without crossing scales"
        )
    splits = config["SPLIT_K"]
    if splits < 1 or splits & (splits - 1):
        raise ValueError("SPLIT_K must be a positive power of two")

    M, N, K, out = kernels.prepare_block_fp8_matmul_inputs(
        A, B, As, Bs, block_size, output_dtype
    )
    if M == 0 or N == 0:
        return out
    scales_a = As.view(M, As.shape[-1])
    partials = (
        torch.empty((splits, M, N), dtype=torch.float32, device=A.device)
        if splits > 1
        else out
    )
    grid = (
        triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),
        splits,
    )
    kernels._w8a8_block_fp8_matmul_hopper[grid](
        A,
        B,
        partials,
        scales_a,
        Bs,
        M,
        N,
        K,
        block_size[0],
        block_size[1],
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        out.stride(-2),
        out.stride(-1),
        scales_a.stride(0),
        scales_a.stride(1),
        Bs.stride(1),
        Bs.stride(0),
        **config,
        needs_masking=bool(K % bk),
    )
    if splits > 1:
        kernels._reduce_block_fp8_split_k[(triton.cdiv(M * N, 256),)](
            partials,
            out,
            M * N,
            splits,
            256,
        )
    return out


def install():
    """Install measured configs for this SM120 serving stack, once per process."""
    global _INSTALLED
    if _INSTALLED:
        return True
    if os.environ.get("DSV41_SM120_FP8_DISABLE", "1") == "1":
        return False
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        return False

    with Path(__file__).with_name("fp8_sm120_configs.json").open() as source:
        table = json.load(source)
    if table["device_name"] != torch.cuda.get_device_name():
        _LOG.info("No measured FP8 launch configurations for this device")
        return False
    configurations = {
        (entry["N"], entry["K"], *entry["block_size"]): {
            int(m): cfg for m, cfg in entry["configs"].items()
        }
        for entry in table["shapes"]
    }
    original = kernels.w8a8_block_fp8_matmul_triton

    @functools.lru_cache(maxsize=None)
    def choose(M, N, K, block_n, block_k):
        configs = configurations.get((N, K, block_n, block_k))
        if not configs or M < min(configs) or M > max(configs):
            return None
        return configs[min(configs, key=lambda m: abs(m - M))]

    @functools.wraps(original)
    def optimized(A, B, As, Bs, block_size, output_dtype=torch.float16):
        M = A.numel() // A.shape[-1]
        config = choose(M, B.shape[0], B.shape[1], *block_size)
        if (
            config is None
            or As.dtype != torch.float32
            or Bs.dtype != torch.float32
            or A.dtype != torch.float8_e4m3fn
            or B.dtype != torch.float8_e4m3fn
            or output_dtype not in (torch.bfloat16, torch.float16, torch.float32)
        ):
            return original(A, B, As, Bs, block_size, output_dtype)
        return matmul(A, B, As, Bs, block_size, output_dtype, config=config)

    # The universal kernel entry point looks up this global at call time;
    # fp8_utils also imports it by value and may already be loaded at startup.
    # Publish the pre-wrap kernel so the MXFP8 engine can chain tuned->MXFP8->
    # triton without double-applying the tuned configs.
    optimized.__sm120_tuned__ = original
    kernels.w8a8_block_fp8_matmul_triton = optimized
    from sglang.srt.layers.quantization import fp8_utils

    fp8_utils.w8a8_block_fp8_matmul_triton = optimized
    _INSTALLED = True
    _LOG.info(
        "SM120 tuned split-K FP8 GEMM installed for %d weight shapes",
        len(configurations),
    )
    return True
