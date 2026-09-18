"""Fused per-32-group FP8 activation quantizer emitting the cuBLASLt MX scale layout.

One pass over the activations computes, for each (row, 32-wide group):
  * the power-of-two (UE8M0) scale ``2^ceil(log2(amax/448))`` -- the recipe
    SGLang/DeepGEMM use for Blackwell block-FP8 with ``scale_ue8m0=True``;
  * the e4m3 payload ``round(x / scale)``;
  * the scale byte written DIRECTLY into the 128x4-tiled, 32x16-swizzled MX
    layout cuBLASLt's ``CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0`` expects.

Doing the permutation in-kernel matters: the equivalent torch op sequence
(clamp/ceil/pad/permute/reshape) costs a flat ~26 us of launch overhead -- more
than several of the GEMMs it feeds (5120x288 is 9 us).

Blocked index (verified exhaustively against the CUTLASS reference layout for
rows in {128,256,384,512,2048,4096,5120} x cols in {4,8,16,32,160,192}):

    idx = (r//128 * cb + c//4) * 512
        + (r%128 % 32) * 16 + (r%128 // 32) * 4 + (c % 4)

where cb = ceil(cols/4) and the tile is 128 rows x 4 groups, contiguous after
padding. ``512 == 128 rows * 4 groups`` is the per-tile element count.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_FP8_MAX = tl.constexpr(448.0)


@triton.jit
def _quant_ue8m0_blocked_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    M,
    K: tl.constexpr,
    GROUPS: tl.constexpr,
    CB: tl.constexpr,          # ceil(GROUPS / 4) column tiles
    BLOCK_M: tl.constexpr,
):
    """x:(M,K) bf16 -> q:(M,K) e4m3 ; s: blocked UE8M0 bytes (rb*128 * cb*4)."""
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < M

    cols = pid_g * 32 + tl.arange(0, 32)
    x = tl.load(x_ptr + rows[:, None] * K + cols[None, :],
                mask=rmask[:, None], other=0.0).to(tl.float32)

    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-10)
    exponent = tl.minimum(tl.maximum(tl.ceil(tl.log2(amax / _FP8_MAX)), -127.0),
                          127.0)
    scale = tl.exp2(exponent)
    q = tl.clamp(x / scale[:, None], -_FP8_MAX, _FP8_MAX).to(tl.float8e4nv)
    tl.store(q_ptr + rows[:, None] * K + cols[None, :], q, mask=rmask[:, None])

    # blocked scale position
    r_loc = rows % 128
    idx = ((rows // 128) * CB + (pid_g // 4)) * 512 \
        + (r_loc % 32) * 16 + (r_loc // 32) * 4 + (pid_g % 4)
    tl.store(s_ptr + idx, (exponent.to(tl.int32) + 127).to(tl.uint8), mask=rmask)


def quantize_activation_mxfp8(x2d: torch.Tensor, out_q=None, out_s=None):
    """(M,K) bf16/fp16 -> (e4m3 (M,K), blocked UE8M0 scale bytes).

    ``out_q``/``out_s`` let callers reuse preallocated buffers so the call stays
    allocation-free inside a CUDA graph.
    """
    assert x2d.dim() == 2 and x2d.is_cuda, "expects a 2D CUDA tensor"
    m, k = x2d.shape
    assert k % 32 == 0, f"K={k} must be a multiple of 32"
    groups = k // 32
    cb = -(-groups // 4)
    x2d = x2d.contiguous()

    q = out_q if out_q is not None else torch.empty(
        (m, k), device=x2d.device, dtype=torch.float8_e4m3fn)
    s = out_s if out_s is not None else torch.zeros(
        (-(-m // 128) * 128) * (cb * 4), device=x2d.device, dtype=torch.uint8)

    BLOCK_M = 32
    _quant_ue8m0_blocked_kernel[(triton.cdiv(m, BLOCK_M), groups)](
        x2d, q, s, m, k, groups, cb, BLOCK_M=BLOCK_M, num_warps=4)
    return q, s