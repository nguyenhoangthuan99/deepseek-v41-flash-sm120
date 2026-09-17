"""Paged-FP8 sparse attention for SM120 -- the layout SGLang and vLLM actually serve.

sparse_attn_sm120.py operates on DeepSeek's reference KV layout: dense bf16
[b, n, d]. Serving engines instead keep a paged FP8 arena. Materialising the
dense form is not an option at prefill sizes -- m=2048 x topk=512 gathered rows
of 512 bf16 values is ~1 GB per layer -- so the gather and dequant are fused
into the attention kernel and land straight in shared memory.

DSv4-Flash page layout (MODEL1), matching
sglang/kernels/ops/attention/flash_mla_sm120.py:

    nope_dim=448, rope_dim=64, quant tile=64, page_size=256
    per page, as bytes:
      [0, page_size*576)                  nope+rope, 576 B per token:
          [slot*576,       +448)          448x FP8 e4m3   (NoPE)
          [slot*576 + 448, +128)          64x BF16        (RoPE)
      [page_size*576, +page_size*8)       scales, 8 B per token:
          7x UE8M0 (one per 64 NoPE channels) + 1 pad
    page_bytes = ceil(page_size*584 / 576) * 576   # 149760 at page_size=256

Dequant is nope[t*64:(t+1)*64] * 2**(scale_byte - 127).

The arena is handed to the kernel as three contiguous views of one buffer -- FP8
for NoPE, BF16 for RoPE, UE8M0 for the scales -- so nothing is copied and no
strided tensor is required.
"""
import functools

import torch
import tilelang
import tilelang.language as T

from sparse_attn_sm120 import (
    BLOCK_CANDIDATES,
    _smem_budget,
    pad_heads,
    pass_configs,
)

FP8 = "float8_e4m3"
BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"
UINT8 = "uint8"

NOPE_DIM = 448
ROPE_DIM = 64
QUANT_TILE = 64
NUM_TILES = NOPE_DIM // QUANT_TILE  # 7
SCALE_STRIDE = NUM_TILES + 1  # 8
NOPE_ROPE_STRIDE = NOPE_DIM + ROPE_DIM * 2  # 576
BYTES_PER_TOKEN = NOPE_DIM + ROPE_DIM * 2 + SCALE_STRIDE  # 584
D = NOPE_DIM + ROPE_DIM  # 512


def page_bytes_for(page_size: int) -> int:
    """Bytes per page, rounded up to a 576-byte multiple (SGLang's convention)."""
    return -(-(page_size * BYTES_PER_TOKEN) // NOPE_ROPE_STRIDE) * NOPE_ROPE_STRIDE


def arena_views(arena_u8: torch.Tensor):
    """Three contiguous dtype views of one [num_pages, page_bytes] uint8 arena."""
    assert arena_u8.dtype == torch.uint8 and arena_u8.is_contiguous()
    num_pages, pb = arena_u8.shape
    assert pb % 2 == 0, "page_bytes must be even to alias the RoPE section as bf16"
    return (
        arena_u8.view(torch.float8_e4m3fn),
        arena_u8.view(torch.bfloat16),
        arena_u8,
    )


_LUT_CACHE = {}


def ue8m0_lut(device):
    """256-entry table of 2**(b-127), the UE8M0 scale each byte encodes.

    A lookup replaces the reinterpret(b << 23) bit-trick: exact to the same bits,
    but TVM 0.1.12 (what the SGLang dev-dsv41 image ships) fails to vectorise the
    shift -- "Check failed: (value.dtype().is_scalar())" in Broadcast. The table
    is 1 KB and stays in cache.
    """
    t = _LUT_CACHE.get(device)
    if t is None:
        b = torch.arange(256, dtype=torch.float32, device=device)
        t = torch.pow(2.0, b - 127.0)
        _LUT_CACHE[device] = t
    return t


def _smem_bytes_paged(h: int, block: int, d: int) -> int:
    """Dense kernel's footprint plus the staged per-tile scale tile."""
    return 2 * h * d + 2 * block * d + 2 * h * block + 4 * block * NUM_TILES + 2048


@functools.lru_cache(maxsize=None)
def choose_block_paged(h: int, d: int, device_index: int = 0) -> int:
    budget = _smem_budget(device_index)
    for block in BLOCK_CANDIDATES:
        if _smem_bytes_paged(h, block, d) <= budget:
            return block
    raise RuntimeError(
        f"paged sparse_attn cannot fit: h={h} d={d} needs "
        f"{_smem_bytes_paged(h, BLOCK_CANDIDATES[-1], d)} B > {budget} B available."
    )


@tilelang.jit(pass_configs=pass_configs)
def sparse_attn_paged_kernel(h: int, d: int, page_size: int, page_bytes: int,
                             scale=None, block: int = 64, threads: int = 256):
    """Sparse attention reading a paged FP8 arena directly.

    Identical online-softmax structure to the dense kernel; only the kv_shared
    fill changes, gathering and dequantising in place.
    """
    b = T.symbolic("b")
    m = T.symbolic("m")
    num_pages = T.symbolic("num_pages")
    topk = T.symbolic("topk")
    if scale is None:
        scale = (1.0 / d) ** 0.5

    num_stages = 2
    num_blocks = tilelang.cdiv(topk, block)
    scale_section = page_size * NOPE_ROPE_STRIDE
    half_bytes = page_bytes // 2

    @T.prim_func
    def sparse_attn_paged_kernel_(
        q: T.Tensor[(b, m, h, d), BF16],
        arena_f8: T.Tensor[(num_pages, page_bytes), FP8],
        arena_bf: T.Tensor[(num_pages, half_bytes), BF16],
        arena_u8: T.Tensor[(num_pages, page_bytes), UINT8],
        o: T.Tensor[(b, m, h, d), BF16],
        attn_sink: T.Tensor[(h,), FP32],
        topk_idxs: T.Tensor[(b, m, topk), INT32],
    ):
        with T.Kernel(m, b, threads=threads) as (bx, by):
            q_shared = T.alloc_shared((h, d), BF16)
            kv_shared = T.alloc_shared((block, d), BF16)
            o_shared = T.alloc_shared((h, d), BF16)
            acc_s_cast = T.alloc_shared((h, block), BF16)
            scales = T.alloc_shared((block, NUM_TILES), FP32)

            idxs = T.alloc_fragment(block, INT32)
            pages = T.alloc_fragment(block, INT32)
            slots = T.alloc_fragment(block, INT32)
            acc_s = T.alloc_fragment((h, block), FP32)
            acc_o = T.alloc_fragment((h, d), FP32)
            scores_max = T.alloc_fragment(h, FP32)
            scores_max_prev = T.alloc_fragment(h, FP32)
            scores_scale = T.alloc_fragment(h, FP32)
            scores_sum = T.alloc_fragment(h, FP32)
            sum_exp = T.alloc_fragment(h, FP32)

            T.clear(acc_o)
            T.clear(sum_exp)
            T.fill(scores_max, -1e30)
            T.copy(q[by, bx, :, :], q_shared)

            for t in T.Pipelined(num_blocks, num_stages=num_stages):
                for i in T.Parallel(block):
                    pos = t * block + i
                    idxs[i] = T.if_then_else(pos < topk, topk_idxs[by, bx, pos], -1)
                    pages[i] = T.if_then_else(idxs[i] != -1, idxs[i] // page_size, 0)
                    slots[i] = T.if_then_else(idxs[i] != -1, idxs[i] % page_size, 0)

                # UE8M0 byte v encodes 2**(v-127); reinterpreting v<<23 as fp32
                # builds that exponent directly, with no exp2 call. Staged in
                # shared memory rather than re-read per channel: the inline form
                # reloads the same byte 64 times and measured 360us vs 130us.
                for i, tt in T.Parallel(block, NUM_TILES):
                    # UE8M0 byte v encodes 2**(v-127). exp2 on the widened
                    # byte is exact for integer exponents and, unlike the
                    # reinterpret(v << 23) bit-trick, vectorises on the TVM in
                    # the dev-dsv41 image (0.1.12 fails that shift with
                    # "Check failed: (value.dtype().is_scalar())"). A LUT gather
                    # works too but breaks the software pipeline, since it would
                    # depend on pages/slots from the same stage.
                    scales[i, tt] = T.exp2(
                        T.Cast(FP32, arena_u8[pages[i],
                               scale_section + slots[i] * SCALE_STRIDE + tt]) - 127.0
                    )

                for i, j in T.Parallel(block, d):
                    kv_shared[i, j] = T.if_then_else(
                        idxs[i] == -1,
                        T.Cast(BF16, 0),
                        T.if_then_else(
                            j < NOPE_DIM,
                            T.Cast(
                                BF16,
                                T.Cast(FP32, arena_f8[pages[i],
                                                      slots[i] * NOPE_ROPE_STRIDE + j])
                                * scales[i, j // QUANT_TILE],
                            ),
                            arena_bf[pages[i],
                                     slots[i] * (NOPE_ROPE_STRIDE // 2)
                                     + (NOPE_DIM // 2) + (j - NOPE_DIM)],
                        ),
                    )

                for i, j in T.Parallel(h, block):
                    acc_s[i, j] = T.if_then_else(idxs[j] != -1, 0, -T.infinity(FP32))
                T.gemm(q_shared, kv_shared, acc_s, transpose_B=True,
                       policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(h, block):
                    acc_s[i, j] *= scale
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(h):
                    scores_scale[i] = T.exp(scores_max_prev[i] - scores_max[i])
                for i, j in T.Parallel(h, block):
                    acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(h):
                    sum_exp[i] = sum_exp[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)
                for i, j in T.Parallel(h, d):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(acc_s_cast, kv_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i in T.Parallel(h):
                sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])
            for i, j in T.Parallel(h, d):
                acc_o[i, j] /= sum_exp[i]
            T.copy(acc_o, o_shared)
            T.copy(o_shared, o[by, bx, :, :])

    return sparse_attn_paged_kernel_


def sparse_attn_paged(q, arena_u8, attn_sink, topk_idxs, softmax_scale,
                      page_size=256, block=None):
    """Sparse attention over a paged FP8 KV arena.

    q:         [b, m, h, 512] bf16
    arena_u8:  [num_pages, page_bytes] uint8, contiguous
    topk_idxs: [b, m, topk] int32, token-level indices, -1 = empty slot
    """
    b, s, h, d = q.size()
    assert d == D, f"expected head_dim {D}, got {d}"
    q, attn_sink = pad_heads(q, attn_sink)
    h_pad = q.size(2)

    num_pages, pb = arena_u8.shape
    assert pb == page_bytes_for(page_size), (
        f"arena page stride {pb} != expected {page_bytes_for(page_size)} "
        f"for page_size={page_size}"
    )
    a_f8, a_bf, a_u8 = arena_views(arena_u8)

    if block is None:
        block = choose_block_paged(h_pad, d, q.device.index or 0)
    o = torch.empty_like(q)
    sparse_attn_paged_kernel(h_pad, d, page_size, pb, softmax_scale, block)(
        q, a_f8, a_bf, a_u8, o, attn_sink, topk_idxs
    )
    return o.narrow(2, 0, h).contiguous() if h < 16 else o
