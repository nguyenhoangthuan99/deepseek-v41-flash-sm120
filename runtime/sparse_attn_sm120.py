"""SM120-adaptive sparse attention for DeepSeek-V4.1-Flash.

DeepSeek's reference kernel (inference/kernel.py) hardcodes the KV tile at
block=64, which was chosen for SM90/SM100. Those parts expose 227 KB of opt-in
dynamic shared memory per block; SM120 (RTX PRO 6000 Blackwell) exposes 99 KB:

    shared_memory_per_block_optin = 101376 bytes

So the stock kernel lowers fine for sm_120 and then dies at launch with
"Failed to set the allowed dynamic shared memory size to 141312" whenever the
per-rank head count is large. That single limit -- not a missing kernel -- is
what "SM120 unsupported" means for this op.

The kernel body is arithmetically independent of `block`: it strides the top-k
list in tiles and masks the tail via `t * block + i < topk`. So the fix is to
pick the largest tile that fits the device budget instead of assuming 64.

Measured shared-memory model (bytes), exact against observed launch requests:

    smem(h, block, d) = 2*h*d      # q_shared, aliased with o_shared
                      + 2*block*d  # kv_shared
                      + 2*h*block  # acc_s_cast
                      + 2048       # barriers / alignment slack

    h=64, block=64, d=512 -> 65536 + 65536 + 8192 + 2048 = 141312  (observed)
    h=16, block=64, d=512 -> 16384 + 65536 + 2048 + 2048 =  86016  (fits)

Usage -- drop-in for kernel.sparse_attn:

    import sparse_attn_sm120
    sparse_attn_sm120.install()   # monkeypatches kernel.sparse_attn
"""
# No `from __future__ import annotations` here: TileLang resolves the T.Tensor
# annotations via get_type_hints() against module globals, and PEP 563 would
# stringify them so the closed-over shape vars (b, n, d, ...) fail to resolve.
import functools
from typing import Optional

import torch
import tilelang
import tilelang.language as T

BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"

# Mirrors the reference kernel's pass configs.
pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}

SMEM_SLACK = 2048
# Powers of two only. T.gemm's FullRow policy partitions the tile across 8 warps
# and rejects shapes it cannot cover evenly -- block=48 fails to lower at h=32
# ("No valid warp partition ... M=32, N=48"), so multiples of 16 are not enough.
BLOCK_CANDIDATES = (64, 32, 16)


def smem_bytes(h: int, block: int, d: int) -> int:
    """Dynamic shared memory the kernel will request, in bytes."""
    return 2 * h * d + 2 * block * d + 2 * h * block + SMEM_SLACK


@functools.lru_cache(maxsize=None)
def _smem_budget(device_index: int = 0) -> int:
    props = torch.cuda.get_device_properties(device_index)
    # Present as shared_memory_per_block_optin on modern torch; fall back to the
    # conservative per-block figure on builds that don't expose the opt-in value.
    budget = getattr(props, "shared_memory_per_block_optin", None)
    if budget is None:
        budget = props.shared_memory_per_block
    return int(budget)


@functools.lru_cache(maxsize=None)
def choose_block(h: int, d: int, device_index: int = 0) -> int:
    """Largest 16-aligned KV tile whose shared-memory request fits the device."""
    budget = _smem_budget(device_index)
    for block in BLOCK_CANDIDATES:
        if smem_bytes(h, block, d) <= budget:
            return block
    raise RuntimeError(
        f"sparse_attn cannot fit on this device: h={h} d={d} needs at least "
        f"{smem_bytes(h, BLOCK_CANDIDATES[-1], d)} bytes of shared memory but only "
        f"{budget} are available. Reduce heads per rank (raise tensor parallelism)."
    )


@tilelang.jit(pass_configs=pass_configs)
def sparse_attn_kernel(h: int, d: int, scale=None, block: int = 64, threads: int = 256):
    """DeepSeek's sparse_attn_kernel with the KV tile and thread count exposed.

    Body is unchanged from inference/kernel.py; only `block` and `threads`
    become parameters so a caller can fit the SM120 shared-memory budget.
    """
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")
    topk = T.symbolic("topk")
    if scale is None:
        scale = (1.0 / d) ** 0.5

    num_stages = 2
    num_blocks = tilelang.cdiv(topk, block)

    @T.prim_func
    def sparse_attn_kernel_(
        q: T.Tensor[(b, m, h, d), BF16],
        kv: T.Tensor[(b, n, d), BF16],
        o: T.Tensor[(b, m, h, d), BF16],
        attn_sink: T.Tensor[(h,), FP32],
        topk_idxs: T.Tensor[(b, m, topk), INT32],
    ):
        with T.Kernel(m, b, threads=threads) as (bx, by):
            q_shared = T.alloc_shared((h, d), BF16)
            kv_shared = T.alloc_shared((block, d), BF16)
            o_shared = T.alloc_shared((h, d), BF16)
            acc_s_cast = T.alloc_shared((h, block), BF16)

            idxs = T.alloc_fragment(block, INT32)
            acc_s = T.alloc_fragment((h, block), FP32)
            acc_o = T.alloc_fragment((h, d), FP32)
            scores_max = T.alloc_fragment(h, FP32)
            scores_max_prev = T.alloc_fragment(h, FP32)
            scores_scale = T.alloc_fragment(h, FP32)
            scores_sum = T.alloc_fragment(h, FP32)
            sum_exp = T.alloc_fragment(h, FP32)

            T.clear(acc_o)
            T.clear(sum_exp)
            # Finite lower bound instead of -inf: an all-empty row would otherwise
            # produce exp(-inf - (-inf)) = NaN. Zeros match the training kernel.
            T.fill(scores_max, -1e30)
            T.copy(q[by, bx, :, :], q_shared)

            for t in T.Pipelined(num_blocks, num_stages=num_stages):
                for i in T.Parallel(block):
                    idxs[i] = T.if_then_else(
                        t * block + i < topk, topk_idxs[by, bx, t * block + i], -1
                    )
                for i, j in T.Parallel(block, d):
                    kv_shared[i, j] = T.if_then_else(idxs[i] != -1, kv[by, idxs[i], j], 0)
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

    return sparse_attn_kernel_


_PAD_CACHE = {}


def pad_heads(q, attn_sink):
    """Pad the head dim up to 16 into cached buffers.

    T.gemm's FullRow policy needs each of the 8 warps to own a multiple of 16
    rows, so h < 16 cannot lower directly -- hence the reference kernel's pad.
    Doing it with torch.cat allocates on every call, which at TP=8 (h=8) costs
    more than the attention kernel itself; reuse one buffer and copy instead.
    The pad region is zeroed once and stays zero.
    """
    b, s, h, d = q.size()
    if h >= 16:
        return q, attn_sink
    key = (b, s, d, q.dtype, q.device)
    buf = _PAD_CACHE.get(key)
    if buf is None:
        buf = (
            q.new_zeros(b, s, 16, d),
            attn_sink.new_zeros(16),
        )
        _PAD_CACHE[key] = buf
    q_pad, sink_pad = buf
    q_pad[:, :, :h, :].copy_(q)
    sink_pad[:h].copy_(attn_sink)
    return q_pad, sink_pad


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    block: Optional[int] = None,
) -> torch.Tensor:
    """Drop-in replacement for kernel.sparse_attn with an SM120-aware KV tile."""
    b, s, h, d = q.size()
    q, attn_sink = pad_heads(q, attn_sink)
    h_padded = q.size(2)
    if block is None:
        block = choose_block(h_padded, d, q.device.index or 0)
    o = torch.empty_like(q)
    kernel = sparse_attn_kernel(h_padded, d, softmax_scale, block)
    kernel(q, kv, o, attn_sink, topk_idxs)
    if h < 16:
        o = o.narrow(2, 0, h).contiguous()
    return o


def install() -> None:
    """Monkeypatch kernel.sparse_attn so the reference model runs on SM120."""
    import kernel

    kernel.sparse_attn = sparse_attn
    try:
        import model

        model.sparse_attn = sparse_attn
    except ImportError:
        pass
