"""Split-KV (flash-decoding) sparse attention for SM120.

The reference kernel grids over (m, b), so a decode step launches exactly b CTAs.
At batch 1 that is one CTA on a 188-SM RTX PRO 6000 -- measured 4.1 GB/s of KV
gather versus 1769 GB/s for the same kernel during prefill. Fitting shared memory
makes the kernel *run* on SM120 (see sparse_attn_sm120.py); this makes decode
*fast* by giving the GPU more than one CTA of work.

Each split computes a normalised partial output and its log-sum-exp:

    o_s   = sum_{j in s} exp(logit_j - m_s) v_j / l_s
    lse_s = m_s + log(l_s)

With M = max(max_s lse_s, sink), those recombine exactly:

    denom = sum_s exp(lse_s - M) + exp(sink - M)
    O     = sum_s o_s * exp(lse_s - M) / denom

because o_s * exp(lse_s - M) = sum_{j in s} exp(logit_j - M) v_j. The attn_sink
term belongs to the row, not to any split, so it enters once here. An empty split
keeps m_s = -1e30 and l_s = 0; clamping the divisor sends o_s to 0 and lse_s to
about -1e30, so it contributes nothing and an all-empty row yields zeros --
matching the reference kernel's convention.
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
    smem_bytes,
)

BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"

# Guards 0/0 in a split whose top-k slots are all empty.
_EPS = 1e-30


@tilelang.jit(pass_configs=pass_configs)
def sparse_attn_split_kernel(h: int, d: int, scale=None, block: int = 64,
                             splits: int = 8, threads: int = 256):
    """Normalised partial attention over one slice of the top-k list."""
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")
    topk = T.symbolic("topk")
    if scale is None:
        scale = (1.0 / d) ** 0.5

    num_stages = 2
    blocks_per_split = T.ceildiv(T.ceildiv(topk, block), splits)

    @T.prim_func
    def sparse_attn_split_kernel_(
        q: T.Tensor[(b, m, h, d), BF16],
        kv: T.Tensor[(b, n, d), BF16],
        o_partial: T.Tensor[(splits, b, m, h, d), FP32],
        lse_partial: T.Tensor[(splits, b, m, h), FP32],
        topk_idxs: T.Tensor[(b, m, topk), INT32],
    ):
        with T.Kernel(m, b, splits, threads=threads) as (bx, by, bz):
            q_shared = T.alloc_shared((h, d), BF16)
            kv_shared = T.alloc_shared((block, d), BF16)
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
            T.fill(scores_max, -1e30)
            T.copy(q[by, bx, :, :], q_shared)

            for t in T.Pipelined(blocks_per_split, num_stages=num_stages):
                for i in T.Parallel(block):
                    pos = (bz * blocks_per_split + t) * block + i
                    idxs[i] = T.if_then_else(pos < topk, topk_idxs[by, bx, pos], -1)
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

            for i, j in T.Parallel(h, d):
                o_partial[bz, by, bx, i, j] = acc_o[i, j] / T.max(sum_exp[i], _EPS)
            for i in T.Parallel(h):
                lse_partial[bz, by, bx, i] = scores_max[i] + T.log(T.max(sum_exp[i], _EPS))

    return sparse_attn_split_kernel_


@tilelang.jit(pass_configs=pass_configs)
def combine_splits_kernel(h: int, d: int, splits: int, threads: int = 256):
    """Merge normalised per-split partials by log-sum-exp, applying attn_sink once."""
    b = T.symbolic("b")
    m = T.symbolic("m")

    @T.prim_func
    def combine_splits_kernel_(
        o_partial: T.Tensor[(splits, b, m, h, d), FP32],
        lse_partial: T.Tensor[(splits, b, m, h), FP32],
        attn_sink: T.Tensor[(h,), FP32],
        o: T.Tensor[(b, m, h, d), BF16],
    ):
        with T.Kernel(m, b, threads=threads) as (bx, by):
            acc = T.alloc_fragment((h, d), FP32)
            lse_max = T.alloc_fragment(h, FP32)
            denom = T.alloc_fragment(h, FP32)
            w = T.alloc_fragment(h, FP32)

            T.clear(acc)
            # Fold the sink into the shared max so no term can overflow.
            for i in T.Parallel(h):
                lse_max[i] = attn_sink[i]
            for s in T.serial(splits):
                for i in T.Parallel(h):
                    lse_max[i] = T.max(lse_max[i], lse_partial[s, by, bx, i])
            for i in T.Parallel(h):
                denom[i] = T.exp(attn_sink[i] - lse_max[i])

            for s in T.serial(splits):
                for i in T.Parallel(h):
                    w[i] = T.exp(lse_partial[s, by, bx, i] - lse_max[i])
                    denom[i] += w[i]
                for i, j in T.Parallel(h, d):
                    acc[i, j] += o_partial[s, by, bx, i, j] * w[i]

            for i, j in T.Parallel(h, d):
                o[by, bx, i, j] = T.Cast(BF16, acc[i, j] / denom[i])

    return combine_splits_kernel_


@functools.lru_cache(maxsize=None)
def choose_split_config(h: int, d: int, topk: int, batch_rows: int,
                        device_index: int = 0):
    """Pick (block, splits) so decode fills the SMs without over-splitting."""
    budget = _smem_budget(device_index)
    block = next(bk for bk in BLOCK_CANDIDATES if smem_bytes(h, bk, d) <= budget)
    num_sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    tiles = max(1, -(-topk // block))
    # Once the (m, b) grid alone fills the machine, splitting only buys an extra
    # combine launch. Measured: batch 32 at topk=512 regresses 0.62x when split.
    if batch_rows >= num_sms // 2:
        return block, 1
    want = max(1, -(-num_sms // max(batch_rows, 1)))
    return block, max(1, min(tiles, want))


_WORKSPACE = {}


def _workspace(splits, b, s, h, d, device):
    """Reuse the fp32 partial buffers across calls.

    They are pure scratch -- written by every split before the combine reads
    them -- and reallocating 2 MB per call costs more than the kernels do at
    decode sizes, which is what hid the split win in the first measurement.
    """
    key = (splits, b, s, h, d, device)
    buf = _WORKSPACE.get(key)
    if buf is None:
        buf = (
            torch.empty((splits, b, s, h, d), dtype=torch.float32, device=device),
            torch.empty((splits, b, s, h), dtype=torch.float32, device=device),
        )
        _WORKSPACE[key] = buf
    return buf


def sparse_attn_split(q, kv, attn_sink, topk_idxs, softmax_scale,
                      block=None, splits=None):
    """Split-KV sparse attention. Same signature as kernel.sparse_attn."""
    b, s, h, d = q.size()
    q, attn_sink = pad_heads(q, attn_sink)
    h_pad = q.size(2)
    topk = topk_idxs.size(-1)

    auto_block, auto_splits = choose_split_config(
        h_pad, d, topk, b * s, q.device.index or 0
    )
    block = auto_block if block is None else block
    splits = auto_splits if splits is None else splits

    if splits == 1:
        # Single split is the base kernel plus a pointless combine launch.
        from sparse_attn_sm120 import sparse_attn as _base_sparse_attn

        o = _base_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, block)
        return o.narrow(2, 0, h).contiguous() if h < 16 else o

    o_partial, lse_partial = _workspace(splits, b, s, h_pad, d, q.device)

    sparse_attn_split_kernel(h_pad, d, softmax_scale, block, splits)(
        q, kv, o_partial, lse_partial, topk_idxs
    )
    o = torch.empty_like(q)
    combine_splits_kernel(h_pad, d, splits)(o_partial, lse_partial, attn_sink, o)

    if h < 16:
        o = o.narrow(2, 0, h).contiguous()
    return o
