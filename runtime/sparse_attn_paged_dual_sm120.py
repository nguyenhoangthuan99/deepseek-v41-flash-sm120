"""Two-KV-source paged sparse attention for SM120.

DeepSeek-V4.1's attention reads two KV streams at once: the sliding window and
the compressed stream. DeepSeek's own reference concatenates them into a single
sparse_attn call (model.py:779 `kv = torch.cat([kv, compress_kv], dim=1)`);
SGLang keeps them in separate paged arenas and passes the second as
`extra_k_cache` / `extra_indices` / `extra_topk_length`.

sparse_attn_paged_sm120 handles one arena, so those calls fell through to
SGLang's own path -- which materialises the dequantised KV in fp32 and OOMs at
long context:

    _gather_and_dequant -> nope_fp8.view(...).float()
    torch.OutOfMemoryError: Tried to allocate 1.75 GiB   (183k-token prompt)

Attending the arenas separately and merging by log-sum-exp is exactly equivalent
to one softmax over their union, because with M = max(lse_main, lse_extra, sink):

    denom = exp(lse_main - M) + exp(lse_extra - M) + exp(sink - M)
    O     = (o_main * exp(lse_main - M) + o_extra * exp(lse_extra - M)) / denom

and o_src * exp(lse_src - M) = sum_{j in src} exp(logit_j - M) v_j. So this
reuses the combine kernel already validated for split-KV, with the two arenas
occupying the two slots.

Also adds `topk_length`, which the one-arena path ignored: SGLang marks the valid
prefix of each index row with a length rather than padding with -1, and honouring
only the -1 sentinel would attend to stale slots past the end.
"""
import functools

import torch
import tilelang
import tilelang.language as T

from sparse_attn_sm120 import _smem_budget, pad_heads, pass_configs
from sparse_attn_sm120_split import combine_splits_kernel
from sparse_attn_paged_sm120 import (
    D,
    NOPE_DIM,
    NOPE_ROPE_STRIDE,
    NUM_TILES,
    QUANT_TILE,
    ROPE_DIM,
    SCALE_STRIDE,
    _smem_bytes_paged,
    arena_views,
    ue8m0_lut,
    page_bytes_for,
)

FP8 = "float8_e4m3"
BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"
UINT8 = "uint8"

_EPS = 1e-30
BLOCK_CANDIDATES = (64, 32, 16)


@functools.lru_cache(maxsize=None)
def choose_block_dual(h, d, device_index=0):
    budget = _smem_budget(device_index)
    for block in BLOCK_CANDIDATES:
        if _smem_bytes_paged(h, block, d) <= budget:
            return block
    raise RuntimeError(f"no viable block for h={h} d={d} within {budget} B")


@tilelang.jit(pass_configs=pass_configs)
def paged_partial_kernel(h: int, d: int, page_size: int, page_bytes: int,
                         nslots: int, slot: int, scale=None, block: int = 64,
                         threads: int = 256):
    """Attend one paged arena, emitting a normalised partial plus its log-sum-exp.

    No attn_sink here -- it belongs to the row, so the combine applies it once
    across both arenas.
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
    def paged_partial_kernel_(
        q: T.Tensor[(b, m, h, d), BF16],
        arena_f8: T.Tensor[(num_pages, page_bytes), FP8],
        arena_bf: T.Tensor[(num_pages, half_bytes), BF16],
        arena_u8: T.Tensor[(num_pages, page_bytes), UINT8],
        topk_idxs: T.Tensor[(b, m, topk), INT32],
        topk_len: T.Tensor[(b, m), INT32],
        o_partial: T.Tensor[(nslots, b, m, h, d), FP32],
        lse_partial: T.Tensor[(nslots, b, m, h), FP32],
    ):
        with T.Kernel(m, b, threads=threads) as (bx, by):
            q_shared = T.alloc_shared((h, d), BF16)
            kv_shared = T.alloc_shared((block, d), BF16)
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
                    # A slot counts only inside the row's valid prefix; SGLang
                    # leaves stale indices past topk_len rather than clearing.
                    idxs[i] = T.if_then_else(
                        T.if_then_else(pos < topk, pos < topk_len[by, bx], False),
                        topk_idxs[by, bx, pos],
                        -1,
                    )
                    pages[i] = T.if_then_else(idxs[i] != -1, idxs[i] // page_size, 0)
                    slots[i] = T.if_then_else(idxs[i] != -1, idxs[i] % page_size, 0)

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

            for i, j in T.Parallel(h, d):
                o_partial[slot, by, bx, i, j] = acc_o[i, j] / T.max(sum_exp[i], _EPS)
            for i in T.Parallel(h):
                lse_partial[slot, by, bx, i] = scores_max[i] + T.log(T.max(sum_exp[i], _EPS))

    return paged_partial_kernel_


_WS = {}


def _workspace(nslots, b, m, h, d, device):
    key = (nslots, b, m, h, d, device)
    w = _WS.get(key)
    if w is None:
        w = (torch.empty((nslots, b, m, h, d), dtype=torch.float32, device=device),
             torch.empty((nslots, b, m, h), dtype=torch.float32, device=device))
        _WS[key] = w
    return w


_LEN_CACHE = {}


def _as_topk_len(topk_length, b, m, topk, device):
    """Normalise topk_length to an int32 [b, m] tensor; None means all valid."""
    if topk_length is None:
        key = (b, m, topk, device)
        t = _LEN_CACHE.get(key)
        if t is None:
            t = torch.full((b, m), topk, dtype=torch.int32, device=device)
            _LEN_CACHE[key] = t
        return t
    t = topk_length
    if t.dim() == 1:
        t = t.view(b, 1).expand(b, m)
    elif t.dim() == 3:
        t = t.view(b, m)
    return t.to(torch.int32).contiguous()


def sparse_attn_paged_dual(q, arena_main, idx_main, attn_sink, softmax_scale,
                           arena_extra=None, idx_extra=None,
                           topk_length=None, extra_topk_length=None,
                           page_size=256, extra_page_size=None, block=None):
    """Sparse attention over one or two paged FP8 KV arenas.

    With arena_extra None this is the single-source path; with it supplied the
    two arenas are attended separately and merged by log-sum-exp.
    """
    b, s, h, d = q.size()
    assert d == D, f"expected head_dim {D}, got {d}"
    q, attn_sink = pad_heads(q, attn_sink)
    h_pad = q.size(2)
    if block is None:
        block = choose_block_dual(h_pad, d, q.device.index or 0)

    sources = [(arena_main, idx_main, topk_length, page_size)]
    if arena_extra is not None:
        sources.append((arena_extra, idx_extra, extra_topk_length,
                        extra_page_size if extra_page_size else page_size))

    nslots = len(sources)
    o_partial, lse_partial = _workspace(nslots, b, s, h_pad, d, q.device)

    for slot, (arena, idx, tl, ps) in enumerate(sources):
        pb = arena.shape[1]
        assert pb == page_bytes_for(ps), (
            f"slot {slot}: page stride {pb} != {page_bytes_for(ps)} for page_size={ps}"
        )
        a_f8, a_bf, a_u8 = arena_views(arena)
        idx = idx.contiguous().to(torch.int32)
        topk = idx.shape[-1]
        tl_t = _as_topk_len(tl, b, s, topk, q.device)
        paged_partial_kernel(h_pad, d, ps, pb, nslots, slot, softmax_scale, block)(
            q, a_f8, a_bf, a_u8, idx, tl_t, o_partial, lse_partial
        )

    o = torch.empty_like(q)
    combine_splits_kernel(h_pad, d, nslots)(o_partial, lse_partial, attn_sink, o)
    return o.narrow(2, 0, h).contiguous() if h < 16 else o
