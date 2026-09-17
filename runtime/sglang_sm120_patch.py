"""Use TileLang prefill with upstream FlashInfer decode for SGLang on SM120.

Install by putting this directory on PYTHONPATH with a sitecustomize that calls
`install()`, so every forked SGLang worker patches before its first forward:

    PYTHONPATH=/opt/dsv41/runtime
    # sitecustomize.py
    import sglang_sm120_patch; sglang_sm120_patch.install()

What it replaces
----------------
`sglang.kernels.ops.attention.flash_mla_sm120.flash_mla_with_kvcache_sm120`
is shared by prefill and decode. Batches within SGLang's
`SM120_DECODE_MAX_TOKENS` limit retain upstream FlashInfer decode; larger
batches use the TileLang paged kernel. This avoids the installed FlashInfer
prefill kernel's unsupported 8-head V4.1 configuration without replacing its
fast decode path. Set SGLANG_SM120_FLASHMLA_BACKEND=flashinfer at startup.

Both KV sources are handled: the sliding window and the compressed stream
(`extra_k_cache` / `extra_indices`) are attended separately and merged by
log-sum-exp, which is equivalent to one softmax over their union. That path is
why this exists -- SGLang's own SM120 fallback materialises the dequantised KV
in fp32 and dies at long context:

    _gather_and_dequant -> nope_fp8.view(...).float()
    torch.OutOfMemoryError: Tried to allocate 1.75 GiB   (183k-token prompt)

Unhandled TileLang inputs fall through to upstream unless DSV41_SM120_STRICT=1,
which raises instead. Intentional small-batch FlashInfer dispatch is not a
fallback. FlashInfer quantizes NoPE queries to FP8; TileLang uses BF16 queries,
so the two paths are not bit-identical. test_hybrid_attention.py checks both
sides of the dispatch boundary against their respective numerical contracts.

The serving image already reaches this entry point for sparse prefill. The
legacy PATCH-sglang-sparse-prefill.diff concerns older SGLang checkouts only.
"""
import logging
import os
import sys

import torch

_log = logging.getLogger("sglang_sm120_patch")
_INSTALLED = False

# page_size is not passed to flash_mla_with_kvcache_sm120; it is recoverable
# from the k_cache view the backend hands over, which is
# [num_pages, page_size, 1, bytes_per_token].
_DEFAULT_PAGE_SIZE = 256


def _is_sm120() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] == 12


def _as_arena_u8(k_cache: torch.Tensor):
    """Return ([num_pages, page_bytes] uint8 contiguous, page_size).

    The backend's k_cache is a possibly-strided view over the raw page buffer,
    so recover the real page stride rather than trusting the logical shape.
    """
    page_size = k_cache.shape[1] if k_cache.ndim >= 3 else _DEFAULT_PAGE_SIZE
    page_bytes = k_cache.stride(0)
    num_pages = k_cache.shape[0]
    raw = k_cache.as_strided((num_pages, page_bytes), (page_bytes, 1))
    if raw.dtype != torch.uint8:
        raw = raw.view(torch.uint8)
    return raw.contiguous(), page_size


def _wrap(original, *, decode_max_tokens):
    from sparse_attn_paged_dual_sm120 import sparse_attn_paged_dual
    from sparse_attn_paged_sm120 import page_bytes_for

    def flash_mla_with_kvcache_sm120(**kwargs):
        # Match upstream's decode/prefill dispatch boundary. FlashInfer supports
        # these small batches; old prefill dispatch lacks V4.1's extra page sizes.
        # Keep the fused decode path instead of replacing it with TileLang.
        if kwargs["q"].shape[0] <= decode_max_tokens:
            return original(**kwargs)
        strict = os.environ.get("DSV41_SM120_STRICT", "0") == "1"
        reason = None
        if not _is_sm120():
            reason = "not an SM120 device"

        if reason is None:
            try:
                q = kwargs["q"]
                arena, page_size = _as_arena_u8(kwargs["k_cache"])
                if arena.shape[1] != page_bytes_for(page_size):
                    raise ValueError(
                        f"page stride {arena.shape[1]} != {page_bytes_for(page_size)} "
                        f"expected for page_size={page_size}"
                    )
                # Backend layout is [num_tokens, (1,) heads, d]; the kernel wants
                # [b, m, h, d] with one query row per token.
                q4 = q.unsqueeze(1) if q.ndim == 3 else q
                idx = kwargs["indices"]
                idx3 = idx.unsqueeze(1) if idx.ndim == 2 else idx
                sink = kwargs.get("attn_sink")
                if sink is None:
                    sink = torch.full((q4.shape[2],), -float("inf"),
                                      dtype=torch.float32, device=q4.device)
                scale = kwargs.get("softmax_scale") or q4.shape[-1] ** -0.5

                # Second KV source: the compressed stream alongside the window.
                extra = kwargs.get("extra_k_cache")
                arena_x = idx_x = None
                ps_x = None
                if extra is not None:
                    arena_x, ps_x = _as_arena_u8(extra)
                    if arena_x.shape[1] != page_bytes_for(ps_x):
                        raise ValueError(
                            f"extra page stride {arena_x.shape[1]} != "
                            f"{page_bytes_for(ps_x)} for page_size={ps_x}"
                        )
                    idx_x = kwargs.get("extra_indices_in_kvcache")
                    if idx_x is None:
                        raise ValueError("extra_k_cache given without extra_indices")
                    if idx_x.ndim == 2:
                        idx_x = idx_x.unsqueeze(1)

                o = sparse_attn_paged_dual(
                    q4, arena, idx3, sink, float(scale),
                    arena_extra=arena_x, idx_extra=idx_x,
                    topk_length=kwargs.get("topk_length"),
                    extra_topk_length=kwargs.get("extra_topk_length"),
                    page_size=page_size, extra_page_size=ps_x,
                )
                return (o if q.ndim == 4 else o.squeeze(1)), None
            except Exception as exc:
                if strict:
                    raise
                reason = f"{type(exc).__name__}: {exc}"

        if strict:
            raise RuntimeError(f"sglang_sm120_patch declined: {reason}")
        _log.debug("sglang_sm120_patch falling through to upstream: %s", reason)
        return original(**kwargs)

    flash_mla_with_kvcache_sm120.__wrapped__ = original
    return flash_mla_with_kvcache_sm120


def install() -> bool:
    """Patch SGLang's SM120 entry point. Idempotent; returns whether it patched."""
    global _INSTALLED
    if _INSTALLED:
        return True
    if not _is_sm120():
        _log.info("sglang_sm120_patch: not SM120, nothing to do.")
        return False

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from sglang.kernels.ops.attention import flash_mla_sm120 as mod
    except ImportError as exc:
        _log.warning("sglang_sm120_patch: sglang not importable (%s)", exc)
        return False

    original = getattr(mod, "flash_mla_with_kvcache_sm120", None)
    if original is None or getattr(original, "__wrapped__", None) is not None:
        return False
    mod.flash_mla_with_kvcache_sm120 = _wrap(
        original, decode_max_tokens=mod.SM120_DECODE_MAX_TOKENS
    )
    _INSTALLED = True
    _log.info("sglang_sm120_patch: TileLang prefill installed; upstream decode retained.")
    return True
