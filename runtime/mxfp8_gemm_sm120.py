"""SM120 cuBLASLt MXFP8 dense-GEMM engine for DeepSeek-V4.1-Flash block-FP8 linears.

Why this is exact: the checkpoint's block-FP8 weights use `[32, 32]` blocks with
`scale_fmt = ue8m0` (power-of-two scales). A 32x32 UE8M0 block expands LOSSLESSLY
into 1x32 (MX) groups -- one group scale per row of 32 weights -- so cuBLASLt's
native MXFP8 GEMM (``CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0``) computes the
*same product* as the serving W8A8 block-FP8 kernel. Measured rel-L2 vs fp32 is
1.66e-3, identical to the tuned Triton kernel (both are the bf16-output floor).

Dispatch: only when the measured benchmark shows MXFP8 faster at this (N, K, M).
`mxfp8_crossover.json` holds the per-shape M threshold; below it the tuned Triton
split-K kernel wins and we keep it.

Enable with DSV41_SM120_MXFP8=1 in addition to DSV41_SM120_FP8_DISABLE=0. It is
installed by ``sitecustomize`` after ``sm120_fp8_patch`` so it can wrap the tuned
kernel rather than replace it. Every failure path falls back to Triton.

Activation quantization matches the serving contract: per-32-group e4m3 payload
with **UE8M0 (power-of-two) scales**, which is what the vLLM/SGLang
``sglang_per_token_group_quant_fp8(scale_ue8m0=True)`` path already produces for
Blackwell block-FP8 (the same recipe DeepGEMM uses on SM100/120).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import ctypes
from pathlib import Path

import torch

_LOG = logging.getLogger("mxfp8_gemm_sm120")
_INSTALLED = False
_DISABLED = False
_LIB = None
_KG = 32
_FUSED_QUANT = None
# Only route calls with at least this many input rows to cuBLASLt MXFP8.
# Prefill chunks are 1024-4096 rows; decode is ~1-128. Override with
# DSV41_SM120_MXFP8_MIN_M (0 = always use MXFP8, the isolated-benchmark setting).
MIN_M_MXFP8 = int(os.environ.get("DSV41_SM120_MXFP8_MIN_M", "512"))


# --------------------------------------------------------------------------- #
# cuBLASLt wrapper (JIT-compiled once; cached under runtime/build)
# --------------------------------------------------------------------------- #
def _build_library() -> ctypes.CDLL | None:
    here = Path(__file__).resolve().parent
    src = here / "cublaslt_mxfp8.cu"
    if not src.is_file():
        _LOG.info("MXFP8 CUDA source not packaged; engine disabled")
        return None
    lib = here / "build" / "cublaslt_mxfp8.so"
    lib.parent.mkdir(exist_ok=True)
    if not lib.is_file():
        nvcc = os.environ.get("NVCC", "nvcc")
        cmd = [nvcc, "-O3", "-std=c++17", "--shared", "-Xcompiler", "-fPIC",
               "-gencode", "arch=compute_120a,code=sm_120a", str(src),
               "-lcublasLt", "-o", str(lib)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            _LOG.warning("MXFP8 nvcc build failed: %s", proc.stderr[-400:])
            return None
    dll = ctypes.CDLL(str(lib))
    dll.mxfp8_gemm.restype = ctypes.c_int
    dll.mxfp8_gemm.argtypes = ([ctypes.c_void_p] * 5 + [ctypes.c_int] * 3
                               + [ctypes.c_void_p])
    dll.last_error.restype = ctypes.c_char_p
    return dll


def _to_blocked(x: torch.Tensor) -> torch.Tensor:
    """cuBLASLt/CUTLASS MX scale layout: 128x4 tiles, inner 32x16 swizzle."""
    rows, cols = x.shape
    rb, cb = -(-rows // 128), -(-cols // 4)
    padded = torch.zeros(rb * 128, cb * 4, dtype=x.dtype, device=x.device)
    padded[:rows, :cols] = x
    blocks = padded.view(rb, 128, cb, 4).permute(0, 2, 1, 3)
    return (blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1).contiguous())


# --------------------------------------------------------------------------- #
# Weight-scale cache: 32x32 UE8M0 blocks -> 1x32 row-major blocked layout
# --------------------------------------------------------------------------- #
_WEIGHT_CACHE: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _prepare_weight(weight: torch.Tensor, weight_scale: torch.Tensor):
    """Return (fp8 payload (N,K), blocked UE8M0 scales) for cuBLASLt.

    `weight` is E4M3 (N, K); `weight_scale` is the checkpoint's (N/32, K/32)
    UE8M0 block scale. Expanding each block to its 32 rows yields per-row 1x32
    groups that MXFP8 consumes directly -- the payload is unchanged, so this is
    a layout change only, computed once per weight tensor.
    """
    key = (weight.data_ptr(), weight.shape[0], weight.shape[1])
    cached = _WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    # (N/32, K/32) -> (N, K/32): every row of a 32-row block shares the scale.
    rows = weight_scale.repeat_interleave(32, dim=0)
    if rows.shape[0] != weight.shape[0]:
        rows = rows[: weight.shape[0]]
    # UE8M0 stored as exponent+127 byte; scale bytes are already in that form.
    if rows.dtype != torch.uint8:
        exp = rows.to(torch.float32).log2().round().to(torch.int32) + 127
        rows = exp.clamp(0, 255).to(torch.uint8)
    blocked = _to_blocked(rows.contiguous())
    result = (weight.contiguous(), blocked)
    _WEIGHT_CACHE[key] = result
    return result


# --------------------------------------------------------------------------- #
# Activation quantization: per-32-group e4m3 + UE8M0 blocked scales
# --------------------------------------------------------------------------- #
def _load_fused_quant():
    """Imported once: doing this per call costs a module lookup in the hot path."""
    global _FUSED_QUANT
    if _FUSED_QUANT is None:
        from mxfp8_act_quant import quantize_activation_mxfp8
        _FUSED_QUANT = quantize_activation_mxfp8
    return _FUSED_QUANT


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
class _Thresholds:
    def __init__(self, table_path: Path):
        self.by_shape: dict[tuple[int, int], int | None] = {}
        self.always = set()
        self.never = set()
        try:
            data = json.loads(table_path.read_text())
            for key, entry in data.get("shapes", {}).items():
                nk = (int(entry["N"]), int(entry["K"]))
                if entry.get("never_wins"):
                    self.never.add(nk)
                elif entry.get("always_wins"):
                    self.always.add(nk)
                else:
                    self.by_shape[nk] = entry.get("threshold_m")
        except Exception as exc:  # noqa: BLE001
            _LOG.info("No usable MXFP8 crossover table (%s); engine stays off", exc)


def _should_use(M: int, N: int, K: int, table: _Thresholds) -> bool:
    # Phase gate: the MXFP8 path pays per-call host overhead (two fresh
    # allocations + dispatch) that the tuned Triton kernel avoids. Isolated GEMM
    # timings favour MXFP8 at every M, but in-serving at decode-scale M the
    # overhead dominates: measured C1 throughput fell 196 -> 123 tok/s with the
    # engine on, while prefill TTFT improved 2.55 -> 1.90 s. So engage only for
    # large-M (prefill) calls and leave decode on the tuned Triton kernel.
    if M < MIN_M_MXFP8:
        return False
    nk = (N, K)
    if nk in table.never:
        return False
    if nk in table.always:
        return True
    threshold = table.by_shape.get(nk)
    return threshold is not None and M >= threshold


def matmul(A, B, As, Bs, block_size, output_dtype=torch.bfloat16, *, thresholds,
           lib):
    """Block-scaled MXFP8 GEMM; caller has already checked the dispatch table."""
    if output_dtype != torch.bfloat16:
        raise TypeError("MXFP8 engine currently supports bf16 output")
    if lib is None:
        raise RuntimeError("MXFP8 library unavailable")
    m, k = A.shape
    n = B.shape[0]
    out = torch.empty(m, n, device=A.device, dtype=torch.bfloat16)
    payload_w, blocked_w = _prepare_weight(B, Bs)
    payload_a, blocked_a = _load_fused_quant()(A.reshape(m, k),
                                               out_q=None, out_s=None)
    stream = torch.cuda.current_stream().cuda_stream
    rc = lib.mxfp8_gemm(payload_a.data_ptr(), payload_w.data_ptr(),
                        blocked_a.data_ptr(), blocked_w.data_ptr(),
                        out.data_ptr(), m, n, k, ctypes.c_void_p(stream))
    if rc:
        raise RuntimeError(f"mxfp8_gemm rc={rc}: {lib.last_error().decode()}")
    return out


def install():
    """Wrap the SM120 tuned FP8 kernel with MXFP8 dispatch. Best-effort only."""
    global _INSTALLED, _DISABLED, _LIB
    if _INSTALLED:
        return True
    if os.environ.get("DSV41_SM120_MXFP8", "0") not in ("1", "true", "yes"):
        return False
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        return False

    table_path = Path(__file__).with_name("mxfp8_crossover.json")
    thresholds = _Thresholds(table_path)
    if not (thresholds.by_shape or thresholds.always):
        return False

    lib = _build_library()
    if lib is None:
        return False
    _LIB = lib

    from sglang.kernels.ops.quantization import fp8_kernel as kernels
    original = kernels.w8a8_block_fp8_matmul_triton

    def _fallback(*args, **kwargs):
        """Call the CURRENT entry point, minus ourselves.

        Snapshotting the kernel at install time is wrong: whichever of
        {tuned SM120 patch, MXFP8 engine} installs second would capture the raw
        Triton kernel and silently bypass the other. Resolve at call time so the
        order of installation cannot change which kernels are in play.
        """
        current = kernels.w8a8_block_fp8_matmul_triton
        if getattr(current, "__sm120_mxfp8__", False):
            # We are the outermost wrapper; step inward to the previous layer.
            current = getattr(current, "__sm120_inner__", original)
        return current(*args, **kwargs)

    def optimized(A, B, As, Bs, block_size, output_dtype=torch.bfloat16):
        global _DISABLED
        if _DISABLED:
            return _fallback(A, B, As, Bs, block_size, output_dtype=output_dtype)
        try:
            if (
                A.dim() == 2
                and A.is_cuda
                and A.dtype == torch.bfloat16
                and B.dtype == torch.float8_e4m3fn
                and block_size == [_KG, _KG]
                and A.shape[1] % _KG == 0
                and B.shape[0] % _KG == 0
                and _should_use(A.shape[0], B.shape[0], A.shape[1], thresholds)
            ):
                return matmul(A, B, As, Bs, block_size,
                              output_dtype=output_dtype,
                              thresholds=thresholds, lib=lib)
        except Exception as exc:  # noqa: BLE001
            _DISABLED = True
            _LOG.warning("MXFP8 engine disabled after failure: %s", exc)
        return _fallback(A, B, As, Bs, block_size, output_dtype=output_dtype)

    optimized.__sm120_mxfp8__ = True
    optimized.__sm120_inner__ = original
    kernels.w8a8_block_fp8_matmul_triton = optimized
    try:
        from sglang.srt.layers.quantization import fp8_utils
        fp8_utils.w8a8_block_fp8_matmul_triton = optimized
    except Exception:  # noqa: BLE001
        pass
    _INSTALLED = True
    _LOG.info("SM120 MXFP8 dispatch installed (%d shapes)", len(thresholds.by_shape))
    return True