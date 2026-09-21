# syntax=docker/dockerfile:1
ARG BASE_IMAGE=lmsysorg/sglang@sha256:4a5d132a06a77c8331e15845f2e925adc788b00105097ad55409afa3f4fa4860
FROM ${BASE_IMAGE} AS hybrid

ARG BASE_IMAGE
ARG SGLANG_COMMIT=da64c5cbb8cf6bfd39be19da43573fdfd484c43a
LABEL org.opencontainers.image.title="DeepSeek-V4.1-Flash SM120 deployment kit" \
      org.opencontainers.image.description="Pinned SGLang source and SM120 adapters; rebuilt artifacts are not historical image replicas" \
      org.opencontainers.image.source="https://github.com/nguyenhoangthuan99/deepseek-v41-flash-sm120" \
      org.opencontainers.image.revision="" \
      org.opencontainers.image.authors="nguyenhoangthuan99" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      org.deepseek-v41-flash-sm120.sglang.revision="${SGLANG_COMMIT}"

COPY versions.json /opt/dsv41/versions.json
# The base supplies Torch and the rest of the validated serving stack. Install
# only these two wheels: never let dependency resolution replace Torch/SGLang.
RUN python3 -c 'import json,os; p=json.load(open("/opt/dsv41/versions.json")); assert p["base_image"] == os.environ["BASE_IMAGE"]; assert p["sglang"]["commit"] == os.environ["SGLANG_COMMIT"]' \
    && python3 -m pip install --no-cache-dir --no-deps tilelang==0.1.14 apache-tvm-ffi==0.1.11

# Replace, rather than overlay, the editable installed Python tree. No pip
# reinstall: its existing distribution metadata and all dependency pins remain.
RUN rm -rf /sgl-workspace/sglang/python
COPY sglang/python/ /sgl-workspace/sglang/python/
COPY runtime/ /opt/dsv41/runtime/
COPY docker/verify_image.py /opt/dsv41/verify_image.py
COPY docker/sglang-source-manifest.json /opt/dsv41/sglang-source-manifest.json
COPY LICENSE NOTICE /opt/dsv41/
COPY licenses/ /opt/dsv41/licenses/
ENV PYTHONPATH=/opt/dsv41/runtime:/sgl-workspace/sglang/python \
    DSV41_BUILD_TARGET=hybrid
WORKDIR /opt/dsv41

FROM hybrid AS runtime
LABEL org.opencontainers.image.description="Pinned SGLang/SM120 runtime with FlashInfer 0.6.18 minimal PR5121 backport" \
      flashinfer.pr5121.commit="24804035ad9d8372e5883007f2d391986f17314d"
COPY docker/pr5121.patch docker/pr5121-0.6.18.patch docker/apply_pr.py /opt/flashinfer-pr5121/
RUN python3 /opt/flashinfer-pr5121/apply_pr.py

# Explicit SM120 family architecture permits CPU-only builds. The private name
# bypasses only the stale sparse-MLA cache; release flags match the tested build.
RUN FLASHINFER_CUDA_ARCH_LIST=12.0f MAX_JOBS=4 FLASHINFER_JIT_VERBOSE=1 FLASHINFER_JIT_DEBUG=0 \
    python3 -c 'import hashlib,json; from pathlib import Path; from flashinfer.jit.mla import gen_sparse_mla_sm120_module; pins=json.loads(Path("/opt/dsv41/versions.json").read_text())["flashinfer"]; s=gen_sparse_mla_sm120_module(); assert s.name == pins["jit_module"] and not s.is_aot; s.build(verbose=True); p=s.get_library_path(); assert p.is_file(); m=Path("/opt/flashinfer-pr5121/manifest.json"); d=json.loads(m.read_text()); d.update(library=str(p),library_sha256=hashlib.sha256(p.read_bytes()).hexdigest()); m.write_text(json.dumps(d,indent=2)+"\n"); print(m.read_text())'
ENV DSV41_BUILD_TARGET=runtime

# ---------------------------------------------------------------------------
# mxfp8: runtime image with the cuBLASLt MXFP8 dense-GEMM engine pre-enabled.
#
# Adds nothing to the existing stages except defaults; the engine and its fused
# activation quantizer already arrive via `COPY runtime/`. JIT-compiles the
# ~4 KB cuBLASLt wrapper at first use so the 20 MB CUDA toolchain stays out of
# the image, and proves at build time that nvcc + a UE8M0-capable cuBLASLt are
# present (both are required, neither is guaranteed by the base).
# ---------------------------------------------------------------------------
FROM runtime AS mxfp8
LABEL org.opencontainers.image.description="SM120 runtime with the cuBLASLt MXFP8 dense-GEMM engine enabled (opt-out via DSV41_SM120_MXFP8=0)"

ARG SGLANG_COMMIT=da64c5cbb8cf6bfd39be19da43573fdfd484c43a
RUN set -eux; \
    command -v nvcc >/dev/null || { echo "nvcc missing: MXFP8 engine cannot JIT"; exit 1; }; \
    grep -q CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 /usr/local/cuda/include/cublasLt.h \
      || { echo "cuBLASLt lacks VEC32_UE8M0: engine unsupported"; exit 1; }; \
    test -f /opt/dsv41/runtime/mxfp8_gemm_sm120.py; \
    test -f /opt/dsv41/runtime/mxfp8_act_quant.py; \
    test -f /opt/dsv41/runtime/cublaslt_mxfp8.cu; \
    test -f /opt/dsv41/runtime/mxfp8_crossover.json; \
    cd /opt/dsv41/runtime && nvcc -O3 -std=c++17 --shared -Xcompiler -fPIC \
      -gencode arch=compute_120a,code=sm_120a cublaslt_mxfp8.cu -lcublasLt \
      -o /tmp/mxfp8_probe.so; \
    rm -f /tmp/mxfp8_probe.so

ENV DSV41_SM120_MXFP8=1 \
    DSV41_SM120_FP8_DISABLE=0 \
    DSV41_SM120_MXFP8_MIN_M=512
