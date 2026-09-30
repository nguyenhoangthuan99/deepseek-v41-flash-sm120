#!/usr/bin/env bash
# Builds the vLLM engine image for the SM120 DeepSeek-V4.1-Flash kit from public
# sources only. No pre-built prerequisite images are required.
#
#   ./docker-build-vllm.sh
#
# Three images are produced:
#   1. dsv41-vllm-base:sm120  the pinned fork commit built with its own
#                             docker/Dockerfile (torch, CUDA, FlashInfer and the
#                             fork-compiled extensions).
#   2. dsv41-vllm-ext:sm120   the compiled package tree from (1) plus a DeepGEMM
#                             _C rebuilt with the SM120 page-32 gates.
#   3. dsv41-vllm:sm120       the overlay engine image that serve-vllm.sh runs.
#
# Override any of the variables below in the environment. The build context for
# step 1 is a temporary git export of the pinned commit with a version tag;
# neither the fork repository nor this one carries tags, and the fork's
# Dockerfile derives its wheel version from git.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
VLLM_SRC=${VLLM_SRC:-$ROOT/vllm}
# The fork's main was rewritten upstream; this commit is the current tip and has
# the same tree as the previously pinned revision (803434a5702fa871123b1ec370b6e96c47d00c2e).
VLLM_COMMIT=${VLLM_COMMIT:-005b0af0d996d354dde582152909be96f7feac07}
VLLM_VERSION_TAG=${VLLM_VERSION_TAG:-v0.28.1rc1}
CUDA_VERSION=${CUDA_VERSION:-13.0.3}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}
FLASHINFER_VERSION=${FLASHINFER_VERSION:-0.6.18.post1}
# SM120-only kernels: this kit targets RTX PRO 6000 / SM120 hosts. vLLM gates
# every per-arch component on this list, so with 12.0 alone it reports
# "CUDA supported target architectures: 12.0" and skips Machete, AllSpark,
# scaled_mm_c3x_sm90, SM10x/11x NVFP4/MXFP4 and CUTLASS MLA.
#
# The FlashAttention subproject is the exception, and its logs still mention
# `_sm80` source files. Upstream vllm-flash-attn intersects its own lists with
# this one: FA2 resolves to "8.0+PTX" (PTX only, JIT-compiled on SM120, because
# FA2 has no native SM120 kernels) and FA3 resolves to an empty list, so the
# Hopper (9.0a) kernels are not built at all. `vllm.vllm_flash_attn` tolerates
# the absent `_vllm_fa3_C` and reports FA3_AVAILABLE=False.
# The fork's Dockerfile defaults to MAX_JOBS=2, which serializes the CUDA
# compile for the ~410 targets in the extensions build. Default to the host core
# count instead and let the caller dial it down on memory-constrained machines.
MAX_JOBS=${MAX_JOBS:-$(nproc)}
NVCC_THREADS=${NVCC_THREADS:-2}
TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}
BASE_IMAGE=${BASE_IMAGE:-dsv41-vllm-base:sm120}
EXT_IMAGE=${EXT_IMAGE:-dsv41-vllm-ext:sm120}
IMAGE=${IMAGE:-dsv41-vllm:sm120}
CONTEXT_DIR=${CONTEXT_DIR:-/tmp/dsv41-vllm-context}
STEPS=${STEPS:-all}

fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "Required command not found: $1"; }

need docker
need git
need tar
[[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]] || fail "MAX_JOBS must be a positive integer"
[[ "$NVCC_THREADS" =~ ^[1-9][0-9]*$ ]] || fail "NVCC_THREADS must be a positive integer"
[[ -d "$VLLM_SRC/docker" ]] || fail "vLLM submodule not found at $VLLM_SRC; run: git submodule update --init --recursive"

case "$STEPS" in
  all|base|ext|overlay) ;;
  *) fail "STEPS must be one of: all, base, ext, overlay" ;;
esac

actual=$(git -C "$VLLM_SRC" rev-parse HEAD)
[[ "$actual" == "$VLLM_COMMIT" ]] || fail "vLLM submodule is at $actual, expected $VLLM_COMMIT"
dirty=$(git -C "$VLLM_SRC" status --porcelain)
[[ -z "$dirty" ]] || fail "vLLM submodule has local changes; commit or discard them first:
$dirty"

if [[ "$STEPS" == all || "$STEPS" == base ]]; then
  printf '==> exporting %s to %s\n' "${VLLM_COMMIT:0:12}" "$CONTEXT_DIR"
  rm -rf -- "$CONTEXT_DIR"
  mkdir -p -- "$CONTEXT_DIR"
  git -C "$VLLM_SRC" archive --format=tar "$VLLM_COMMIT" | tar -x -C "$CONTEXT_DIR"
  # Deterministic synthetic history so the fork's setuptools_scm call resolves to
  # the versioned tag instead of failing on a tag-less checkout. The recorded
  # source revision remains VLLM_COMMIT (also passed below as an image label).
  (
    cd -- "$CONTEXT_DIR"
    git init -q -b main
    git add -A
    GIT_AUTHOR_NAME="dsv41 build" GIT_AUTHOR_EMAIL="build@invalid" \
    GIT_COMMITTER_NAME="dsv41 build" GIT_COMMITTER_EMAIL="build@invalid" \
    GIT_AUTHOR_DATE="2026-01-01T00:00:00+00:00" GIT_COMMITTER_DATE="2026-01-01T00:00:00+00:00" \
      git -c commit.gpgsign=false commit -qm "vLLM fork source at $VLLM_COMMIT"
    git tag "$VLLM_VERSION_TAG"
  )

  printf '==> building %s from the pinned fork source (MAX_JOBS=%s, NVCC_THREADS=%s)\n' \
    "$BASE_IMAGE" "$MAX_JOBS" "$NVCC_THREADS"
  docker build -f "$CONTEXT_DIR/docker/Dockerfile" "$CONTEXT_DIR" \
    --target vllm-openai \
    --build-arg "CUDA_VERSION=$CUDA_VERSION" \
    --build-arg "PYTHON_VERSION=$PYTHON_VERSION" \
    --build-arg "FLASHINFER_VERSION=$FLASHINFER_VERSION" \
    --build-arg "torch_cuda_arch_list=$TORCH_CUDA_ARCH_LIST" \
    --build-arg "max_jobs=$MAX_JOBS" \
    --build-arg "nvcc_threads=$NVCC_THREADS" \
    --build-arg "VLLM_BUILD_COMMIT=$VLLM_COMMIT" \
    -t "$BASE_IMAGE"
fi

if [[ "$STEPS" == all || "$STEPS" == ext ]]; then
  printf '==> building %s\n' "$EXT_IMAGE"
  docker build -f "$ROOT/docker/Dockerfile.vllm-ext" "$ROOT" \
    --build-arg "BASE_IMAGE=$BASE_IMAGE" \
    --build-arg "CUDA_VERSION=$CUDA_VERSION" \
    -t "$EXT_IMAGE"
fi

if [[ "$STEPS" == all || "$STEPS" == overlay ]]; then
  printf '==> building %s\n' "$IMAGE"
  docker build -f "$ROOT/Dockerfile.vllm" "$ROOT" \
    --build-arg "BASE_IMAGE=$BASE_IMAGE" \
    --build-arg "EXT_IMAGE=$EXT_IMAGE" \
    --build-arg "VLLM_COMMIT=$VLLM_COMMIT" \
    -t "$IMAGE"
fi

printf '\nBuilt images:\n'
for img in "$BASE_IMAGE" "$EXT_IMAGE" "$IMAGE"; do
  docker image inspect "$img" --format '  {{.RepoTags}} {{.Id}}' 2>/dev/null || true
done
printf '\nLaunch with: MODEL_DIR=<checkpoint> IMAGE=%s ./serve-vllm.sh\n' "$IMAGE"
