#!/usr/bin/env bash
# vLLM engine launcher with the best measured SM120 configuration.
#
# Validated A/B (matched cold workload, 168 requests, medians, 1M context):
#   decode tok/s  C1 179.9, C4 433.6, C16 914.5, C32 1226.9
#   (untuned kernels: 166.9 / 386.4 / 811.3 / 1149.7)
# Tuning = DSpark K3 speculative decoding, FULL_DECODE_ONLY CUDA graphs,
# FlashInfer b12x/split-K MXFP8 linear kernel, DeepGEMM MoE small-M
# alignment 64 (bit-identical outputs), NCCL PHB.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
IMAGE=${IMAGE:-dsv41-vllm:sm120}
NAME=${NAME:-dsv41-vllm}
BIND_HOST=${BIND_HOST:-127.0.0.1}
PORT=${PORT:-30100}
TP=${TP:-8}
CONTEXT_LEN=${CONTEXT_LEN:-1048576}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
MAX_RUNNING=${MAX_RUNNING:-32}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-8192}
BLOCK_SIZE=${BLOCK_SIZE:-64}
# DSpark speculative decoding depth. K3 measured best overall; SPEC=0 disables.
SPEC=${SPEC:-1}
DSPARK_TOKENS=${DSPARK_TOKENS:-3}
# Measured-win kernel overrides; KERNEL_CONFIG= (empty) reverts to defaults.
KERNEL_CONFIG=${KERNEL_CONFIG-'{"linear_backend":"flashinfer_b12x","deep_gemm_moe_small_m_alignment":64}'}
CUDAGRAPH_CONFIG=${CUDAGRAPH_CONFIG:-'{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[3,4,6,8,12,16,24,32,48,64,96,128]}'}
NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-PHB}
GPU_DEVICES=${GPU_DEVICES:-all}
CACHE_DIR=${CACHE_DIR:-/mnt/nas/.cache/dsv41-sm120}
LOGDIR=${LOGDIR:-$ROOT/logs}
EXTRA_ARGS=${EXTRA_ARGS:-}

fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }
[[ -n "${MODEL_DIR:-}" ]] || fail "MODEL_DIR is required (DeepSeek-V4.1-Flash checkpoint directory)"
[[ -f "$MODEL_DIR/config.json" ]] || fail "MODEL_DIR has no config.json: $MODEL_DIR"
command -v docker >/dev/null || fail "docker not found"
docker image inspect "$IMAGE" >/dev/null 2>&1 || fail "image $IMAGE missing; build with: docker build -f Dockerfile.vllm -t $IMAGE ."
! docker container inspect "$NAME" >/dev/null 2>&1 || fail "container $NAME already exists; stop/remove it first"

mkdir -p "$LOGDIR" "$CACHE_DIR/flashinfer" "$CACHE_DIR/home" ||
  fail "Cannot create log/cache directories; pre-create writable directories or set LOGDIR and CACHE_DIR"
RUN_LOG="$LOGDIR/vllm-$(date -u +%Y%m%dT%H%M%SZ).log"

# Space-separated aliases all served from the same endpoint.
SERVED_NAMES=${SERVED_NAMES:-deepseek-v41-flash}
read -r -a served <<<"$SERVED_NAMES"
args=(--model /model --served-model-name "${served[@]}" --tensor-parallel-size "$TP"
  --max-model-len "$CONTEXT_LEN" --block-size "$BLOCK_SIZE"
  --gpu-memory-utilization "$GPU_MEM_UTIL" --trust-remote-code
  --enable-auto-tool-choice --tool-call-parser deepseek_v41
  --reasoning-parser deepseek_v41
  --host 0.0.0.0 --port "$PORT"
  --max-num-seqs "$MAX_RUNNING" --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
  --compilation-config "$CUDAGRAPH_CONFIG")
[[ -z "$KERNEL_CONFIG" ]] || args+=(--kernel-config "$KERNEL_CONFIG")
[[ "$SPEC" != 1 ]] || args+=(--speculative-config
  "{\"method\":\"dspark\",\"num_speculative_tokens\":$DSPARK_TOKENS,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"standard\"}")
[[ -z "$EXTRA_ARGS" ]] || { read -r -a extra <<<"$EXTRA_ARGS"; args+=("${extra[@]}"); }

docker run --init --rm --name "$NAME" --gpus "$GPU_DEVICES" --ipc=host --shm-size=64g \
  -p "$BIND_HOST:$PORT:$PORT" \
  -v "$MODEL_DIR:/model:ro" \
  -v "$CACHE_DIR/flashinfer:/cache/flashinfer" \
  -v "$CACHE_DIR/home:/root/.cache" \
  -e VLLM_USE_V2_MODEL_RUNNER=1 -e NCCL_P2P_LEVEL="$NCCL_P2P_LEVEL" \
  --entrypoint /opt/sm120-venv/bin/python "$IMAGE" \
  -m vllm.entrypoints.openai.api_server "${args[@]}" 2>&1 | tee "$RUN_LOG"
