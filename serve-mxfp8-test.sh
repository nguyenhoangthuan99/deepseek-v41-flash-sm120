#!/usr/bin/env bash
# Standalone MXFP8 end-to-end bring-up for a DEDICATED test host (VM .106).
#
# This script is intentionally INDEPENDENT of deploy.sh: it neither reads nor
# modifies the production deployment path. It launches its own container, its own
# runtime overlay, its own port, and its own logs. Nothing here runs against the
# serving host.
#
# What it proves end-to-end: the cuBLASLt MXFP8 dense-GEMM engine
# (fused UE8M0 activation quant + CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0)
# wired into the real SGLang block-FP8 linear path, measured against the same
# stack with the engine off.
#
# Usage (on the GPU host, all GPUs free):
#   ./serve-mxfp8-test.sh check          # toolchain/preflight only
#   ./serve-mxfp8-test.sh start [on|off] # launch; default "on"
#   ./serve-mxfp8-test.sh bench          # A/B throughput, writes results
#   ./serve-mxfp8-test.sh stop
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# The kit image carries the FlashInfer PR5121 sparse-MLA prefill backport the
# model needs; the bare base image crashes at first prefill without it.
IMAGE=${IMAGE:-deepseek-v41-flash-sm120:mxfp8}
NAME=${NAME:-dsv41-mxfp8test}
PORT=${PORT:-30100}
TP=${TP:-8}
MODEL_DIR=${MODEL_DIR:-/mnt/nas/alex/models/deepseek-ai/DeepSeek-V4.1-Flash}
CONTEXT_LEN=${CONTEXT_LEN:-1048576}
MEM_FRACTION=${MEM_FRACTION:-0.80}
MAX_RUNNING=${MAX_RUNNING:-32}
LOGDIR=${LOGDIR:-$ROOT/logs}
RUNTIME=$ROOT/runtime
CACHE=${CACHE:-/workspace/mxfp8-test-cache}

fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "Missing command: $1"; }

check() {
    need docker
    [[ -d "$MODEL_DIR" ]] || fail "MODEL_DIR not found: $MODEL_DIR (weights stay outside the repo)"
    docker image inspect "$IMAGE" >/dev/null 2>&1 || fail "Image not found: $IMAGE"
    nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi unavailable"
    local free=0
    while read -r used; do (( used < 2000 )) && free=$((free+1)); done < \
        <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    (( free >= TP )) || fail "Need $TP idle GPUs; found $free"
    for f in mxfp8_gemm_sm120.py mxfp8_act_quant.py cublaslt_mxfp8.cu \
             mxfp8_crossover.json sitecustomize.py; do
        [[ -f "$RUNTIME/$f" ]] || fail "runtime/$f missing"
    done
    nvcc --version | tail -1
    echo "OK: $free idle GPUs, image present, runtime overlay complete"
}

stop() { docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" || echo "$NAME not running"; }

start() {
    local mode=${1:-on}
    case "$mode" in on) MX=1 ;; off) MX=0 ;; *) fail "mode must be on|off" ;; esac
    check
    mkdir -p "$LOGDIR" "$CACHE"
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    # Publish the runtime overlay + shipped CUDA source into a host cache dir so the
    # container can JIT-compile the MXFP8 wrapper into it (persisted across runs).
    cp "$RUNTIME"/*.py "$RUNTIME"/*.json "$RUNTIME"/*.cu "$CACHE"/ 2>/dev/null || true
    docker run -d --name "$NAME" --gpus all --ipc=host --shm-size 32g \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -p "0.0.0.0:${PORT}:${PORT}" \
        -v "$MODEL_DIR:/model:ro" \
        -v "$CACHE:/opt/dsv41/runtime" \
        -e PYTHONPATH=/opt/dsv41/runtime:/sgl-workspace/sglang/python \
        -e DSV41_SM120_MXFP8="$MX" \
        -e DSV41_SM120_MXFP8_MIN_M="${MXFP8_MIN_M:-512}" \
        -e DSV41_SM120_FP8_DISABLE=0 \
        -e DSV41_SM120_DISABLE=1 \
        -e NCCL_P2P_LEVEL=PHB \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        -e SGLANG_SM120_FLASHMLA_BACKEND=flashinfer \
        -e SGLANG_RAGGED_VERIFY_MODE=static \
        -e SGLANG_SIMULATE_ACC_LEN=-1 \
        -e SGLANG_DSPARK_ENABLE_SPS_RECORD=0 \
        -e SGLANG_DEFAULT_THINKING=false \
        -e PYTHONUNBUFFERED=1 \
        -e DSV41_SM120_STRICT=1 \
        "$IMAGE" python3 -m sglang.launch_server \
        --model-path /model --host 0.0.0.0 --port "$PORT" \
        --served-model-name deepseek-v41-flash \
        --tool-call-parser deepseekv41 --reasoning-parser deepseek-v41 \
        --trust-remote-code --random-seed 599261575 \
        --tensor-parallel-size "$TP" --ep-size "$TP" \
        --context-length "$CONTEXT_LEN" --mem-fraction-static "$MEM_FRACTION" \
        --max-running-requests "$MAX_RUNNING" --chunked-prefill-size 2048 \
        --moe-runner-backend flashinfer_mxfp4 \
        --enable-deepseek-v4-fp4-indexer --num-continuous-decode-steps 4 \
        --speculative-algorithm DSPARK --speculative-dspark-block-size 3 \
        --watchdog-timeout 3600 >"$LOGDIR/launch-$mode.log" 2>&1
    echo "launched $NAME (MXFP8=$mode) on port $PORT; waiting for health..."
    for _ in $(seq 1 90); do
        if curl -sf -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            echo "healthy after ~$(( _ * 10 ))s"; return 0
        fi
        sleep 10
    done
    fail "did not become healthy; see $LOGDIR/launch-$mode.log"
}

bench() {
    need python3
    local out=$LOGDIR/mxfp8-ab-$(date +%Y%m%d-%H%M%S)
    mkdir -p "$out"
    python3 "$ROOT/scripts/bench/bench_serving.py" --base-url "http://127.0.0.1:$PORT" \
        --out "$out" || fail "benchmark failed; see $out"
    echo "results in $out"
}

case "${1:-}" in
    check) check ;;
    start) start "${2:-on}" ;;
    stop)  stop ;;
    bench) bench ;;
    *) sed -n '2,26p' "$0"; exit 1 ;;
esac