#!/usr/bin/env bash
# All operational commands live here. Environment files are never auto-sourced.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
IMAGE=${IMAGE:-deepseek-v41-flash-sm120:latest}
BUILD_TARGET=${BUILD_TARGET:-runtime}
NAME=${NAME:-dsv41}
BIND_HOST=${BIND_HOST:-127.0.0.1}
PORT=${PORT:-30000}
TP=${TP:-8}
EP=${EP:-8}
MOE_BACKEND=${MOE_BACKEND:-flashinfer_mxfp4}
SPEC=${SPEC:-1}
DSPARK_BLOCK=${DSPARK_BLOCK:-3}
CONTEXT_LEN=${CONTEXT_LEN:-1048576}
MEM_FRACTION=${MEM_FRACTION:-0.80}
MAX_RUNNING=${MAX_RUNNING:-32}
DECODE_STEPS=${DECODE_STEPS:-4}
# Scheduler-latency knobs (see SCHEDULING-BLOCKING.md). Defaults reproduce the
# validated throughput config; serve-disagg.sh's `aggregated` mode overrides them
# to cut staggered-arrival head-of-line blocking.
CHUNKED_PREFILL=${CHUNKED_PREFILL:-2048}
MIXED_CHUNK=${MIXED_CHUNK:-0}
SCHED_CONSERV=${SCHED_CONSERV:-}
EXTRA_ARGS=${EXTRA_ARGS:-}
# GPU selection: "all" or a Docker device spec, e.g. GPU_DEVICES=device=0,1,2,3
# (used by serve-disagg.sh to place prefill/decode pools on GPU subsets).
GPU_DEVICES=${GPU_DEVICES:-all}
RANDOM_SEED=${RANDOM_SEED:-599261575}
NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-PHB}
# Optional NCCL protocol override. Empty (default) keeps NCCL's tuner.
# Measured in-server on this PHB PCIe topology (3-round medians, 768-tok greedy):
#   NCCL_PROTO=Simple: C32 aggregate +14%, but C1 -24% and C4 -17%; C16 and
#   32k-prefill TTFT unchanged. Opt in only for saturated-batch serving.
NCCL_PROTO=${NCCL_PROTO:-}
DSV41_SM120_DISABLE=${DSV41_SM120_DISABLE:-1}
DSV41_SM120_FP8_DISABLE=${DSV41_SM120_FP8_DISABLE:-0}
# cuBLASLt MXFP8 dense-GEMM engine for the block-FP8 linears (opt-in). Requires
# DSV41_SM120_FP8_DISABLE=0. Dispatches per shape/M from the measured
# runtime/mxfp8_crossover.json; falls back to the tuned Triton kernel elsewhere.
DSV41_SM120_MXFP8=${DSV41_SM120_MXFP8:-0}
STRICT=${STRICT:-1}
DRY_RUN=${DRY_RUN:-0}
BASE_URL=${BASE_URL:-http://127.0.0.1:${PORT}}
MANAGED_LABEL=org.deepseek-v41-flash-sm120.managed
LOGDIR=$ROOT/logs

fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "Required command not found: $1"; }
usage() {
    cat <<'USAGE'
Usage: ./deploy.sh {init|build|start|stop|restart|logs|status|check-image|smoke}

  init         Initialize the pinned SGLang submodule; never track a newer branch.
  build        Build IMAGE from this repository (BUILD_TARGET=runtime or hybrid).
  start        Verify image/checkpoint/idle GPUs, then serve in the foreground.
  stop         Stop only the managed container named NAME, never remove/prune.
  restart      Stop the managed NAME, then run the same checked foreground start.
  logs         Follow NAME's Docker logs, or logs/latest.log after container exit.
  status       Show NAME's state and check BASE_URL/health; failure exits nonzero.
  check-image  Run CPU-only package/source/parser/backport verification in IMAGE.
  smoke        Run the API smoke client, writing logs/smoke-api.json.

MODEL_DIR is required for start/restart and is mounted read-only at /model.
DRY_RUN=1 ./deploy.sh start prints the actual shell-quoted Docker command without
Docker/GPU access, checkpoint loading, lock acquisition, or any container changes.
No .env is automatically sourced. Explicit opt-in: set -a; source .env; set +a
Use command-scoped overrides after sourcing, e.g. SPEC=0 ./deploy.sh start.
USAGE
}

pin() {
    python3 - "$ROOT/versions.json" "$1" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1]))
for key in sys.argv[2].split("."):
    value = value[key]
print(value)
PY
}

check_submodule() {
    need git
    need python3
    [[ -f "$ROOT/sglang/.git" && -d "$ROOT/sglang/python/sglang" ]] ||
        fail "SGLang submodule is missing. Run $ROOT/deploy.sh init first."
    local expected actual dirty
    expected=$(pin sglang.commit)
    actual=$(git -C "$ROOT/sglang" rev-parse HEAD)
    [[ "$actual" == "$expected" ]] ||
        fail "SGLang revision is $actual, expected $expected. Run deploy.sh init; no newer branch is accepted."
    dirty=$(git -C "$ROOT/sglang" status --porcelain --untracked-files=normal)
    [[ -z "$dirty" ]] || fail "SGLang submodule has local changes; refusing an unpinned build."
}

init() {
    need git
    need python3
    git -C "$ROOT" submodule update --init --recursive -- sglang
    check_submodule
}

build() {
    check_submodule
    need docker
    case "$BUILD_TARGET" in runtime|hybrid) ;; *) fail "BUILD_TARGET must be runtime or hybrid" ;; esac
    docker build --progress=plain --target "$BUILD_TARGET" \
        --build-arg "BASE_IMAGE=$(pin base_image)" \
        --build-arg "SGLANG_COMMIT=$(pin sglang.commit)" \
        --tag "$IMAGE" --file "$ROOT/Dockerfile.sglang" "$ROOT"
}

docker_ready() {
    need docker
    docker info >/dev/null || fail "Docker daemon is unavailable or access is denied."
}

require_local_runtime() {
    local endpoint
    if [[ -n "${DOCKER_CONTEXT:-}" ]]; then
        endpoint=$(docker context inspect "$DOCKER_CONTEXT" --format '{{(index .Endpoints "docker").Host}}')
    elif [[ -n "${DOCKER_HOST:-}" ]]; then
        endpoint=$DOCKER_HOST
    else
        endpoint=$(docker context inspect --format '{{(index .Endpoints "docker").Host}}')
    fi
    [[ "$endpoint" == unix://* ]] ||
        fail "Run start/restart directly on the GPU host with a local Docker socket; remote endpoints cannot use local GPU/checkpoint checks."
}

container_exists() { docker container inspect "$NAME" >/dev/null 2>&1; }

check_image() {
    docker_ready
    docker image inspect "$IMAGE" >/dev/null 2>&1 ||
        fail "Image $IMAGE is not available locally; run deploy.sh build."
    # No --gpus, network, checkpoint mount, or server process is permitted here.
    docker run --rm --network none --env CUDA_VISIBLE_DEVICES= \
        --env NVIDIA_VISIBLE_DEVICES=void --entrypoint python3 \
        "$IMAGE" /opt/dsv41/verify_image.py
}

check_model() {
    need python3
    MODEL_DIR=$(python3 - "$MODEL_DIR" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1]).expanduser().resolve()
if not root.is_dir():
    sys.exit("MODEL_DIR must be an existing checkpoint directory")
for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
    path = root / name
    if not path.is_file():
        sys.exit(f"Missing checkpoint file: {name}")
    with path.open() as handle:
        if name != "tokenizer.json":
            json.load(handle)
index = json.loads((root / "model.safetensors.index.json").read_text())
shards = set(index.get("weight_map", {}).values())
if not shards:
    sys.exit("Checkpoint index has no weight shards")
for name in sorted(shards):
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        sys.exit(f"Checkpoint shard escapes MODEL_DIR: {name}")
    if not path.is_file() or path.stat().st_size == 0:
        sys.exit(f"Missing or empty checkpoint shard: {name}")
print(root)
PY
    ) || fail "Checkpoint validation failed."
    [[ "$MODEL_DIR" != *:* && "$MODEL_DIR" != *,* ]] ||
        fail "MODEL_DIR cannot contain ':' or ',' (Docker bind-mount syntax)."
}

check_gpus() {
    need nvidia-smi
    local active devices count
    active=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader) ||
        fail "Unable to query active GPU compute processes."
    [[ -z "${active//[[:space:]]/}" ]] ||
        fail "GPU compute workloads are already running; refusing to interfere. Stop them explicitly outside this kit."
    devices=$(nvidia-smi --query-gpu=index --format=csv,noheader) ||
        fail "Unable to query GPUs."
    count=0
    while IFS= read -r device; do
        [[ -z "$device" ]] || count=$((count + 1))
    done <<< "$devices"
    (( count >= TP )) || fail "Found $count GPUs, but TP=$TP requires at least $TP."
    (( EP == 0 || EP <= count )) || fail "EP=$EP exceeds the available $count GPUs."
}

launch_args() {
    RUN=(docker run --gpus "$GPU_DEVICES" --rm --name "$NAME"
        --label "$MANAGED_LABEL=true"
        --shm-size 32g --ipc=host --cap-add SYS_PTRACE
        --ulimit memlock=-1 --ulimit stack=67108864
        --publish "$BIND_HOST:$PORT:$PORT"
        --volume "$MODEL_DIR:/model:ro"
        --env PYTHONPATH=/opt/dsv41/runtime:/sgl-workspace/sglang/python
        --env "DSV41_SM120_STRICT=$STRICT"
        --env "DSV41_SM120_DISABLE=$DSV41_SM120_DISABLE"
        --env "DSV41_SM120_FP8_DISABLE=$DSV41_SM120_FP8_DISABLE"
        --env "DSV41_SM120_MXFP8=$DSV41_SM120_MXFP8"
        --env "NCCL_P2P_LEVEL=$NCCL_P2P_LEVEL")
    if [[ -n "$NCCL_PROTO" ]]; then RUN+=(--env "NCCL_PROTO=$NCCL_PROTO"); fi
    RUN+=(--env SGLANG_SM120_FLASHMLA_BACKEND=flashinfer
        --env SGLANG_RAGGED_VERIFY_MODE=static
        --env SGLANG_SIMULATE_ACC_LEN=-1
        --env SGLANG_DSPARK_ENABLE_SPS_RECORD=0
        --env SGLANG_DEFAULT_THINKING=false
        --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        --env PYTHONUNBUFFERED=1
        "$IMAGE" python3 -m sglang.launch_server
        --model-path /model --host 0.0.0.0 --port "$PORT"
        --served-model-name deepseek-v41-flash
        --tool-call-parser deepseekv41 --reasoning-parser deepseek-v41
        --trust-remote-code --random-seed "$RANDOM_SEED"
        --tensor-parallel-size "$TP" --context-length "$CONTEXT_LEN"
        --mem-fraction-static "$MEM_FRACTION" --max-running-requests "$MAX_RUNNING"
        --chunked-prefill-size "$CHUNKED_PREFILL" --moe-runner-backend "$MOE_BACKEND")
    if [[ "$EP" != 0 ]]; then RUN+=(--ep-size "$EP"); fi
    RUN+=(--enable-deepseek-v4-fp4-indexer --num-continuous-decode-steps "$DECODE_STEPS")
    if [[ "$SPEC" == 1 ]]; then
        RUN+=(--speculative-algorithm DSPARK --speculative-dspark-block-size "$DSPARK_BLOCK")
    fi
    if [[ "$MIXED_CHUNK" == 1 ]]; then RUN+=(--enable-mixed-chunk); fi
    if [[ -n "$SCHED_CONSERV" ]]; then RUN+=(--schedule-conservativeness "$SCHED_CONSERV"); fi
    RUN+=(--watchdog-timeout 3600)
    if [[ -n "$EXTRA_ARGS" ]]; then
        # shellcheck disable=SC2206
        RUN+=($EXTRA_ARGS)
    fi
    # Keep default target/draft CUDA graphs; the model disables prefill graphs.
}

check_launch_settings() {
    [[ -n "${MODEL_DIR:-}" ]] || fail "Set MODEL_DIR to your checkpoint directory."
    [[ "$NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || fail "Invalid container NAME."
    [[ "$PORT" =~ ^[1-9][0-9]*$ ]] && (( PORT <= 65535 )) || fail "Invalid PORT."
    local variable
    for variable in TP DSPARK_BLOCK CONTEXT_LEN MAX_RUNNING DECODE_STEPS CHUNKED_PREFILL; do
        [[ "${!variable}" =~ ^[1-9][0-9]*$ ]] || fail "$variable must be a positive integer."
    done
    [[ "$EP" =~ ^(0|[1-9][0-9]*)$ ]] || fail "EP must be a nonnegative integer."
    for variable in SPEC STRICT DSV41_SM120_DISABLE DSV41_SM120_FP8_DISABLE MIXED_CHUNK DSV41_SM120_MXFP8; do
        [[ "${!variable}" == 0 || "${!variable}" == 1 ]] || fail "$variable must be 0 or 1."
    done
}

start() {
    check_launch_settings
    if [[ "$DRY_RUN" == 1 ]]; then
        need python3
        MODEL_DIR=$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$MODEL_DIR")
        launch_args
        printf '%q ' "${RUN[@]}"
        printf '\n'
        return
    fi
    need flock
    # One host-local lock across every NAME and repository checkout; retained
    # throughout the foreground run. A restart allows the old launcher to exit.
    exec 9>/tmp/dsv41-sm120-deploy.lock
    flock -w 5 9 || fail "Another kit launcher holds the host GPU lock."
    docker_ready
    if container_exists; then
        fail "Container $NAME already exists; it will not be replaced. Use an explicit managed stop first."
    fi
    require_local_runtime
    check_model
    check_gpus
    check_image
    # Recheck immediately before launch; never intentionally share busy GPUs.
    check_gpus
    launch_args
    mkdir -p "$LOGDIR"
    local log rc
    log=$LOGDIR/$(date -u +%Y%m%dT%H%M%SZ)-${NAME}-$$.log
    : > "$log"
    ln -sfn -- "$(basename -- "$log")" "$LOGDIR/latest.log"
    printf 'Serving %s; output: %s\n' "$NAME" "$log"
    set +e
    "${RUN[@]}" 2>&1 | tee -a "$log"
    rc=$?
    set -e
    printf 'Foreground server exited with status %s\n' "$rc" | tee -a "$log"
    return "$rc"
}

stop() {
    docker_ready
    if ! container_exists; then
        printf 'Container %s is not present; nothing stopped.\n' "$NAME"
        return
    fi
    local inspected id managed
    inspected=$(docker container inspect --format "{{.Id}} {{index .Config.Labels \"$MANAGED_LABEL\"}}" "$NAME")
    read -r id managed <<< "$inspected"
    [[ "$managed" == true ]] || fail "Container $NAME is not managed by this kit; refusing to stop it."
    # Stop the inspected identity, not a name that another operator could reuse.
    docker stop --timeout 60 "$id"
}

logs() {
    docker_ready
    if container_exists; then
        docker logs --follow --tail 100 "$NAME"
    else
        [[ -f "$LOGDIR/latest.log" ]] || fail "No container $NAME or saved latest.log exists."
        tail -n 100 -F "$LOGDIR/latest.log"
    fi
}

status() {
    docker_ready
    container_exists || fail "Container $NAME is not present."
    docker container inspect --format '{{.Name}}: {{.State.Status}} (exit={{.State.ExitCode}})' "$NAME"
    [[ $(docker container inspect --format '{{.State.Running}}' "$NAME") == true ]] ||
        fail "Container $NAME is not running."
    need python3
    python3 - "$BASE_URL" <<'PY'
import sys
import urllib.request
url = sys.argv[1].rstrip("/") + "/health"
try:
    with urllib.request.urlopen(url, timeout=10) as response:
        if response.status != 200:
            sys.exit(f"Health check failed: HTTP {response.status}")
        print(f"API healthy: {url}")
except Exception as exc:
    sys.exit(f"API is not ready: {exc}")
PY
}

smoke() {
    need python3
    mkdir -p "$LOGDIR"
    python3 "$ROOT/scripts/smoke_api.py" --base-url "$BASE_URL" --output "$LOGDIR/smoke-api.json"
}

[[ $# == 1 ]] || { usage >&2; exit 2; }
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || fail "DRY_RUN must be 0 or 1."
if [[ "$DRY_RUN" == 1 && "$1" != start ]]; then
    fail "DRY_RUN=1 is supported only for start; no command has been executed."
fi
case "$1" in
    init) init ;;
    build) build ;;
    start) start ;;
    stop) stop ;;
    restart) check_launch_settings; docker_ready; require_local_runtime; check_model; check_image; stop; start ;;
    logs) logs ;;
    status) status ;;
    check-image) check_image ;;
    smoke) smoke ;;
    help|--help|-h) usage ;;
    *) usage >&2; exit 2 ;;
esac
