#!/usr/bin/env bash
# PD (prefill/decode) disaggregation launcher + the runnable low-latency fallback
# for staggered-arrival head-of-line blocking (see docs/SCHEDULING-BLOCKING.md and
# ../deepseek-ai/serve-dsv41/SCHEDULING-BLOCKING.md).
#
# HONEST SUMMARY (verify with `./serve-disagg.sh plan`):
#   DeepSeek-V4.1-Flash has NO separable encoder/decoder weights. The vision
#   encoder (ViT) is ~1 GB; the "~8B prefill / ~16B decode" asymmetry is a
#   compute-path difference over the SHARED 510 GB model. In SGLang PD
#   disaggregation BOTH pools load the FULL model, so a 4+4 split needs
#   510/4 = 127.6 GB/GPU > 95.6 GB VRAM -> it DOES NOT FIT. It only fits if the
#   203 GB Engram tables move to host RAM (76.8 GB/GPU, ~12 GB KV left), and even
#   then the KV transport (mooncake/nixl) needs InfiniBand/RDMA, which this
#   single PCIe-PHB node lacks. Viable disagg here = 8+8 across TWO nodes + IB.
#
# So on THIS 8-GPU box the fix for the blocking is the aggregated low-latency
# mode below, not disaggregation.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
DEPLOY="$ROOT/deploy.sh"
PLAN="$ROOT/scripts/disagg_plan.py"

# Inherit ALL best (non-disaggregated) configuration. deploy.sh already bakes the
# validated defaults (flashinfer_mxfp4 MoE, fp4 indexer, tuned SM120 block-FP8,
# DSPARK, 1M context, mem-fraction 0.80, PHB, parsers). This also pulls the local
# override (MODEL_DIR, IMAGE, seed, BIND_HOST, ...) so disagg roles and the
# aggregated mode launch with the exact validated stack, overriding only the
# disaggregation-specific and anti-blocking knobs on top.
# Inherit the best config as DEFAULTS ONLY: parse `export VAR=value` lines and set
# each var unless the caller already set it. Precedence: caller env > ENV_FILE >
# deploy.sh defaults. (A plain `source` would clobber e.g. SPEC=0 from the CLI.)
ENV_FILE=${ENV_FILE:-$ROOT/deploy.sh.tmp}
if [[ -f "$ENV_FILE" ]]; then
    while IFS= read -r _line; do
        [[ "$_line" =~ ^[[:space:]]*export[[:space:]]+([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
        _var=${BASH_REMATCH[1]}; _val=${BASH_REMATCH[2]}
        _val=${_val%\"}; _val=${_val#\"}; _val=${_val%\'}; _val=${_val#\'}
        [[ -n "${!_var+x}" ]] || export "$_var=$_val"
    done < "$ENV_FILE"
fi

# Topology / transport (only used by the real disagg roles).
PREFILL_GPUS=${PREFILL_GPUS:-4}
DECODE_GPUS=${DECODE_GPUS:-4}
ENGRAM_CPU=${ENGRAM_CPU:-0}
DISAGG_BACKEND=${DISAGG_BACKEND:-mooncake}
DISAGG_IB_DEVICE=${DISAGG_IB_DEVICE:-}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998}
HOST=${HOST:-127.0.0.1}
PREFILL_PORT=${PREFILL_PORT:-30001}
DECODE_PORT=${DECODE_PORT:-30002}
LB_PORT=${LB_PORT:-30000}
EMIT=${EMIT:-0}

fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "Required command not found: $1"; }

usage() {
    cat <<'USAGE'
serve-disagg.sh COMMAND

  plan [ARGS...]   Weight/VRAM-fit report (wraps scripts/disagg_plan.py). Proves
                   whether a prefill/decode GPU split fits. Pass planner flags,
                   e.g. --prefill 4 --decode 4 [--engram-cpu].

  aggregated [start|restart]
                   RUNNABLE FIX for this 8-GPU box. Relaunches the validated
                   config keeping DSPARK ON (inherited) plus spec-compatible
                   anti-blocking knobs: num-continuous-decode-steps=1,
                   chunked-prefill-size=1024, schedule-conservativeness=0.8.
                   mixed-chunk is enabled only if you pass SPEC=0 (it does not
                   compose with spec on a single pool). Needs MODEL_DIR; GPU host.

  disagg [--emit]  Attempt PD disaggregation for PREFILL_GPUS+DECODE_GPUS (default
                   4+4). Runs the weight-fit preflight and the IB/RDMA check; on
                   this PCIe-PHB single node it refuses with the exact math.
                   --emit prints the multi-node prefill/decode/lb commands.

  prefill | decode | lb
                   Launch one PD role (real disagg hardware only: >=8 GPUs/pool
                   and InfiniBand). Gated by the same preflight. EMIT=1 prints
                   instead of executing.

Environment: PREFILL_GPUS, DECODE_GPUS, ENGRAM_CPU, DISAGG_BACKEND (mooncake|nixl),
DISAGG_IB_DEVICE, BOOTSTRAP_PORT, HOST, PREFILL_PORT, DECODE_PORT, LB_PORT, EMIT.
USAGE
}

engram_flag() { [[ "$ENGRAM_CPU" == 1 ]] && echo "--engram-cpu" || true; }

preflight_fit() {
    need python3
    # shellcheck disable=SC2046
    python3 "$PLAN" --prefill "$PREFILL_GPUS" --decode "$DECODE_GPUS" $(engram_flag) \
        || fail "Weight-fit preflight failed: the ${PREFILL_GPUS}+${DECODE_GPUS} split does not fit. See table above; on this box use './serve-disagg.sh aggregated'."
}

preflight_transport() {
    [[ -n "$DISAGG_IB_DEVICE" ]] || fail \
"PD KV transport ($DISAGG_BACKEND) requires InfiniBand/RDMA. This host is PCIe-PHB \
with no IB device. Set DISAGG_IB_DEVICE on RDMA-capable, multi-node hardware. \
On this single node use './serve-disagg.sh aggregated'."
}

# EXTRA_ARGS fragments for each disaggregation role.
role_extra() {
    local role="$1"
    case "$role" in
        prefill) printf -- '--disaggregation-mode prefill --disaggregation-transfer-backend %s --disaggregation-bootstrap-port %s --disaggregation-ib-device %s' "$DISAGG_BACKEND" "$BOOTSTRAP_PORT" "$DISAGG_IB_DEVICE" ;;
        decode)  printf -- '--disaggregation-mode decode --disaggregation-transfer-backend %s --disaggregation-ib-device %s' "$DISAGG_BACKEND" "$DISAGG_IB_DEVICE" ;;
    esac
}

launch_role() {
    local role="$1" gpus port name devlist
    preflight_fit
    preflight_transport
    if [[ "$role" == prefill ]]; then gpus=$PREFILL_GPUS; port=$PREFILL_PORT; name=dsv41-prefill;
        devlist="device=$(seq -s, 0 $((PREFILL_GPUS-1)))"
    else gpus=$DECODE_GPUS; port=$DECODE_PORT; name=dsv41-decode;
        devlist="device=$(seq -s, "$PREFILL_GPUS" $((PREFILL_GPUS+DECODE_GPUS-1)))"
    fi
    # Inherit best config from the sourced env; override only disagg-specific keys.
    # Speculation is a decode-time feature: keep the inherited best SPEC (DSPARK)
    # on the decode pool, disable it on the prefill-only pool.
    local -a rolenv=(
        NAME="$name" PORT="$port" TP="$gpus" EP="$gpus"
        GPU_DEVICES="$devlist" BIND_HOST="${BIND_HOST:-0.0.0.0}"
        EXTRA_ARGS="$(role_extra "$role")"
    )
    [[ "$role" == prefill ]] && rolenv+=(SPEC=0)
    if [[ "$EMIT" == 1 ]]; then
        printf '%s ' "${rolenv[@]}"; printf 'DRY_RUN=1 %s start\n' "$DEPLOY"
        env "${rolenv[@]}" DRY_RUN=1 "$DEPLOY" start
    else
        exec env "${rolenv[@]}" "$DEPLOY" start
    fi
}

launch_lb() {
    # sglang_router is the current PD load balancer (pip install sglang-router).
    local cmd="python3 -m sglang_router.launch_router --pd-disaggregation \
--prefill http://$HOST:$PREFILL_PORT --decode http://$HOST:$DECODE_PORT \
--host 0.0.0.0 --port $LB_PORT"
    if [[ "$EMIT" == 1 ]]; then echo "$cmd"; else preflight_transport; exec $cmd; fi
}

cmd=${1:-}; shift || true
case "$cmd" in
    plan)   need python3; exec python3 "$PLAN" "$@" ;;
    aggregated)
        action=${1:-restart}
        [[ "$action" =~ ^(start|restart)$ ]] || fail "aggregated takes start|restart"
        # Aggregated single-pool fallback (used when disagg does not fit). Keep the
        # inherited best speculation (DSPARK) ON; apply only spec-compatible
        # anti-blocking knobs. mixed-chunk overlap does NOT compose with spec on one
        # pool, so enable it only when the caller explicitly drops spec (SPEC=0).
        # In real disaggregation this tradeoff is gone: prefill/decode are separate
        # pools, so DSPARK stays on the decode pool with nothing to block it.
        mixed=0; [[ "${SPEC:-1}" == 0 ]] && mixed=1
        echo "Launching aggregated low-latency config (DSPARK=${SPEC:-1}, mixed-chunk=$mixed, decode-steps=1, chunk=1024, conserv=0.8)..."
        exec env MIXED_CHUNK="$mixed" DECODE_STEPS=1 CHUNKED_PREFILL=1024 \
            SCHED_CONSERV=0.8 "$DEPLOY" "$action"
        ;;
    disagg)
        [[ "${1:-}" == --emit ]] && EMIT=1
        echo "== Weight-fit preflight for ${PREFILL_GPUS}+${DECODE_GPUS} =="
        if ! python3 "$PLAN" --prefill "$PREFILL_GPUS" --decode "$DECODE_GPUS" $(engram_flag); then
            cat >&2 <<EOF

Refusing: the ${PREFILL_GPUS}+${DECODE_GPUS} split cannot hold the model on this
hardware. Options:
  1. Run the blocking fix that works here:  ./serve-disagg.sh aggregated
  2. Try Engram-on-CPU (adds per-token latency at layers 1,14, unverified in
     SGLang): ENGRAM_CPU=1 ./serve-disagg.sh plan --prefill 4 --decode 4
  3. Real disagg: 8+8 across TWO nodes with InfiniBand (PREFILL_GPUS=8
     DECODE_GPUS=8 DISAGG_IB_DEVICE=mlx5_0 on RDMA hardware).
EOF
            exit 3
        fi
        preflight_transport
        echo "Fit + transport OK. Emitting role commands:"
        EMIT=1 launch_role prefill; EMIT=1 launch_role decode; EMIT=1 launch_lb
        ;;
    prefill) launch_role prefill ;;
    decode)  launch_role decode ;;
    lb)      launch_lb ;;
    ""|-h|--help|help) usage ;;
    *) usage; fail "Unknown command: $cmd" ;;
esac
