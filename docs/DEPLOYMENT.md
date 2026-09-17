# Deployment guide

## Scope and prerequisites

Use the [README quick start](../README.md) for clone, explicit environment export, build, and launch. The recipe targets **8× NVIDIA RTX PRO 6000 Blackwell Server Edition, SM120**, not an arbitrary eight-GPU machine. Install Docker/BuildKit, a compatible NVIDIA driver, NVIDIA Container Toolkit, Git, Bash, util-linux `flock`, Python 3, and tmux on the host. The checkpoint must already exist on storage mounted on that host. No host CUDA compiler, host SGLang installation, manual source patching, or model download is part of the deployment process.

Run `start` and `restart` directly on the GPU host with a local Docker Unix socket. They refuse remote Docker endpoints so checkpoint/GPU checks cannot accidentally inspect a different machine. `build` and CPU-only `check-image` may use a remote Docker daemon; that does not make remote GPU preflight safe.

Keep a stable checkout and image while an instance is running. Do not modify launcher source in place during a session, edit the submodule to chase upstream main, or inject new code into a live container. Change configuration through the exported environment and a deliberate stop/start; code changes require a rebuild.

## Configuration workflow

Copy `.env.example` to `.env`, edit `MODEL_DIR`, then explicitly export the file in the shell that invokes the launcher:

```bash
set -a
source .env
set +a
```

`.env` is shell code, not an untrusted input format. The launcher **does not source it automatically**, so an intentional command-scoped override takes precedence:

```bash
SPEC=0 ./deploy.sh start
```

All file paths internal to the launcher resolve relative to `deploy.sh`, not the caller's working directory. Examples below assume the checkout is the shell's current directory. `MODEL_DIR` must be an existing checkpoint directory; an absolute path avoids ambiguity. A useful generic location is `/srv/models/DeepSeek-V4.1-Flash`, which may be a shared-storage mount. Mount the actual storage first; an empty mount point is not a usable checkpoint. Model files are exposed read-only at `/model` inside the container and never copied into an image.

### Operational variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_DIR` | **Required; no default** | Existing checkpoint directory on the Docker host, mounted read-only at `/model`. |
| `IMAGE` | `deepseek-v41-flash-sm120:latest` | Image tag for build, launch, and image validation. Use a distinct tag for rollback builds. |
| `BUILD_TARGET` | `runtime` | Docker stage to build: native `runtime` or fallback `hybrid`. |
| `NAME` | `dsv41` | Container name; lifecycle commands affect only this name and protect unmanaged containers. |
| `BIND_HOST` | `127.0.0.1` | Host-side published-address binding. No API authentication is configured. |
| `PORT` | `30000` | Published API port. |
| `TP` | `8` | Tensor-parallel size. This kit's measured configuration uses eight GPUs. |
| `EP` | `8` | Expert-parallel size; `0` omits expert-parallel configuration for TP-only Marlin rollback. |
| `MOE_BACKEND` | `flashinfer_mxfp4` | MoE backend. `marlin` is the documented alternative. |
| `SPEC` | `1` | Enable static DSpark speculation; set `0` for non-speculative serving. |
| `DSPARK_BLOCK` | `3` | Static speculative block size. `5` is a code-oriented measured alternative, not the default. |
| `CONTEXT_LEN` | `1048576` | Configured token limit, **not** a proven full-window workload capacity. |
| `MEM_FRACTION` | `0.80` | Static GPU-memory fraction. KV allocation does not guarantee every request mixture fits. |
| `MAX_RUNNING` | `32` | Maximum simultaneously running requests; not a guarantee of 32 full-window requests. |
| `DECODE_STEPS` | `4` | Continuous decode steps. |
| `RANDOM_SEED` | `599261575` | Server seed; not a promise of bitwise reproducibility across backends. |
| `NCCL_P2P_LEVEL` | `PHB` | Measured host-topology policy. Host-specific performance may differ; not a universal NCCL recommendation. |
| `DSV41_SM120_DISABLE` | `1` | Disable the legacy TileLang sparse-attention override. Native image uses `1`; hybrid rollback requires `0`. Does **not** disable MoE or dense-FP8 tuning. |
| `DSV41_SM120_FP8_DISABLE` | `0` | Keep runtime dense block-FP8 tuning enabled. `1` disables that adapter; the quoted performance does not apply. |
| `STRICT` | `1` | Raise on a legacy TileLang attention fallback failure. It is not a global startup-hook guarantee; native attention normally bypasses that adapter. |
| `BASE_URL` | `http://127.0.0.1:${PORT}` | API endpoint used by operational health/smoke checks; set explicitly when accessing a protected forwarded endpoint. |
| `DRY_RUN` | `0` | `1` makes `start` print its Docker argv without launching or stopping a container; skips live-resource exclusivity checks. |

The following policy is built into the recipe, rather than a second undocumented family of tuning switches:

- Tool-call parser: `deepseekv41`; reasoning parser: `deepseek-v41`.
- Static DSpark uses `SGLANG_SIMULATE_ACC_LEN=-1`; SPS recording is disabled (`0`). No compact/ragged DSpark mode is enabled.
- Target and draft CUDA graphs are enabled by default. **Full-model prefill graphs remain disabled by model policy.** A direct attention-kernel graph test does not override that policy.
- No forced thinking mode. A request opts into reasoning with `reasoning_effort: "high"`.
- Runtime imports use `PYTHONPATH=/opt/dsv41/runtime:/sgl-workspace/sglang/python`, avoiding an older SGLang namespace shadowing the pinned source.
- The checkpoint mount is read-only; runtime adapters are baked into the image at `/opt/dsv41/runtime`, not mounted from a mutable host experiment directory.
- The inherited startup hook logs import failures instead of aborting. Image integrity checks do not prove GPU-side adapter activation; inspect startup warnings and perform serving validation on the intended GPU model.
- The served model name is `deepseek-v41-flash`; chunked prefill is `2048`, the DeepSeek-V4 FP4 indexer is enabled, and the watchdog timeout is `3600` seconds.
- The launcher enables `--trust-remote-code`. Use only a trusted checkpoint/model code; a read-only mount is not a Python sandbox.

`.env.example` gives the standard configuration, not credentials. No `API_KEY` variable or implied authentication is introduced.

## Pinned image construction

Run all image operations through the one entry point:

```bash
./deploy.sh init
./deploy.sh build
./deploy.sh check-image
```

`init` initializes the `sglang` submodule and checks its HEAD against [versions.json](../versions.json). It deliberately selects the preview commit `da64c5cbb8cf6bfd39be19da43573fdfd484c43a`, which matches the historical serving source. Do not replace it with the fork's newer main-based contribution branch and assume equivalence.

The Dockerfile has two stages:

1. **`hybrid`** starts from the digest-pinned SGLang image, pins TileLang `0.1.14` and apache-tvm-ffi `0.1.11`, copies `sglang/python` into `/sgl-workspace/sglang/python`, and installs the packaged runtime adapters. It preserves the base Torch/SGLang installation instead of upgrading them through pip. With the legacy attention override enabled, this reproduces the fallback native-decode/TileLang-prefill integration.
2. **`runtime`** adds the minimal FlashInfer `0.6.18` sparse-MLA prefill dispatch backport from [PR #5121](https://github.com/flashinfer-ai/flashinfer/pull/5121), pinned at `24804035ad9d8372e5883007f2d391986f17314d`. It compiles the native module in release mode for `12.0f`, with `MAX_JOBS=4` and debug disabled. This is the default image stage.

The base is `lmsysorg/sglang@sha256:4a5d132a06a77c8331e15845f2e925adc788b00105097ad55409afa3f4fa4860`. Package versions, source hashes, module identity, and **historical** image digests are recorded in `versions.json`. Historical digests identify the measured artifacts; they are not a promise that a local rebuild produces the same image digest or an instruction to pull a private historical image.

Runtime adapters are still required by this pinned preview integration. The newer [SGLang PR #39872](https://github.com/sgl-project/sglang/pull/39872) contributes narrower block-FP8 dispatch/configuration changes on top of the kernel work in [#39657](https://github.com/sgl-project/sglang/pull/39657). Related bring-up work includes [#38970](https://github.com/sgl-project/sglang/pull/38970) for low-ratio indexer metadata and [#38969](https://github.com/sgl-project/sglang/pull/38969) for unsupported-prefill fallback. These links explain provenance and boundaries; this kit does not assert complete stock-main support or a passed full upstream CI run for #39872.

## Start, health, logs, and stop

Inspect the actual launch command safely first:

```bash
DRY_RUN=1 ./deploy.sh start
```

This can run on a busy host. It is a command preview, not proof that the checkpoint, GPU resources, driver, or image can serve the model.

For a real launch, export `.env` inside the tmux shell that will run the server, then:

```bash
./deploy.sh start
```

Alternatively, from the already configured shell use `tmux new-session -s dsv41 './deploy.sh start'`. Existing tmux servers may preserve an older environment; explicit export inside the tmux shell is the reliable option. Detach with `Ctrl-b`, then `d`. The command stays in the foreground and tees output to ignored `logs/` using a timestamped file plus `latest.log`.

The launcher takes a host `flock`, refuses a container already named `NAME`, and refuses busy compute GPUs before a real start. It does not stop unrelated containers or GPU jobs. This is a safety check, not a cluster scheduler or an atomic reservation against non-cooperating applications. The launch uses `--gpus all`, automatic container removal, host IPC, 32 GiB shared memory, unlimited memlock, a 67,108,864-byte stack limit, and `SYS_PTRACE`.

From a second configured shell:

```bash
./deploy.sh status
./deploy.sh logs
```

`status` reports the named container state and checks HTTP `/health`, exiting nonzero for an unavailable/unhealthy instance. A running container may still be loading weights; readiness is a separate condition. `logs` follows the named container's recent output while it exists and falls back to the saved `logs/latest.log` after container removal. Health checks do not generate tokens or establish model quality.

```bash
./deploy.sh smoke
```

`smoke` invokes the packaged API client against `BASE_URL` and stores its report at `logs/smoke-api.json`. It requires a healthy GPU-backed server and makes real inference requests. Inspect that report for your own deployment; historical parser results do not establish that a local rebuild passed. This is not a benchmark, full-context stress test, or a replacement for workload-specific acceptance testing.

```bash
./deploy.sh stop
./deploy.sh restart
```

Both operations protect unrelated containers by requiring this kit's management label. `stop` affects only `NAME`. `restart` stops that instance and performs a new foreground launch with the current settings. There is no global kill/prune operation. Stop before changing image/backend settings; merely rebuilding or changing `.env` does not update an already running container.

### CPU validation is not serving validation

`check-image` launches a short-lived CPU-only container running `/opt/dsv41/verify_image.py`. It checks package/source identity, parser availability, and the backport manifest as appropriate to the selected stage, without loading weights or requesting GPUs. It does **not** execute the SM120 kernels, allocate the KV pool, capture full-model graphs, prove driver compatibility, or send inference. A successful local `start`, healthy endpoint, and `smoke` are distinct steps; none of them by themselves reproduce the historical performance/quality matrix.

## Parser and reasoning request shapes

The launch explicitly selects `deepseekv41` for tools and `deepseek-v41` for reasoning. Send OpenAI-compatible chat-completion requests to `/v1/chat/completions` using the model identifier reported by `/v1/models`; `deploy.sh smoke` exercises the packaged API path. The examples below are **JSON request shapes**, not extra shell commands or benchmark records.

A normal request leaves reasoning opt-in unset:

```json
{
  "model": "deepseek-v41-flash",
  "messages": [{"role": "user", "content": "Explain tensor parallelism in two sentences."}],
  "max_tokens": 256,
  "temperature": 0
}
```

The launcher loads `/model` and advertises `deepseek-v41-flash`; if your proxy intentionally remaps that identifier, use its advertised identifier. Reasoning is enabled per request, not globally forced by deployment:

```json
{
  "model": "deepseek-v41-flash",
  "messages": [{"role": "user", "content": "Compare two ways to partition this matrix multiplication and explain the trade-off."}],
  "reasoning_effort": "high",
  "max_tokens": 1024,
  "temperature": 0
}
```

A tool-call request supplies a schema. The parser converts the model's tool syntax into structured API tool calls; the server does **not** execute the tool:

```json
{
  "model": "deepseek-v41-flash",
  "messages": [{"role": "user", "content": "What is the weather in Paris? Use the weather tool."}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get current weather for a city.",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }
  }],
  "tool_choice": "auto",
  "max_tokens": 512,
  "temperature": 0
}
```

Validate tool arguments and authorization in the client before any side effect. For streaming, consume the structured deltas rather than assuming every chunk contains complete JSON. Historical parser checks covered eleven API scenarios; those checks were separate from the older throughput matrix.

## Rollback choices

Do not change a backend under a running process. Stop the named instance, preserve any logs you need, and select one coherent configuration. Command-scoped overrides below do not rewrite `.env`; repeat them after a future stop/start or record the intended configuration in your private `.env`.

### Disable speculation only

Keep the native image, EP8, FlashInfer experts, and dense-FP8 tuning:

```bash
./deploy.sh stop
SPEC=0 ./deploy.sh start
```

This is useful when aggregate high-concurrency prose throughput matters more than C1 latency. Historical C32 prose was slightly slower with DSpark3 than without it.

### Native attention with TP-only Marlin experts

Keep the default native `runtime` image and native attention; remove expert parallelism and speculation:

```bash
./deploy.sh stop
MOE_BACKEND=marlin EP=0 SPEC=0 DSV41_SM120_DISABLE=1 ./deploy.sh start
```

This is a backend alternative, not a guarantee of equivalent throughput or numerical output. See the separate expert-kernel error discussion in [RESULTS.md](RESULTS.md).

### Hybrid native-decode/TileLang-prefill image

Build a distinct image tag so the native image remains available:

```bash
BUILD_TARGET=hybrid IMAGE=deepseek-v41-flash-sm120:hybrid ./deploy.sh build
BUILD_TARGET=hybrid IMAGE=deepseek-v41-flash-sm120:hybrid ./deploy.sh check-image
./deploy.sh stop
IMAGE=deepseek-v41-flash-sm120:hybrid MOE_BACKEND=marlin EP=0 SPEC=0 DSV41_SM120_DISABLE=0 CONTEXT_LEN=393216 MEM_FRACTION=0.80 ./deploy.sh start
```

This enables the legacy attention override and uses TP-only Marlin without speculation. **Hybrid + DSpark is not validated.** Do not simply point native defaults at the hybrid image: the native-preferred attention policy and speculative settings are different. Return to the normal image by stopping the instance and launching with the default image/settings exported from the original `.env`.

DSpark block sizes 1/3/5/8 were screened; `3` is the balanced default and `5` a code-oriented alternative. Compact/ragged DSpark failed an Engram equal-block assertion. It is not a supported rollback/performance switch; do not bypass that assertion.

## Exposure and local data

The default localhost binding is deliberate. No authentication is enabled by the launch recipe. Expose it externally only through a trusted, authenticated proxy with network access controls. Do not assume a private network, a renamed model, or an obscure port provides authorization. Protect prompts, outputs, and shared model files independently of API access.

Keep `.env`, generated logs, caches, and checkpoints out of version control and image build contexts. Review any diagnostic artifact before sharing it. This repository intentionally contains generic paths and sanitized measurement summaries instead of raw server logs or private infrastructure addresses.
