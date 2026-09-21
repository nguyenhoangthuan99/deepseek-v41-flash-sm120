# DeepSeek-V4.1-Flash on 8× SM120

A pinned build and deployment kit for **eight NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs** (approximately 96 GiB each), using the existing [DeepSeek-V4.1-Flash checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash). SGLang remains the default recipe; the current validated VM100 deployment uses **vLLM on port 30000**, serving **`deepseek-v41-flash`** with DSpark K3 and explicit tool/reasoning parsers.

The repository preserves historical SGLang measurements alongside a successful full-model vLLM deployment and fresh API/benchmark evidence. **The vLLM deployment used retained fork-built extensions, not a verified clean public self-build.** Earlier vLLM 1M-context exercise on VM106 is separate from VM100: VM100's 1,048,576-token configuration was verified, but no 1M request was made in this deployment check. No complete stock-SGLang-main support is claimed. See [results and limitations](docs/RESULTS.md).

## Engines

Two serving engines are packaged, each with its own Dockerfile and launcher:

| Engine | Dockerfile | Launcher | Submodule |
| --- | --- | --- | --- |
| SGLang (default recipe, port 30000) | `Dockerfile.sglang` | `deploy.sh` | `sglang/` @ `da64c5cb` |
| vLLM (launcher default 30100; validated deployment 30000) | `Dockerfile.vllm` | `serve-vllm.sh` | `vllm/` @ [`4980e062`](https://github.com/nguyenhoangthuan99/vllm/commit/4980e06225532feca2ebad8c834dab3f7739acb9) |

### Current vLLM deployment

The measured configuration uses DSpark K3, FULL_DECODE_ONLY CUDA graphs,
1,048,576-token configured context, FlashInfer b12x/split-K MXFP8 linear
kernels, and DeepGEMM MoE small-M alignment 64. Auto tool choice is enabled;
both vLLM parsers are named `deepseek_v41`.

The final parser-enabled deployment passed **168/168 requests across 21/21
scenarios, three repeats**, on the matched raw `/v1/completions` workload.
Decode medians (128 input / 256 output tokens) were **201.4606 / 480.4955 /
929.0074 / 1259.3453 tok/s** at concurrency 1/4/16/32, versus historical
SGLang **177.0261 / 433.3856 / 900.7754 / 1179.9050**. These are end-to-end
aggregate output rates, not tool/reasoning performance or a fresh alternating
A/B. Prefill median TTFT was **162.56 / 517.88 / 869.81 ms** at 1024/4096/7168
input tokens. See [benchmark metrics](results/vllm-vm100-benchmark.json) and
[deployment/debug summary](results/vllm-vm100-deployment.json).

Before the parser restart, smoke passed 7/7 and concurrent retrieval 32/32.
The subsequent nine-case parser probe had eight strict passes and one
structured named-call result with a finish-reason caveat; the packaged
`smoke_api.py --engine vllm` client later passed all **11 scenarios** against
the live deployment. vLLM emits `message.reasoning` / `delta.reasoning`, not
SGLang's `reasoning_content`. Forced named calls returned valid `tool_calls`
with `finish_reason="stop"`; auto/required returned `"tool_calls"`. See
[API validation](results/vllm-vm100-api-validation.json).

With a compatible image already available, launch from a persistent terminal
after stopping the other engine and preserving its rollback configuration:

```bash
mkdir -p /mnt/nas/.cache/dsv41-sm120/flashinfer /mnt/nas/.cache/dsv41-sm120/home
MODEL_DIR=/srv/models/DeepSeek-V4.1-Flash \
  CACHE_DIR=/mnt/nas/.cache/dsv41-sm120 PORT=30000 BIND_HOST=127.0.0.1 \
  ./serve-vllm.sh
```

The cache root must be writable by the launcher user. The launcher mounts its
`flashinfer/` and `home/` subdirectories and uses Docker `--init` for child
reaping. The replacement deployment used `PORT=30000 BIND_HOST=0.0.0.0`;
keep localhost binding unless authenticated network controls protect access.
Both engines need all eight GPUs; different ports do not allow co-residency.
See [deployment and rollback instructions](docs/DEPLOYMENT.md).

### vLLM build prerequisites and limits

The local `BASE_IMAGE` and `EXT_IMAGE` defaults are **not publicly pullable
images**. The validated image reused retained extensions compiled from the
pinned fork; stock upstream vLLM extensions lack required custom ops.
`Dockerfile.vllm` overlays the fork's Python source and applies three
FlashInfer SM120 patches. The DeepGEMM page-32 patch and build script are
**copied for base rebuilds, not applied by the overlay**.

The fork Dockerfile currently pins FlashInfer **`0.6.18.post1`**, while the
overlay asserts **`0.6.18`** from `versions.json`. A clean self-build remains
**unverified**: matching runtime, FlashInfer patch/version pins, fork-compiled
extensions, and the required DeepGEMM base build must be established before
the documented two-image build path can be treated as reproducible. The
Dockerfile's self-build comments are not proof that a clean build passed.

An [upstream contribution assessment](docs/RESULTS.md#upstream-contribution-assessment)
identifies the FP4 quantization-group fix as the strongest first PR candidate,
separates performance work and existing upstream overlap, and records the
remaining validation requirements. No upstream PR or issue has been opened.

## What is pinned

[versions.json](versions.json) records the source, dependency, backport, and historical image identities.

| Component | Pin |
| --- | --- |
| SGLang submodule | [`da64c5cbb8cf6bfd39be19da43573fdfd484c43a`](https://github.com/nguyenhoangthuan99/sglang/commit/da64c5cbb8cf6bfd39be19da43573fdfd484c43a) |
| Base image | `lmsysorg/sglang@sha256:4a5d132a06a77c8331e15845f2e925adc788b00105097ad55409afa3f4fa4860` |
| TileLang / apache-tvm-ffi | `0.1.14` / `0.1.11` |
| FlashInfer | `0.6.18` plus the minimal [PR #5121](https://github.com/flashinfer-ai/flashinfer/pull/5121) backport from `24804035ad9d8372e5883007f2d391986f17314d` |
| Native sparse-MLA compilation | CUDA architecture `12.0f`, release mode, `MAX_JOBS=4`, debug disabled |

The submodule deliberately uses the **tested preview commit**, not the newer main-based [SM120 contribution in SGLang PR #39872](https://github.com/sgl-project/sglang/pull/39872). Runtime adapters in `runtime/` remain necessary to reproduce this preview integration. The contribution PR is narrower than this deployment kit; its kernel results are not full-model validation of current main, and no successful full upstream CI result is asserted here.

The [source-parity record](results/source-parity.json) confirms that all **4,770 tracked files in SGLang's `python/` tree** at the pinned commit matched the original deployed native image, with zero missing files and zero hash mismatches. This is source-parity evidence, not a fresh rebuilt-image serving test.

During initial **SGLang packaging**, both Docker targets were built from this repository and passed CPU-only package/source/parser checks; the native target's compiled backport was verified too. The Bash safety controls and eleven API scenarios against the then-existing server were exercised without a restart. [package-validation.json](results/package-validation.json) did not claim a new full-model run of those packaged SGLang images. Later SGLang serving experiments and the current vLLM full-model deployment are recorded separately in [RESULTS.md](docs/RESULTS.md).

## SGLang prerequisites

- Linux x86-64 with eight available RTX PRO 6000 Blackwell Server Edition GPUs and an NVIDIA driver compatible with the pinned CUDA 13 runtime.
- Docker with BuildKit and the NVIDIA Container Toolkit configured for GPU containers; permission to use Docker. Building the image does not require model weights or a GPU.
- Git, Bash, `flock` (util-linux), `nvidia-smi`, and `tmux`; Python 3.9+ for preflight and the API smoke client. The Docker build supplies its own compiler and Python build dependencies.
- Network access to this public repository and the pinned source/image/package dependencies during the build.
- An existing, complete checkpoint on locally mounted or shared storage with sufficient RAM/storage bandwidth for loading. **You supply `MODEL_DIR`; the kit does not download weights.** Reserve all eight GPUs and adequate host RAM/disk space for the checkpoint and image build.

## SGLang: clone, build, and launch

Clone and enter the repository:

```bash
git clone --recurse-submodules https://github.com/nguyenhoangthuan99/deepseek-v41-flash-sm120.git
cd deepseek-v41-flash-sm120
cp .env.example .env
```

Edit `.env` so `MODEL_DIR` points to your existing checkpoint directory, for example `/srv/models/DeepSeek-V4.1-Flash`. The file contains shell assignments: load only a trusted local file. Export it explicitly; `deploy.sh` never sources `.env` implicitly.

```bash
set -a
source .env
set +a
./deploy.sh init
./deploy.sh build
./deploy.sh check-image
DRY_RUN=1 ./deploy.sh start
```

`init` initializes the submodule and checks its pinned HEAD. The default build selects Docker's `runtime` stage; `check-image` runs short-lived **CPU-only** package/source/parser/backport checks, not inference. `DRY_RUN=1` prints the actual container command without launching or stopping anything, even on a busy host.

Start in a persistent terminal session, from the same exported shell:

```bash
tmux new-session -s dsv41 './deploy.sh start'
```

An existing tmux server can retain an older environment. In that case, start a tmux shell, enter the repository, explicitly export `.env` there using the commands above, and run `./deploy.sh start` inside it. Detach with `Ctrl-b`, then `d`. The launcher stays in the foreground, recording a timestamped log and `logs/latest.log`. It refuses an existing named container or busy compute GPUs; it does not evict another workload.

From another shell, enter the checkout and export `.env` as above before using:

```bash
./deploy.sh status
./deploy.sh logs
./deploy.sh smoke
```

The API is bound to **`127.0.0.1:30000` by default**. Wait for model loading/readiness before `smoke`; smoke sends real inference requests and writes `logs/smoke-api.json`. It is separate from image validation and does not reproduce the throughput matrix or test the full context window.

```bash
./deploy.sh stop
./deploy.sh restart
```

`stop` targets only `NAME` (default `dsv41`); `restart` stops that named container and launches again in the foreground with the current exported settings. Neither operation performs a global container/GPU cleanup.

## Safety and privacy

- No weights, credentials, caches, raw server logs, or machine-specific storage paths belong in this repository or image. `.env` and generated `logs/` are local operational state; keep them private. Logs and smoke responses can contain user/model content.
- The checkpoint is bind-mounted **read-only at `/model`**, never copied into an image. Shared storage must already be mounted on the Docker host and accessible to the container.
- There is **no built-in API authentication** in this recipe. Keep localhost binding unless a trusted, authenticated reverse proxy and appropriate network controls protect the endpoint. Binding to all interfaces is not an authentication mechanism.
- SGLang receives all GPUs, host IPC, 32 GiB shared memory, unlimited memlock, a 64 MiB stack limit, and `SYS_PTRACE`; its host `flock` is not a scheduler for unrelated workloads. vLLM uses all GPUs by default, host IPC, 64 GiB shared memory, and Docker `--init`. Run only trusted images on a dedicated or carefully controlled host.
- Both launchers enable `--trust-remote-code`; use only a trusted checkpoint and trusted model code. Read-only weights do not sandbox Python code.
- Do not edit a running launcher's source or patch a live container. Preserve rollback configuration before stopping, rebuild when source changes, and restart deliberately. `deploy.sh check-image` is SGLang-specific; neither launcher hot-reloads repository changes.

## Further reading

- [Deployment guide](docs/DEPLOYMENT.md): every setting, image stages, API parsers, operational behavior, and rollback.
- [Results](docs/RESULTS.md): current vLLM deployment, historical performance, quality checks, context limits, kernel evidence, and failed experiments.
- [Measurements JSON](results/measurements.json): structured historical observations, not a certification of the locally built image.

SGLang build, lifecycle, image-validation, and smoke operations use `deploy.sh`. vLLM uses `Dockerfile.vllm`, `serve-vllm.sh`, and `scripts/smoke_api.py --engine vllm`; see the deployment guide for commands and engine-specific limits.
