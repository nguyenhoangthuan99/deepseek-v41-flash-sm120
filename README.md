# DeepSeek-V4.1-Flash on 8× SM120

A pinned build and deployment kit for **eight NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs** (approximately 96 GiB each), using the existing [DeepSeek-V4.1-Flash checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash). The default configuration uses TP8/EP8, FlashInfer MXFP4 experts, tuned block-FP8 dense layers, and static DSpark speculation with block size 3.

This repository packages the integration used in historical serving experiments. **It is not a claim that a newly rebuilt image has passed a fresh full-model benchmark, that a full 1,048,576-token request has been validated, or that stock SGLang main supports this complete configuration.** See [results and limitations](docs/RESULTS.md) and the [machine-readable measurements](results/measurements.json).

## Engines

Two serving engines are packaged, each with its own Dockerfile and launcher:

| Engine | Dockerfile | Launcher | Submodule |
| --- | --- | --- | --- |
| SGLang (default, port 30000) | `Dockerfile.sglang` | `deploy.sh` | `sglang/` @ `da64c5cb` |
| vLLM (port 30100) | `Dockerfile.vllm` | `serve-vllm.sh` | `vllm/` @ [`4980e062`](https://github.com/nguyenhoangthuan99/vllm/commit/4980e06225532feca2ebad8c834dab3f7739acb9) |

The vLLM engine ships the best measured SM120 configuration: DSpark K3
speculative decoding, FULL_DECODE_ONLY CUDA graphs, 1,048,576-token context,
FlashInfer b12x/split-K MXFP8 linear kernels, and DeepGEMM MoE small-M
alignment 64 (`--kernel-config` in `serve-vllm.sh`). Measured on the matched
cold workload (168 requests, medians): decode 179.9 / 433.6 / 914.5 / 1226.9
tok/s at concurrency 1/4/16/32 versus 166.9 / 386.4 / 811.3 / 1127.7 untuned.
All changed packages are pinned in this repository: the vLLM fork source
(submodule, including its custom C++/CUDA ops and vendored DeepGEMM) and the
three FlashInfer SM120 patches plus the DeepGEMM page-32 patch (baked in by
`Dockerfile.vllm` from `vllm/tools/`). See the header of `Dockerfile.vllm`
for the two-image self-build path (`BASE_IMAGE`/`EXT_IMAGE`) and
`versions.json` for pins.

```bash
# vLLM engine
docker build -f Dockerfile.vllm -t dsv41-vllm:sm120 .
MODEL_DIR=/srv/models/DeepSeek-V4.1-Flash ./serve-vllm.sh
```

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

Both Docker targets were built from this repository and passed CPU-only package/source/parser checks; the native target's compiled backport was verified too. The Bash commands, existing-container protection and managed-stop guard were exercised. The packaged API client passed eleven scenarios against the existing validated deployment without restarting it. See [package-validation.json](results/package-validation.json). A full model was not launched from these newly packaged images.

## Prerequisites

- Linux x86-64 with eight available RTX PRO 6000 Blackwell Server Edition GPUs and an NVIDIA driver compatible with the pinned CUDA 13 runtime.
- Docker with BuildKit and the NVIDIA Container Toolkit configured for GPU containers; permission to use Docker. Building the image does not require model weights or a GPU.
- Git, Bash, `flock` (util-linux), `nvidia-smi`, and `tmux`; Python 3.9+ for preflight and the API smoke client. The Docker build supplies its own compiler and Python build dependencies.
- Network access to this public repository and the pinned source/image/package dependencies during the build.
- An existing, complete checkpoint on locally mounted or shared storage with sufficient RAM/storage bandwidth for loading. **You supply `MODEL_DIR`; the kit does not download weights.** Reserve all eight GPUs and adequate host RAM/disk space for the checkpoint and image build.

## Clone, build, and launch

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
- The container receives all GPUs, host IPC, 32 GiB shared memory, unlimited memlock, a 64 MiB stack limit, and `SYS_PTRACE`. Run only trusted images on a dedicated or carefully controlled host. A host `flock` coordinates this launcher; it is not a scheduler for unrelated GPU workloads.
- The launcher enables `--trust-remote-code`; use only a trusted checkpoint and trusted model code. Read-only weights do not sandbox Python code.
- Do not edit a running launcher's source or patch a live container. Stop the named instance, rebuild when source changes, run `check-image`, and restart deliberately. The launcher does not hot-reload repository changes.

## Further reading

- [Deployment guide](docs/DEPLOYMENT.md): every setting, image stages, API parsers, operational behavior, and rollback.
- [Results](docs/RESULTS.md): historical performance, quality checks, context limits, kernel evidence, and failed experiments.
- [Measurements JSON](results/measurements.json): structured historical observations, not a certification of the locally built image.

All build, container-lifecycle, image-validation, and smoke operations are implemented by the single entry point `deploy.sh`; no separate wrapper scripts or edits to the SGLang submodule are required.
