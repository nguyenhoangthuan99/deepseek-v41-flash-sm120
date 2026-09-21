# Results and validation boundaries

Current vLLM deployment evidence is recorded in the sanitized
[deployment/debug summary](../results/vllm-vm100-deployment.json),
[API validation](../results/vllm-vm100-api-validation.json), and
[benchmark metrics](../results/vllm-vm100-benchmark.json). Historical SGLang
measurements remain in [measurements.json](../results/measurements.json).
Operational commands and build prerequisites are in
[DEPLOYMENT.md](DEPLOYMENT.md).

Keep four kinds of evidence distinct:

1. **Current VM100 full-model vLLM deployment:** successful normal starts,
   live API checks, and a matched raw-completions benchmark on the final
   parser configuration. This used retained fork-built extension artifacts,
   not a verified clean rebuild from publicly pullable image dependencies.
2. **Historical full-model serving:** earlier SGLang performance/quality
   observations and the vLLM tuning/long-context exercise on VM106. Neither
   is a fresh alternating A/B against the current VM100 deployment.
3. **Isolated kernel validation:** direct numerical and CUDA-graph tests,
   sometimes against newer contribution trees, without loading the model.
4. **Packaged-image verification:** image construction and CPU
   package/source/parser/manifest checks. `deploy.sh check-image` is
   SGLang-specific and does not serve weights or execute GPU kernels.

No complete stock-main support or full upstream CI success is claimed.
Earlier vLLM 1M-context exercise on VM106 is real but separate: **VM100's
1,048,576-token configuration was verified; no request at 1M was made in
this deployment check.** Historical SGLang validation reached 117,286 tokens.

## Current vLLM deployment on VM100

The live endpoint serves **`deepseek-v41-flash` on port 30000** with TP8,
DSpark K3, FULL_DECODE_ONLY CUDA graphs, configured context 1,048,576,
FlashInfer b12x/split-K MXFP8 linear kernels, and DeepGEMM MoE small-M
alignment 64. Auto tool choice is enabled and both parsers are
`deepseek_v41`. The launcher still defaults to port 30100; port 30000 is
an explicit deployment override.

### Matched raw-completions benchmark

**168/168 requests and 21/21 scenarios passed**, with **three repeats**.
The shared synthetic token-ID workload SHA-256 is
`cadc39231f472c1221324539949cec41871e558f495bf37d0efdbaf3791e31bf`;
the workload is preserved as
[matched-cold-workload.json](../results/matched-cold-workload.json).
The client used raw **`/v1/completions`**, not chat/tool/reasoning requests.
Decode requests used **128 input / 256 output tokens**. Rates below are
median **end-to-end aggregate output tokens / wall seconds**, measured by a
localhost client inside the serving container, not per-request decode rates.

| Concurrency | Current vLLM tok/s | Historical SGLang tok/s | Difference |
| ---: | ---: | ---: | ---: |
| 1 | 201.4606 | 177.0261 | +13.80% |
| 4 | 480.4955 | 433.3856 | +10.87% |
| 16 | 929.0074 | 900.7754 | +3.13% |
| 32 | 1259.3453 | 1179.9050 | +6.73% |

| Prefill input tokens | Current vLLM median TTFT |
| ---: | ---: |
| 1024 | 162.56 ms |
| 4096 | 517.88 ms |
| 7168 | 869.81 ms |

This benchmark ran on the **final parser-enabled configuration**, but it
does **not measure tool or reasoning performance**. The SGLang comparison
is historical, not a freshly rerun alternating A/B, and the deployed engine
configurations differ. It does not isolate a causal engine/kernel speedup.
Other historical tables below use different protocols; do not mix them into
this comparison. Repeating the same workload invocation warms measured
prefixes and is not automatically another cold-cache measurement.

### Live API validation and deployment timeline

- Before the parser restart: **7/7 smoke** and **32/32 concurrent retrieval**
  passed. These checks are not relabeled as final-parser validation.
- After enabling the explicit parsers: a **nine-case probe** recorded eight
  strict passes (auto, auto-stream, reasoning-tool, reasoning-tool-stream,
  required, reasoning, reasoning-stream, tool-roundtrip). The ninth, forced
  named-tool choice, returned valid structured `tool_calls` with
  `finish_reason="stop"`, not the strict `"tool_calls"` finish reason.
- vLLM uses **`message.reasoning` / `delta.reasoning`**; SGLang uses
  **`reasoning_content`**. Auto/required calls used
  `finish_reason="tool_calls"`. A named call must not be discarded solely
  because its finish reason is `stop`.
- A separate, later run of the packaged
  **`scripts/smoke_api.py --engine vllm` passed 11/11 scenarios** against the
  live deployment. This is distinct from the earlier nine-case probe and
  does not erase its named-call caveat.
- The final-parser benchmark passed **168/168**; `/health` returned **200**.
  No 1M-token request was part of this VM100 check.

Startup debugging first encountered a cache permission failure, then a
later **CUDA illegal access during graph warmup** followed by a **secondary
NCCL error**. Fresh NAS caches under `/mnt/nas/.cache/dsv41-sm120` were
populated during a diagnostic `CUDA_LAUNCH_BLOCKING` run; **two normal-mode
starts** then succeeded with the same cache and API checks. Normal serving
has blocking disabled; Docker `--init` provides child-process reaping.
**The root cause was not isolated.** Recovery does not prove cache
incompatibility was the cause of the illegal access.

The original SGLang container used `--rm` and was auto-removed. Its exact
image/arguments/environment/GPU requests/mounts/IPC/limits/ports were
recreated from saved inspection data into a **stopped rollback container
with `AutoRemove=false`**. This is not a fresh validation of rollback
serving. Save inspection data privately **before** stopping an auto-remove
container; see the generic rollback procedure in the deployment guide.

### Build provenance is not clean-build proof

The validated vLLM image used retained extensions compiled from the pinned
fork. The current local `BASE_IMAGE` / `EXT_IMAGE` defaults are not publicly
pullable. The fork Dockerfile pins FlashInfer **0.6.18.post1**, whereas the
overlay asserts **0.6.18**. A clean self-build remains **unverified** and
requires matching dependency/patch pins, compatible runtime and fork-built
extensions, and the required DeepGEMM base build. The overlay applies three
FlashInfer patches but only **copies** the DeepGEMM page-32 patch/build
script; it does not apply/build that patch.

## Historical vLLM tuning and long context on VM106

The earlier [vLLM tuning A/B](../results/vllm-tuned-ab-summary.json) compared
the kernel overrides against default kernel selection with other launch
arguments unchanged: DSpark K3, FULL_DECODE_ONLY, TP8, block size 64,
GPU-memory utilization 0.85, configured context 1,048,576, and port 30100.
Its matched workload used 168 requests and three repeats per concurrency.

| Concurrency | Historical baseline tok/s | Historical tuned tok/s |
| ---: | ---: | ---: |
| 1 | 166.86 | 179.85 |
| 4 | 386.36 | 433.56 |
| 16 | 811.26 | 914.47 |
| 32 | 1149.67 | 1226.94 |

The baseline C32 **median is 1149.67**, not the previously quoted 1127.7.
These values are historical tuning results, not the current VM100 benchmark.

Historical prefill comparisons use the **first invocation only**, taking
the median of its three unique-prefix repeats per input size. The tuned
collection had **four invocations of the same three prefill sizes**, not a
12-size sweep; later invocations reused prefixes and are warm-cache data.

| Input tokens | Historical baseline median TTFT (ms) | Historical tuned median TTFT (ms) |
| ---: | ---: | ---: |
| 1024 | 178.028867 | 175.891351 |
| 4096 | 557.786945 | 547.058299 |
| 7168 | 911.857707 | 928.081635 |

The differences are within about 2%; no material prefill improvement was
established. Historical correctness was **9/10 exact and 5/5 retrieval**,
with the same `exact:reverse` failure as the baseline; smoke passed **7/7**.

Earlier VM106 long-context requests exercised **65,536, 262,144, and
1,048,576 tokens**, with all retrieval checks passing, finite logprobs,
and exact token accounting. Tuned elapsed times were **8.82 / 38.56 /
215.21 seconds**, versus baseline **8.75 / 38.37 / 213.61 seconds**. This
does not validate arbitrary 1M workloads, 32 simultaneous full windows,
or a 1M request on VM100. The earlier tuning server was intentionally
stopped after that A/B.

## Historical SGLang packaging evidence

The [source-parity record](../results/source-parity.json) confirms that
**4,770 tracked files in SGLang's `python/` tree** from the pinned preview
commit matched the original deployed native image: zero missing files and
zero SHA-256 mismatches. It verifies source identity, not the behavior or
image digest of a subsequent rebuild.

The [package-validation record](../results/package-validation.json) covers
the **initial SGLang packaging exercise**: successful `runtime` and `hybrid`
builds, CPU checks of all 4,770 source files and eight runtime extensions in
each image, package/parser/backport integrity, and Bash safety controls.
Its eleven API cases ran against the existing server without a production
restart, not a newly launched packaged SGLang image. Later packaged-image
SGLang serving experiments below and the current vLLM deployment above are
separate evidence; the initial packaging limitation is not a blanket claim
that no full-model serving has since occurred.

## SGLang hardware and historical settings

The measured host used eight NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs (SM120, 188 SMs/GPU, approximately 96 GiB/GPU). The pinned SGLang preview source was `da64c5cbb8cf6bfd39be19da43573fdfd484c43a`; the historical software included PyTorch `2.13.0+cu130`, Triton `3.7.1`, FlashInfer `0.6.18`, and TileLang `0.1.14`. Source/package/backport identities and historical image digests are in [versions.json](../versions.json).

The final native-serving matrix used TP8/EP8, FlashInfer MXFP4 experts, native sparse attention with the minimal FlashInfer PR #5121 dispatch backport, dense block-FP8 tuning, `NCCL_P2P_LEVEL=PHB`, four continuous decode steps, maximum 32 running requests, and static speculation where indicated. DSpark used `SGLANG_SIMULATE_ACC_LEN=-1` and SPS recording disabled. Target/draft decode graphs were enabled; full-model prefill graphs were not enabled.

**The throughput matrix predates both the configured context increase from 393,216 to 1,048,576 and the explicit tool/reasoning parser flags.** Parser checks were a separate later API exercise. Do not relabel this matrix as a benchmark of the exact current launch arguments.

## Historical SGLang warm no-speculation versus DSpark3

Three-round warm aggregate output throughput, in tokens/second:

| Workload | Concurrency | No speculation | DSpark block 3 |
| --- | ---: | ---: | ---: |
| Prose | 1 | 106.73 | 200.79 |
| Prose | 4 | 369.05 | 495.36 |
| Prose | 16 | 926.70 | 970.62 |
| Prose | 32 | 1,394.83 | 1,368.50 |
| Code | 1 | 107.57 | 232.82 |
| Code | 4 | 365.17 | 601.66 |
| Code | 16 | 914.11 | 1,136.35 |
| Code | 32 | 1,368.12 | 1,589.28 |

These are **aggregate output rates**, not per-request decode rates. In particular, C32 does not mean every request decodes at the aggregate rate. Separate C1 decode-only observations were **205.94 tok/s for prose** and **239.71 tok/s for code**; they are not interchangeable with the end-to-end aggregate C1 rows above.

DSpark3 materially helped C1 and the measured code workloads, but it was not a universal improvement: C32 prose decreased from 1,394.83 to 1,368.50 tok/s. Use the non-speculative option when that workload trade-off matters. Warm measurements also do not describe cold model loading, compilation, first-request latency, or every prompt distribution.

Block sizes **1, 3, 5, and 8** were screened. Block 3 was the balanced choice; block 5 was a code-oriented alternative, not the default. Blocks 1 and 8 were not selected as defaults. The JSON retains the recorded DSpark5 rows where available; do not infer unmeasured settings from the DSpark3 table.

### Failed compact/ragged speculation experiment

Compact/ragged DSpark failed an **Engram equal-block assertion**. The failure means that experiment is not a working performance mode in this kit, not that the assertion should be removed. The deployed policy remains static equal-block DSpark. No compact-mode flag, error suppression, or assertion bypass is part of the recipe. Hybrid attention plus DSpark also is not validated.

## NCCL protocol A/B (2026-09-17, packaged-image serving)

A live GPU-trace profile of the packaged-image deployment attributed **53% of C32-decode and 55% of 32k-prefill GPU busy time to NCCL TP8 AllReduce** (`ncclDevKernel_AllReduce_Sum_bf16_RING_LL`). Standalone `torchrun` microbenchmarks at the server's exact payload sizes showed NCCL's protocol tuner selecting `LL` at ~1.3 MB where `Simple` is ~1.7x faster; `LL` halves wire efficiency at large sizes by design.

An in-server A/B (three rounds per leg, medians, cache flushed per cell, 768-token greedy requests; the C4/C16 legs ran after the C1/C32 legs, one restart apart; structured record in [results/nccl-proto-ab.json](../results/nccl-proto-ab.json)):

| Metric | Tuner default | `NCCL_PROTO=Simple` | Change |
| --- | ---: | ---: | --- |
| C1 tok/s | 194.1 | 148.1 | −24% |
| C4 aggregate tok/s | 492.0 | 408.2 | −17% |
| C16 aggregate tok/s | 953.2 | 965.9 | +1% |
| C32 aggregate tok/s | 1,330.5 | 1,518.4 | +14% |
| C32 median per-request tok/s | 44.0 | 51.3 | +17% |
| 32k-prefill TTFT | 5.3 s | 5.3 s | unchanged |

The per-step AllReduce payload scales with batched tokens, so the LL/Simple crossover lands between C16 and C32 (~0.7–1.3 MB). **The kit default keeps NCCL's tuner**: `Simple` wins only near the `MAX_RUNNING=32` saturation point and loses light/moderate concurrency. Deployments with reliably saturated batches can opt in with `NCCL_PROTO=Simple`. The protocol applies per process; NCCL offers no per-message-size override. This is a topology-specific observation on PCIe PHB without NVLink; do not generalize it to NVLink hosts.

The same investigation **rejected** SGLang's custom P2P allreduce for this topology: with the NVLink policy gate bypassed experimentally, its one-shot/two-shot kernels ran 15–30x slower than NCCL over PCIe host bridges, confirming the upstream gate. No gate change ships in this kit.

## Batch-overlap and DP-attention exploration (2026-09-17)

Structured record: [results/overlap-dp-attention-exploration.json](../results/overlap-dp-attention-exploration.json). Same protocol as the NCCL A/B (3-round medians, 768-token greedy requests).

- **Two-batch overlap: blocked.** The pinned tree rejects `--enable-two-batch-overlap` for DeepSeek-V4.1 at startup (`deepseek_v4_hook.py` unsupported-feature raise), and its TBO implementation is prefill-only at this commit (decode raises `NotImplementedError`), so it could not have addressed the decode AllReduce share regardless.
- **Single-batch overlap: skipped.** Its overlap never activates with the `flashinfer_mxfp4` runner, and on SM120 the fallback reroutes shared-expert computation through a dispatcher-hook path untested with this MoE method. Zero expected gain, nonzero correctness risk.
- **DP attention (`--enable-dp-attention --dp-size 8 --enable-dp-lm-head --moe-dense-tp-size 1`): ran, rejected as default.** Measured against the TP8 baseline: C1 −68% (194→62 tok/s), C4 −52%, C16 −9%, **C32 +16%** (1,330→1,546 tok/s), 32k-prefill TTFT 5.3→8.0 s. A single request's attention runs on one GPU under DP, so light-load latency collapses while saturated throughput improves.
- **KV capacity observation (from launch logs):** TP8 attention replicates the KV pool on all 8 ranks (12.53 GB, 6,868,736 full-KV tokens each — one unique copy). DP attention shards it: 5.07 GB and 2,078,720 tokens per rank (replicated attention/dense weights and DP graph reserves shrink the per-rank budget), giving **~16.6 M unique tokens aggregate (≈2.4×)** with a ~2.08 M-token single-request ceiling. DP attention is therefore the right shape for saturated-batch or KV-capacity-bound serving, not for this mixed-load deployment.

These are topology- and workload-specific measurements of the packaged image on the PHB PCIe host; none of them changes the kit defaults.


## Kernel profile and large-M dense-FP8 tuning (2026-09-17)

Per-kernel analysis of the live traces showed the dense block-FP8 GEMM running the tuned kernel only at small M: the shipped configuration table covered M ∈ {1,4,16,32}, so C32 decode (M≈128) and prefill chunks (M≈2048) fell back to the generic Triton kernel — 627 ms of C32-decode and 931 ms of 32k-prefill GPU kernel time. DeepGEMM cannot serve these layers: the checkpoint is quantized with 32×32 weight blocks while DeepGEMM's block-FP8 contract is 1×128/128×128 (SGLang falls back to Triton at `fp8_utils.py` `block_size == [128,128]` gate), which is also why this kit tunes the Triton kernel in the first place.

The existing sweep harness was re-run over M ∈ {64,128,512,2048} (960 benchmarks, 840 bounded numerical/CUDA-graph checks, all passed). Sixteen winners at ≥1.1× over the generic kernel were merged into `runtime/fp8_sm120_configs.json` (isolated-kernel speedups up to 6.0× at (576,5120) M=64 and 3.4× at (1792,5120) M=64). The (25600,6144) shape and most M=2048 points stayed untuned (<1.1×); the adapter preserves the upstream kernel outside the tuned domain.

In-server validation (same 3-round protocol, rebuilt image): **C32 aggregate 1,330.5 → 1,408.7 tok/s (+6%)**, C1 and 32k-prefill TTFT unchanged within run-to-run noise (a C1 recheck spanned 186–203 tok/s across rounds). These are packaged-image serving measurements on the PHB PCIe host, not a general claim.

Remaining profiled candidates, not pursued here: the MXFP4 MoE grouped GEMM (~116 µs mean at C32; FlashInfer-internal, no exposed tuning), the DSpark `_candidate_mask/scores` kernels (~66 ms per C32 window), and the prefill `_hc_mix_stats_partial` hyperconnection kernel (~102 µs mean).


### Custom CUDA / cuBLAS engine comparison (isolated kernels, 2026-09-17)

Structured record: [results/cuda-w8a8-engine-comparison.json](../results/cuda-w8a8-engine-comparison.json). To bound how far the Triton kernel is from this hardware's ceiling, two additional engines were written and benchmarked on the six shapes × M ∈ {64, 128, 512, 2048}: a hand-written `mma.sync.m16n8k32` e4m3 CUDA kernel with per-32-group FP32 scale-accumulation (the Triton kernel's numerics contract) autotuned over tile/split-K variants, and a cuBLASLt MXFP8 path (`VEC32_UE8M0` block scaling) exploiting the fact that the checkpoint's 32×32 UE8M0 weight-block scales expand losslessly to MX's 1×32 groups.

- **cuBLASLt MXFP8 won 19/24 cells** (up to 8.7× over tuned Triton at (25600,6144) M=2048, geometric-mean best-engine speedup 2.5×), the custom CUDA kernel 4/24 (small-M skinny-N cells, 1.5–2.7×), tuned Triton 1/24.
- **Correctness vs Triton:** 468 pairwise output comparisons on identical quantized inputs (including ragged M=100/777 tails) with **zero failures**; most cells bitwise-identical to the tuned Triton bf16 output, worst rel-L2 7.8e-6 — the same disagreement class as generic-vs-tuned Triton on the same inputs. With the benchmark's random (non-power-of-two) synthetic scales the MXFP8 path shows ~3.8e-2 UE8M0 requantization error; the deployed checkpoint declares `scale_fmt=ue8m0`, where the expansion is exact.
- **Not integrated into serving:** these are isolated-kernel measurements alongside an idle server. The MXFP8 path additionally needs one-time weight-scale re-blocking at load and a fused activation-quant epilogue before its GEMM gain translates end-to-end. Kernel sources and raw data live in the private workspace (`sm120-work/cuda-w8a8/`); this kit ships the structured results only.



## Historical SGLang context and capacity

| Observation | Interpretation |
| --- | --- |
| SGLang configured context limit: **1,048,576 tokens** | An admission/configuration limit; a full-window SGLang request was not tested in this record. |
| Earlier SGLang matrix context limit: **393,216 tokens** | The throughput matrix was collected before the limit was raised. |
| Longest verified SGLang request in this record: **117,286 tokens** | This is the observed validation extent, not a claim about every length below it. |
| Logical KV capacity: **6,871,808 tokens** | A logged pool-capacity observation, not maximum proven prompt length or a concurrency guarantee. |
| SWA capacity: **82,432 tokens** | A separate sliding-window pool observation; not additive full-context capacity. |
| Default static memory fraction: **0.80** | Leaves memory outside the static allocation; it does not guarantee arbitrary batch/graph/prefill shapes fit. |
| Default maximum running requests: **32** | Scheduling cap, not evidence that 32 maximum-context requests fit simultaneously. |

The logical and SWA pool figures cannot be multiplied, divided, or added into a guaranteed maximum workload without accounting for the model's cache structure, request mix, graph/workspace allocations, and prefill pressure. The longest verified request and the configured 1M limit must remain separate in any capacity planning.

Target and draft graphs are enabled by default. **Full-model prefill CUDA graphs remain disabled by model policy.** A successful direct attention-kernel graph capture is not evidence that whole-model prefill graph capture is supported.

## Historical SGLang quality and parser observations

- The small greedy question suite scored **24/25**, with the **same sisters-answer error** in the compared variants. This is a narrow consistency observation, not a general accuracy score or proof of bitwise equivalence.
- Long retrieval checks passed **14/14**. The longest verified SGLang request in this record was **117,286 tokens**; a full 1M-token SGLang request was not tested. This does not negate the separate earlier vLLM VM106 exercise.
- The earlier dense-FP8 serving A/B passed **110/110 synthetic retrieval requests in each variant**, plus **46 intermediate-concurrency checks** on the tuned variant. These belong to that earlier A/B, not a new validation run of this packaged image.
- **Eleven parser API scenarios passed** in the historical SGLang parser exercise. The SGLang recipe explicitly sets tool parser `deepseekv41` and reasoning parser `deepseek-v41`. Those checks were separate from the older throughput matrix; vLLM's distinct parser checks are documented above.
- Reasoning is not forced globally. Requests opt in with `reasoning_effort: "high"`; tool schemas are supplied by the client, and clients remain responsible for authorizing/executing tool calls.

These checks do not establish broad model quality, security of arbitrary tool execution, long-context accuracy at untested lengths, or exact equivalence between Marlin and FlashInfer experts.

## Dense block-FP8 contribution: a separate experiment

[SGLang PR #39872](https://github.com/sgl-project/sglang/pull/39872) is a narrower, newer main-based contribution. It reuses the portable Triton `SWAP_AB` / `SPLIT_K` kernel and FP32 reduction supplied by [#39657](https://github.com/sgl-project/sglang/pull/39657), adding SM120 dispatch, device-exact tuning tables, numerical coverage, and preview notes. It is **not** the submodule revision used by this kit and is **not** a complete SM120 model bring-up recipe.

### Earlier full-model tuning A/B

The original serving comparison used TP8/Marlin, hybrid native-decode/TileLang-prefill attention, no speculation, decode graphs, maximum 32 requests, and four continuous decode steps. Only dense-FP8 dispatch changed. Short technical prompts generated 1,536 output tokens/request; API-reported usage was used for unprofiled sustained aggregate throughput:

| Concurrency | Before dense-FP8 tuning | Tuned |
| ---: | ---: | ---: |
| 1 | 35.73 tok/s | 94.46 tok/s |
| 32 | 1,141.06 tok/s | 1,293.99 tok/s |

In a C1 trace over 40 decode steps, dense-FP8 kernel time including the new reduction fell from **826.92 ms to 96.43 ms**. Kernel timings can overlap across streams and cannot be summed into wall-clock time shares. Later native-prefill, NCCL P2P, MoE-backend, and DSpark improvements are separate experiments, not gains attributable solely to this FP8 contribution.

### Direct validation of the newer contribution tree

The PR reports **two manual SM120 tests passed**, covering **186 numerical comparisons** across the six real dense-layer shapes, tuned/intermediate/fallback row counts, output dtypes, and scale layouts. It also reports **six CUDA graphs with three replay states each** (original inputs, changed activations/scales, and zero activation scales). The complete rebased Python tree was used with the local runtime adapter disabled, on an isolated GPU without loading model weights or restarting the model server.

That evidence checks the kernel/configuration contribution; it is not a full-server launch or throughput measurement of current main. Strict FP32-oracle checks on the tuned path do not invent a new precision guarantee for the existing generic fallback's FP16 rounding. No SM90 rerun or full upstream CI success is claimed. Local/manual success must not be described as upstream CI passing; consult the linked PR for current authorization/check status.

## Native sparse-attention backport evidence

[FlashInfer PR #5121](https://github.com/flashinfer-ai/flashinfer/pull/5121) adds the missing 128/256-token extra-cache page instantiations needed by the hierarchical candidate-pool prefill path. This kit carries a minimal pinned backport for FlashInfer `0.6.18`, not an unbounded upgrade to FlashInfer main. The source hashes and module identity are recorded in `versions.json`.

Historical isolated native-backport validation passed **28 selected upstream cases**. An **all-masked-row `+inf` log-sum-exp** result also reproduced on stock FlashInfer; SGLang discards that value in the applicable path. This is a documented upstream numerical behavior, not an assertion that every kernel output is finite or a reason to hide errors. The selected-case result does not mean the full upstream FlashInfer suite passed, and direct kernel graph behavior does not establish full-model prefill-graph support.

The performance numbers in the upstream PR use a different hardware/workload protocol. They are not merged into this kit's eight-GPU matrix or presented as local measurements.

## Expert-kernel numerical comparison

An isolated expert-kernel comparison against an FP32 oracle measured relative L2 error of approximately **4.2% for FlashInfer** versus **0.58% for Marlin**. This is a numerical difference at the expert-kernel output, **not a 4.2% model-accuracy loss**, not a question-suite score, and not an end-to-end quality delta. Quantization, execution paths, and error accumulation require separate model-level evaluation.

The default EP8 FlashInfer backend reflects the measured serving trade-off, not a claim of numerical identity with TP-only Marlin. The documented Marlin rollback remains available for workload-specific evaluation.

## What a local deployment still needs to establish

For SGLang, use the shared command entry point:

```bash
./deploy.sh check-image
./deploy.sh status
./deploy.sh smoke
```

The first command is CPU-only SGLang image validation. The latter two require a separately started SGLang instance; `smoke` makes real model requests. For vLLM, use the explicit `scripts/smoke_api.py --engine vllm` command in [DEPLOYMENT.md](DEPLOYMENT.md), after establishing its image prerequisites and starting it separately. None of these checks is a fresh full matrix, broad quality evaluation, full-window test, or a guarantee of performance on another GPU topology. Preserve local acceptance results without committing private logs, prompts, host identities, or checkpoint paths.
