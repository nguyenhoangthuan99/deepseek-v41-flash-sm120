# Historical results and validation boundaries

The structured record is [results/measurements.json](../results/measurements.json). These are **historical measurements of the pinned serving integration**, not a new benchmark of an image rebuilt from this repository. The runtime configuration is described in [DEPLOYMENT.md](DEPLOYMENT.md).

Three kinds of evidence must remain distinct:

1. **Historical full-model serving:** performance, retrieval, question-suite, and parser observations from the preview integration.
2. **Isolated kernel validation:** direct numerical and CUDA-graph tests, sometimes against a newer contribution tree, without loading the full model.
3. **Packaged-image verification:** local image construction and CPU package/source/parser/manifest checks. Running `deploy.sh check-image` does not run GPU kernels, serve weights, or reproduce either of the other categories. A reader's `deploy.sh smoke` is a fresh API check, not a replacement for the historical benchmark protocol.

Nothing here claims a fresh rebuilt-image throughput/quality matrix, a validated full 1M-token request, complete stock-main support, or full upstream CI success for the contribution PR.

The [source-parity record](../results/source-parity.json) separately confirms that **4,770 tracked files in SGLang's `python/` tree** from the pinned preview commit matched the original deployed native image: zero missing files and zero SHA-256 mismatches. It verifies source identity, not the behavior or image digest of a subsequent rebuild.

The [package-validation record](../results/package-validation.json) records successful builds of both `runtime` and `hybrid`, CPU-only checks of all 4,770 source files and eight runtime extensions in each image, package/parser/backport integrity, and exercised Bash safety controls. The new API smoke client passed eleven cases against the existing validated server, not a newly launched packaged image. No production restart or new full-model throughput claim was made during packaging.

## Hardware and historical settings

The measured host used eight NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs (SM120, 188 SMs/GPU, approximately 96 GiB/GPU). The pinned SGLang preview source was `da64c5cbb8cf6bfd39be19da43573fdfd484c43a`; the historical software included PyTorch `2.13.0+cu130`, Triton `3.7.1`, FlashInfer `0.6.18`, and TileLang `0.1.14`. Source/package/backport identities and historical image digests are in [versions.json](../versions.json).

The final native-serving matrix used TP8/EP8, FlashInfer MXFP4 experts, native sparse attention with the minimal FlashInfer PR #5121 dispatch backport, dense block-FP8 tuning, `NCCL_P2P_LEVEL=PHB`, four continuous decode steps, maximum 32 running requests, and static speculation where indicated. DSpark used `SGLANG_SIMULATE_ACC_LEN=-1` and SPS recording disabled. Target/draft decode graphs were enabled; full-model prefill graphs were not enabled.

**The throughput matrix predates both the configured context increase from 393,216 to 1,048,576 and the explicit tool/reasoning parser flags.** Parser checks were a separate later API exercise. Do not relabel this matrix as a benchmark of the exact current launch arguments.

## Warm no-speculation versus DSpark3

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

## Context and capacity

| Observation | Interpretation |
| --- | --- |
| Current configured context limit: **1,048,576 tokens** | An admission/configuration limit; a full-window request was not tested. |
| Earlier matrix context limit: **393,216 tokens** | The throughput matrix was collected before the limit was raised. |
| Longest verified request: **117,286 tokens** | This is the observed validation extent, not a claim about every length below it. |
| Logical KV capacity: **6,871,808 tokens** | A logged pool-capacity observation, not maximum proven prompt length or a concurrency guarantee. |
| SWA capacity: **82,432 tokens** | A separate sliding-window pool observation; not additive full-context capacity. |
| Default static memory fraction: **0.80** | Leaves memory outside the static allocation; it does not guarantee arbitrary batch/graph/prefill shapes fit. |
| Default maximum running requests: **32** | Scheduling cap, not evidence that 32 maximum-context requests fit simultaneously. |

The logical and SWA pool figures cannot be multiplied, divided, or added into a guaranteed maximum workload without accounting for the model's cache structure, request mix, graph/workspace allocations, and prefill pressure. The longest verified request and the configured 1M limit must remain separate in any capacity planning.

Target and draft graphs are enabled by default. **Full-model prefill CUDA graphs remain disabled by model policy.** A successful direct attention-kernel graph capture is not evidence that whole-model prefill graph capture is supported.

## Quality and parser observations

- The small greedy question suite scored **24/25**, with the **same sisters-answer error** in the compared variants. This is a narrow consistency observation, not a general accuracy score or proof of bitwise equivalence.
- Long retrieval checks passed **14/14**. The longest verified request was **117,286 tokens**; a full 1M-token request remains untested.
- The earlier dense-FP8 serving A/B passed **110/110 synthetic retrieval requests in each variant**, plus **46 intermediate-concurrency checks** on the tuned variant. These belong to that earlier A/B, not a new validation run of this packaged image.
- **Eleven parser API scenarios passed** in the historical parser exercise. The current recipe explicitly sets tool parser `deepseekv41` and reasoning parser `deepseek-v41`. Those checks were separate from the older throughput matrix.
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

Use only the shared command entry point for operational checks:

```bash
./deploy.sh check-image
./deploy.sh status
./deploy.sh smoke
```

The first command is CPU-only image validation. The latter two require a separately started instance; `smoke` makes real model requests. None is a fresh full matrix, broad quality evaluation, full-window test, or a guarantee of acceptable performance on a different GPU topology. Preserve your own local acceptance results without committing private logs, prompts, host identities, or checkpoint paths.
