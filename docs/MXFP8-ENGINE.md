# cuBLASLt MXFP8 vs Triton on the DeepSeek-V4.1-Flash dense block-FP8 shapes

Measured on a dedicated 8× RTX PRO 6000 Blackwell box with **all GPUs idle**
(`research-8xpro6000`), so the numbers are not distorted by a resident model's
memory pressure (which skewed an earlier run on the serving host).

## Why MXFP8 is numerically exact here

The checkpoint declares `weight_block_size = [32, 32]` and `scale_fmt = ue8m0`
(power-of-two scales). A 32×32 UE8M0 block **expands losslessly into 1×32 (MX)
groups** — one scale per row of 32 weights — so cuBLASLt's native MXFP8 GEMM
(`CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0`, e4m3 payload / e8m0 groups) computes
the **same product** as the serving W8A8 block-FP8 kernel.

Measured rel-L2 vs an fp32 reference: **1.66e-3**, identical to the tuned Triton
kernel — i.e. the bf16-output floor. (An earlier 3.8e-2 figure was a benchmark
artifact: random fp32 test scales rounded to UE8M0. With the checkpoint's actual
power-of-two scales it vanishes.)

## Coverage

- **6 dense block-FP8 shapes** — `(1792,5120) (4096,1280) (5120,1024) (576,5120)
  (5120,288) (25600,6144)`.
- **20 M values** — 1,2,4,8,16,32,48,64,96,128,160,256,384,512,768,1024,1536,
  2048,3072,4096 → **120 cells**, no OOM.
- Engines: `triton_tuned` (kit split-K config), `triton_generic` (upstream
  fallback), `cuda_custom` (hand-written `mma.sync` m16n8k32, autotuned),
  `cublas_mxfp8` (cuBLASLt VEC32 UE8M0), `cublas_bf16` (dequantized upper bound).

## Result — MXFP8 wins at every M for every shape

| shape (N,K) | dispatch threshold M | worst ratio | best ratio |
|---|---:|---:|---:|
| 576×5120 | 1 | 1.17× | 5.13× |
| 1792×5120 | 1 | 1.16× | 5.55× |
| 4096×1280 | 1 | 2.40× | 3.73× |
| 5120×288 | 1 | 1.92× | 5.25× |
| 5120×1024 | 1 | 2.25× | 4.09× |
| 25600×6144 | 1 | 1.58× | 8.53× |

Ratios are `triton_tuned_us / cublas_mxfp8_us`. Small-M cells are medians of
3 rounds × 200 iterations (single-round runs showed launch jitter at M≤2);
large-M cells are from the 120-cell sweep.

Representative cells (µs):

| shape | M | triton_tuned | cuda_custom | MXFP8 |
|---|---:|---:|---:|---:|
| 25600×6144 | 1 | 314.6 | 293.8 | **199.3** |
| 25600×6144 | 256 | 1013.3 | 686.8 | **179.3** |
| 25600×6144 | 1024 | 4112.5 | 2012.7 | **482.3** |
| 5120×288 | 1 | 47.9 | 11.5 | **9.2** |
| 5120×1024 | 4096 | 243.6 | 252.1 | **68.0** |
| 4096×1280 | 2048 | 134.7 | 119.2 | **37.5** |

Where the custom CUDA kernel still leads: tiny-M skinny-N shapes where MXFP8's
fixed setup cost dominates (e.g. 576×5120 M≤512 at ~16 µs vs 41 µs), but MXFP8
wins 14/20 cells there too and takes over decisively from M=768.

## Correctness

`check_vs_triton_extended.py`: identical quantized inputs through tuned Triton
(reference), generic Triton (noise yardstick), 4 custom-CUDA variants × split-K
{1,4,8}, and cuBLASLt MXFP8 with po2 (checkpoint-native) scales — 6 shapes × 21 M
values including ragged tails.

**126 cells, 1,764 comparisons, 0 failures.** Most cells bitwise-identical to the
tuned Triton bf16 output; worst disagreement is the same order as
generic-vs-tuned Triton on identical inputs.

## Serving wiring

`runtime/mxfp8_gemm_sm120.py` dispatches per `(N, K, M)` using
`runtime/mxfp8_crossover.json` (all six shapes → threshold 1), keeping the tuned
Triton kernel as fallback for any shape/M not covered or on any failure. Opt in
with `DSV41_SM120_MXFP8=1` (requires `DSV41_SM120_FP8_DISABLE=0`).

Validated in-container: library builds, table parses, all six shapes reproduce
rel-L2 1.66e-3, weight-scale expansion lossless.

## End-to-end serving (aggregated TP8/EP8, VM 106, 8x RTX PRO 6000)

Same image and same environment; **only** `DSV41_SM120_MXFP8` /
`DSV41_SM120_MXFP8_MIN_M` differ. 32 requests per level, 256 output tokens,
temperature 0, greedy.

| conc | engine OFF | ON, prefill-only (`MIN_M=512`) | ON, all-M (`MIN_M=0`) |
|---|---:|---:|---:|
| 1 | 194.5 | **196.8** (+1.2%) | 192.9 |
| 4 | 400.9 | 396.2 (-1.2%) | 397.0 |
| 16 | 719.7 | 695.6 (-3.4%) | **518.9 (-28%)** |
| 32 | 1280.8 | **1332.2 (+4.0%)** | 1332.1 |
| prefill TTFT | 0.291 s | **0.268 s (-8%)** | **2.68 s (9x worse)** |

**Verdict: `MIN_M=512` (engage MXFP8 only for prefill-scale M) is neutral to
slightly positive; all-M regresses.** The per-call overhead of the MXFP8 path
(fresh activation/scale allocations plus a Python dispatch, neither captured in
the decode CUDA graph) costs more than the GEMM win at decode-scale M and while
decoding interleaves with prefill. Greedy correctness probes (`17*23=391`,
`144/12=12`) pass in every arm.

### The bug this exposed

The first A/B showed C1 collapsing 194 -> 123 tok/s with the engine on. Root
cause was **not** MXFP8's speed: the adapter snapshotted its fallback kernel at
install time. When MXFP8 wrapped *before* the tuned SM120 patch, that snapshot
was the **raw Triton kernel**, so every declined call (i.e. all of decode)
bypassed the tuned config table. Fixed by resolving the fallback at call time
(`_fallback()` reads the current entry point and steps inward via
`__sm120_inner__`), which makes installation order irrelevant. After the fix C1
returned to 196.8.

Lesson: when stacking kernel wrappers, never capture the inner callable at
install time.

### Not measured

- No sampled/quality evaluation beyond two greedy arithmetic probes.
- No long-run (hours) stability or memory-growth test.
- Prefill TTFT is sensitive to prefix-cache warmth and to restart-to-restart
  variance in speculative acceptance (0.49-0.71 observed on this model), so the
  prefill row is indicative rather than precise. Throughput rows are stable
  (C1 repeats within +/-0.3%).

## Reproduce

```bash
# dedicated GPU box, all GPUs free
docker run -d --name bench --gpus all --ipc=host --shm-size 32g \
  -v $PWD:/work -w /work -e PYTHONPATH=/sgl-workspace/sglang/python \
  lmsysorg/sglang:dev-dsv41 sleep infinity
docker exec bench bash -lc "cd /work && \
  PYTHONPATH=/opt/dsv41/runtime:/sgl-workspace/sglang/python \
  python3 bench_extended.py --samples 5 --iters 50 --jsonl extended.jsonl"
python3 consolidate.py   # -> mx-engine-results.json, mxfp8_crossover.json
```
## Deployment

```bash
# build the engine-enabled image (adds the PR5121 sparse-MLA backport the model needs)
docker build -f Dockerfile.sglang --target mxfp8 -t deepseek-v41-flash-sm120:mxfp8 .

# standalone test launcher (never touches the production deploy path)
./serve-mxfp8-test.sh check
./serve-mxfp8-test.sh start on     # MXFP8 on, MIN_M default 512
./serve-mxfp8-test.sh start off    # same stack, engine disabled
./serve-mxfp8-test.sh bench
```

Env knobs: `DSV41_SM120_MXFP8=1` enables the engine, `DSV41_SM120_MXFP8_MIN_M`
sets the minimum M at which it engages (default 512 = prefill only; `0` = always,
which measured worse), `DSV41_SM120_FP8_DISABLE` must be `0`.

Note: the `Using default W8A8 Block FP8 kernel config. Performance might be
sub-optimal!` startup line is emitted by **stock** SGLang, which ships no RTX PRO
6000 SM120 config files (0 of them). The tuned table is supplied by this kit's
runtime adapter, so the message appears even when tuned configs are in effect.
Shapes absent from the table (e.g. `1536x5120`, `5120x15360`) genuinely run the
generic kernel — a known, small gap.
