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

**Not yet measured:** end-to-end serving effect. The GEMM timings above exclude
the per-call activation quantization (e4m3 + UE8M0 blocked scales) that
production must fuse; and no full-model throughput comparison from these engines
has been run.

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