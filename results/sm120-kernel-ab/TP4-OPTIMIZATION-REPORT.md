# TP4 optimization pass (our vLLM fork, GPUs 4-7, k=5)

Baseline TP4 (ours, k=5, TP8-tuned graph buckets [8,16,...], FI autotune cache
disabled): 100.2/295.0/375.0 tok/s at 512 (C1/C4/C8), 74.3 at 8192/C1.

## Changes tested

1. **Graph-capture buckets aligned to the DSpark k=5 decode step size.**
   With k=5 each sequence contributes 6 tokens/step, so capture sizes were set to
   multiples of 6: `[6,12,18,24,30,36,42,48,54,60,64]` instead of
   `[8,16,24,32,48,64]`. This removes the padding waste in the captured decode
   graphs (one of the largest single-kernel effects at TP4).
2. **NCCL_PROTO=LL128** on top of (1).
3. **TP4-geometry FlashInfer AutoTuner cache** (new artifact): generated for the
   production mixed sparse-MLA ABI at TP4 head count (H=16) and the actual
   k=5 decode token buckets T=6..64; mounted via `FLASHINFER_AUTOTUNE_DIR`.
   (The TP8 cache is invalid at TP4 and caused IMA; this one matches TP4.)

## Results (short prompts, 128 output; tok/s C1/C4/C8)

| input | baseline | +graph buckets | +LL128 | +TP4 autotune cache |
|---:|---:|---:|---:|---:|
| 512 | 100.2 / 295.0 / 375.0 | **129.9 / 318.3 / 369.9** | 119.4 / 268.4 / 390.2 | 137.0 / 254.2 / 400.9 |
| 1024 | 100.3 / 251.7 / 352.7 | **123.6 / 269.6 / 348.3** | 119.1 / 269.9 / 365.6 | 134.1 / 245.8 / 342.0 |
| 2048 | 105.7 / 242.5 / 278.8 | **127.1 / 238.2 / 307.5** | 119.8 / 217.2 / 300.1 | 131.1 / 253.3 / 294.4 |
| 4096 | 100.3 / 188.8 / 212.9 | **118.2 / 202.3 / 225.0** | 108.4 / 177.6 / 222.9 | 113.3 / 179.1 / 221.0 |
| 8192 | 74.3 / 131.5 / 145.6 | **90.7 / 130.6 / 150.3** | 79.9 / 127.6 / 147.1 | 86.2 / 131.0 / 150.7 |

Deltas of the graph-bucket change (the clear winner): **C1 +18-30%**, C4
512/1024 +7-8%, C8 ~neutral, 8192/C1 +22%. `NCCL_PROTO=LL128` is worse overall
(only 512/1024 C8 slightly better). The TP4 autotune cache helps C1 by 3-8% but
hurts C4 in places (tactic choices for those buckets), so it is not a clear win;
the graph-bucket change alone is the recommended default.

## Best TP4 config for our fork
```
--tensor-parallel-size 4  --max-model-len 32768 --gpu-memory-utilization 0.95
--max-num-seqs 8 --max-num-batched-tokens 4096
DSpark k=5
cudagraph_capture_sizes=[6,12,18,24,30,36,42,48,54,60,64]
FLASHINFER_AUTOTUNE_DIR=/empty   # or the TP4-geometry cache for C1-heavy work
```

## vs Moet TP4 (k=5, same GPUs), 512 input
| C | ours (optimized) | Moet k5 | gap |
|---:|---:|---:|---:|
| 1 | 129.9 | 166.8 | 1.28x (was 1.67x) |
| 4 | **318.3** | 272.1 | ours +17% |
| 8 | 369.9 | 386.9 | ~4% |

At 8192/C1 ours 90.7 vs Moet 122.0 (1.35x gap). The remaining TP4 gap is in
C1/long-context decode, where Moet uses TP4-specific kernels our fork lacks:
dense MXFP8 GEMV for M<=16 (their 1.8-3.4x on decode GEMMs), b12x W4A8 compact
MoE bound at TP4 (our fork's b12x MoE backend is unavailable at this
architecture — attempting it exits at startup), the fused persistent decode MoE,
the sparse-MLA candidate-row load changes, and the two-shot P2P all-reduce.
Those are the next real kernel additions for TP4.
