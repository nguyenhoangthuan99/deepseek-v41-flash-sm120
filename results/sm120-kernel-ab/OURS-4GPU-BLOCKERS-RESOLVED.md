# Our solution on VM106 4 GPUs: blockers, resolution and full test

## Blockers found earlier and their status

1. **DSpark k=7 rejected by our fork** — `num_speculative_tokens:7 must be
   divisible by n_predict=5` (our DSpark draft has 5 MTP layers; only k<=5 or
   multiples of 5 are accepted). **Not resolvable as a config change**: k=7
   requires porting Moet's DSpark extension (their fork supports k>5). Tested our
   stack at **k=5**, its maximum supported value.
2. **TP4 startup illegal memory access** — caused by our baked FlashInfer
   AutoTuner cache (tactics tuned for TP8 head geometry being applied at TP4).
   **Resolved**: run TP4 with `FLASHINFER_AUTOTUNE_DIR` pointing to an empty dir
   (heuristic tactics); the same cache must be regenerated per TP geometry before
   it can be used at TP4.
3. **4096-context HTTP 500 seen in the first TP4 attempt** — with the cache
   disabled and the validated TP4 config it is **stable**: full matrix
   512/1024/2048/4096/8192 x C1/4/8 and long prompts to 24,576 tokens all
   completed (the earlier 500 was a transients/state issue, not reproducible now).

## Working TP4 configuration for our fork
```
image: dsv41-vllm:sm120 (baked build)
GPUs 4-7, --tensor-parallel-size 4
--max-model-len 32768 --block-size 64 --gpu-memory-utilization 0.95
--max-num-seqs 8 --max-num-batched-tokens 4096
--speculative-config '{"method":"dspark","num_speculative_tokens":5,...}'
--compilation-config cudagraph_mode=FULL_DECODE_ONLY, capture sizes [8..64]
-e FLASHINFER_AUTOTUNE_DIR=/empty   (critical at TP4)
```

## Results (warmed, unique prefixes; tok/s)

### Short prompts, 128 output
| input | C | our TP4 k5 | Moet TP4 k5 | Moet TP4 k7 | VM100 TP8 |
|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 100.2 | 166.8 | 136.2 | 160.5 |
| 512 | 4 | **295.0** | 272.1 | 231.5 | 354.4 |
| 512 | 8 | 375.0 | 386.9 | 347.1 | 503.3 |
| 1024 | 1 | 100.3 | 160.5 | 148.5 | 155.1 |
| 1024 | 4 | 251.7 | 261.4 | 246.0 | 332.4 |
| 1024 | 8 | 352.7 | 361.9 | 344.4 | 463.2 |
| 2048 | 1 | 105.7 | 171.6 | 149.3 | 147.9 |
| 2048 | 4 | 242.5 | 267.4 | 220.6 | 285.8 |
| 2048 | 8 | 278.8 | 360.2 | 304.4 | 358.7 |
| 4096 | 1 | 100.3 | 152.1 | 147.8 | 124.5 |
| 4096 | 4 | 188.8 | 237.4 | 225.3 | 215.8 |
| 4096 | 8 | 212.9 | 286.8 | 267.9 | 258.3 |
| 8192 | 1 | 74.3 | 122.0 | 117.6 | 98.3 |
| 8192 | 4 | 131.5 | 165.5 | 164.0 | 145.4 |
| 8192 | 8 | 145.6 | 219.2 | 193.5 | 164.5 |

### Long prompts, 256 output (our mll is 32768, so 24,576 max tested)
| input | C | our TP4 k5 | Moet TP4 k7 | VM100 TP8 |
|---:|---:|---:|---:|---:|
| 16384 | 1 | 79.5 (TTFT 1245 ms) | 128.3 (745) | 103.9 (1222) |
| 16384 | 4 | 129.9 (3294) | 188.7 (1903) | 147.6 (3673) |
| 24576 | 1 | 64.6 (1822) | ~? (32768: 98.7) | ~? (32768: 69.4) |
| 24576 | 4 | 109.4 (4800) | ~? (32768: 124.5) | ~? (32768: 89.9) |

### Long generation (1024 output)
| input | C | our TP4 k5 | Moet TP4 k7 | VM100 TP8 |
|---:|---:|---:|---:|---:|
| 2048 | 1 | 169.7 (TPOT 5.69) | 345.3 (2.78) | 221.9 (4.32) |
| 2048 | 4 | 402.9 (7.37) | 413.8 (8.59) | 387.6 (9.15) |
| 8192 | 1 | 154.6 (5.83) | 296.4 (2.89) | 222.2 (3.90) |
| 8192 | 4 | 347.8 (7.76) | 415.6 (8.42) | 437.3 (7.01) |

## Conclusions
- **Our solution now runs on 4 GPUs** (TP4, k=5) and is stable across
  512-24,576-token prompts, C1-C8 and 1024-token generations.
- Absolute comparison at equal k=5: Moet is much faster at C1 (1.5-1.7x) and
  generally ahead beyond 2048 tokens; our stack is competitive/better at
  C4 512-1024 (295 vs 272) and roughly equal at C8 512.
- k=7 (Moet's served setting) remains out of reach without porting Moet's
  DSpark k>5 extension; until then Moet remains the better 4-GPU deployment.
  For a like-for-like k=5 test our stack is measurable and stable, which was
  the blocker.
