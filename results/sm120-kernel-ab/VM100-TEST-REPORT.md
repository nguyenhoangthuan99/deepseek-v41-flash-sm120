# VM100 deployment test (10.200.108.100:30000) + comparison

Benchmarked the already-running VM100 dsv41-vllm deployment from VM106 with the
same client/protocol used for Moet/our TP8 (unique prefixes, warmed, medians).
VM100: TP8, 1M ctx, DSpark K3, vLLM fork (max_model_len 1,048,576).

## 1. Short prompts (128 output tokens), tok/s

| input | C | VM100 TP8 | Moet TP4 k7 (4 GPU) | our TP8 k3 |
|---:|---:|---:|---:|---:|
| 512 | 1 | 160.5 | 136.2 | 151.3 |
| 512 | 4 | 354.4 | 231.5 | 311.0 |
| 512 | 8 | 503.3 | 347.1 | 472.7 |
| 1024 | 1 | 155.1 | 148.5 | 153.4 |
| 1024 | 4 | 332.4 | 246.0 | 306.1 |
| 2048 | 1 | 147.9 | 149.3 | 150.0 |
| 2048 | 4 | 285.8 | 220.6 | 289.0 |
| 4096 | 1 | 124.5 | 147.8 | 128.5 |
| 4096 | 4 | 215.8 | 225.3 | 223.5 |
| 8192 | 1 | 98.3 | 117.6 | 97.7 |
| 8192 | 4 | 145.4 | 164.0 | 145.7 |
| 8192 | 8 | 164.5 | 193.5 | 161.3 |

VM100 TP8 is the fastest short-context aggregate at C>=4 (2x the GPUs), but
Moet TP4 is 1.4-2.2x better per GPU and already wins in absolute terms at
4k-8k/C1.

## 2. Long prompts (256 output tokens)

| input | C | VM100 TP8 | Moet TP4 k7 |
|---:|---:|---:|---:|
| 16384 | 1 | 103.9 tok/s (TTFT 1222 ms, TPOT 4.75) | **128.3** (745 ms, 5.03) |
| 16384 | 4 | 147.6 (3672 ms, 11.78) | **188.7** (1903 ms, 11.70) |
| 32768 | 1 | 69.4 (2385) | **98.7** (1397) |
| 32768 | 4 | 89.9 (6786) | **124.5** (3596) |
| 65536 | 1 | 42.5 (4753) | **65.3** (2767) |
| 65536 | 4 | 49.4 (12462) | **77.4** (6941) |
| 131072 | 1 | 23.2 (9863) | **38.9** (5470) |
| 131072 | 4 | 24.9 (24494) | **43.7** (13662) |

Moet TP4 beats VM100 TP8 on every long-prompt cell despite half the GPUs:
TTFT 1.4-1.8x lower, output tok/s 1.4-1.8x higher.

## 3. Long generation (1024 output tokens)

| input | C | VM100 TP8 | Moet TP4 k7 |
|---:|---:|---:|---:|
| 2048 | 1 | 221.9 (TPOT 4.32 ms) | **345.3** (2.78) |
| 2048 | 4 | 387.6 (9.15) | **413.8** (8.59) |
| 8192 | 1 | 222.2 (3.90) | **296.4** (2.89) |
| 8192 | 4 | **437.3** (7.01) | 415.6 (8.42) |

Moet wins 3 of 4 long-generation cells (notably C1: 1.33-1.56x), VM100 slightly
ahead at 8192/C4.

## 4. Capacity check
- VM100: 1M ctx, KV reported 21.63 GiB at deployment (TP8). The largest test
  (131072 input x C4 = 524k tokens + 256 output) stayed inside; no KV errors.
- Moet TP4: 2.58 GiB/rank KV (9.62M tokens at 512K) — same test well inside.

## 5. Conclusions
- VM100 deployment is healthy and reproducible: numbers match the earlier
  baseline within noise (512/C1 160.5 vs 155-160 tok/s).
- **Moet TP4 is the strongest option per GPU**: it beats VM100 TP8 on every
  long-prompt cell and most long-generation cells with half the GPUs, and is
  only behind on short-context high-concurrency aggregate where 8 GPUs win.
- If VM100 keeps TP8: it remains best for short-context C>=4 aggregate.
  For long-context/long-generation serving and any 4-GPU deployment, Moet is
  better.
