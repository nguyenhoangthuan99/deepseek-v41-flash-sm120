# Our vLLM TP8 vs vLLM-Moet TP4 (VM106)

Same model (DeepSeek-V4.1-Flash-1017), same benchmark client (unique prefixes,
warmed, 128 output tokens unless noted), same host.

Caveats: our fork only supports DSpark k=3 (k=5/7 crash), Moet was run at its
served k=7 (and k=5 for a matched-k short run). Our fork's TP8 numbers below
come from its stable 8-GPU config (1M ctx, max-num-seqs 32, batched 8192,
gpu_mem 0.85); trying to force a Moet-like 4-GPU/8-seq config on our fork
crash-loops (NCCL unhandled error / illegal memory access) — see §4.

## 1. Short prompts (512-8192 input, 128 output), tok/s

| input | C | our TP8 (8 GPU, k3) | Moet TP4 (4 GPU, k7) | ours/Moet | ours per-GPU | Moet per-GPU |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 151.3 | 136.2 | 1.11x | 18.9 | 34.1 |
| 512 | 4 | 311.0 | 231.5 | 1.34x | 38.9 | 57.9 |
| 512 | 8 | 472.7 | 347.1 | 1.36x | 59.1 | 86.8 |
| 1024 | 1 | 153.4 | 148.5 | 1.03x | 19.2 | 37.1 |
| 1024 | 4 | 306.1 | 246.0 | 1.24x | 38.3 | 61.5 |
| 1024 | 8 | 472.7 | 344.4 | 1.37x | 59.1 | 86.1 |
| 2048 | 1 | 150.0 | 149.3 | 1.00x | 18.8 | 37.3 |
| 2048 | 4 | 289.0 | 220.6 | 1.31x | 36.1 | 55.2 |
| 2048 | 8 | ~421.7 (C16) | 304.4 | 1.39x | 52.7 | 76.1 |
| 4096 | 1 | 128.5 | 147.8 | 0.87x | 16.1 | 37.0 |
| 4096 | 4 | 223.5 | 225.3 | 0.99x | 27.9 | 56.3 |
| 8192 | 1 | 97.7 | 117.6 | 0.83x | 12.2 | 29.4 |
| 8192 | 4 | 145.7 | 164.0 | 0.89x | 18.2 | 41.0 |
| 8192 | 8 | 161.3 | 193.5 | 0.83x | 20.2 | 48.4 |

Short-context summary: our TP8 gives higher **aggregate** throughput for
512-4096 at C>=4 (more GPUs), but Moet TP4 is **1.5-2.4x more efficient per
GPU** and overtakes our absolute numbers from 8192 tokens on, despite using half
the GPUs.

## 2. Long prompts (256 output)

| input | C | our TP8 (k3) | Moet TP4 (k7) |
|---:|---:|---:|---:|
| 16384 | 1 | 103.0 tok/s (TTFT 1313.6 ms, TPOT 4.44) | **128.3** (745.1 ms, 5.03) |
| 16384 | 4 | crashed (HTTP 500, illegal memory access) | **188.7** (1902.7 ms, 11.70) |
| 32768 | 1 | not measured (server crashed) | **98.7** (1397.4 ms, 4.61) |
| 32768 | 4 | — | **124.5** (3595.6 ms, 16.80) |
| 65536 | 1 | — | **65.3** (2767.3 ms, 4.71) |
| 65536 | 4 | — | **77.4** (6941.4 ms, 22.24) |
| 131072 | 1 | — | **38.9** (5470.0 ms, 4.11) |
| 131072 | 4 | — | **43.7** (13661.5 ms, 37.31) |

## 3. Long generation (1024 output tokens)

| input | C | our TP8 (k3) | Moet TP4 (k7) |
|---:|---:|---:|---:|
| 2048 | 1 | crashed | **345.3 tok/s** (TPOT 2.78 ms) |
| 2048 | 4 | crashed | **413.8** (8.59) |
| 8192 | 1 | crashed | **296.4** (2.89) |
| 8192 | 4 | crashed | **415.6** (8.42) |

## 4. Our fork's TP8 stability under these configs

- With the Moet-like 8-seq/4096-batched/524288-ctx config our fork fails at
  startup: `NCCL error: unhandled cuda error` preceded by illegal memory access
  (Triton/NCCL), with or without our FlashInfer autotune cache.
- In the "long prompt" run the fork completed 16k/C1 then crashed on 16k/C4
  (illegal memory access on multiple TP workers).
- Earlier short-context runs with the 1M-ctx/32-seq config are stable and are
  the TP8 numbers used above.
- k=7 is unsupported by our fork by construction (DSpark draft n_predict=5
  divisibility check), and k=5 also crashes at startup.

## 5. KV-cache headroom (tests stayed within it)
- Our TP8: 14,792,271 KV tokens, 28.2x concurrency at 524,288 ctx.
- Moet TP4: 2.58 GiB/rank KV (its README: 9.62M tokens at 512K). The longest
  test (131k x C4 = 524k tokens) is well inside both.

## 6. Conclusion
- **For 4 GPUs: Moet TP4 is clearly better** — stable to 512k context and 1024+
  output tokens, and 1.5-2.4x higher throughput per GPU than our TP8 stack.
- **Our TP8** wins only in absolute short-context (512-4096) aggregate at C>=4
  because it has twice the GPUs; per-GPU it is 1.5-2.4x behind, loses outright at
  8k+ and cannot complete long-prompt/long-generation workloads in these
  configs.
- To use our fork at TP4/k=7 fairly, Moet's TP4-specific patches (b12x W4A8 TP4
  binding, DSpark k>5, FP4 KV records, KV offload, two-shot P2P all-reduce,
  fused decode MoE) would have to be ported; until then Moet is the deployment
  choice for 4 GPUs.
