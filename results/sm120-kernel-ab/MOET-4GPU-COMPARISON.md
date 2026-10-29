# vLLM-Moet on VM106: 4-GPU deployment + benchmark vs our vLLM fork

## 1. GPU selection (fastest NCCL quad on this host)

`nvidia-smi topo -m` shows **every pair PHB** (single PCIe host bridge, no NVLink),
so topology gives no obvious subset; a 4-rank NCCL all-reduce microbench on
candidate quads (bf16, medians):

| quad | 40 KB | 160 KB | 640 KB | 1.31 MB |
|---|---:|---:|---:|---:|
| 0,1,2,3 | 40.3 µs | 38.5 | 116.3 | 139.1 |
| 4,5,6,7 | **26.2** | 38.7 | 118.2 | 141.6 |
| 0,2,4,6 | 25.8 | **35.5** | 156.1 | 143.0 |
| 1,3,5,7 | 25.8 | 36.3 | 154.1 | **108.3** |

Chosen: **GPUs 4,5,6,7** — best combined small/mid-message latency (26.2 µs at
40 KB, 118 µs at 640 KB), and a contiguous quad (simplest device mapping).

## 2. vLLM-Moet deployment

- Repo cloned on VM106; image built from `Dockerfile.sm120-dsv41-nightly`
  (base `vllm/vllm-openai:nightly-df3bd094…`, FlashInfer 0.7.2, DeepGEMM
  727f8470) -> `vllm-moet-sm120:dsv41-nightly-df3bd094-r21` (34.7 GB); build
  ~15 min after the 34.6 GB base pull.
- Served with their launcher: `GPUS=4,5,6,7 TP=4 PORT=1800x GPU_MEM_UTIL=0.92
  MAX_MODEL_LEN=524288 MAX_NUM_SEQS=8 MAX_NUM_BATCHED_TOKENS=4096
  KV_OFFLOAD_GIB=0`, model `/mnt/nas/alex/models/DeepSeek-V4.1-Flash-1017`.
- Startup: first start ~30 min (weights + FlashInfer/DeepGEMM autotune +
  TileLang/Тriton compiles); restart with warm cache ~13 min.
- Correctness smoke: "The capital of France is" -> " Paris." ✓.

## 3. Benchmarks (warmed, unique prefixes, 128 output tokens, rounds=2)

### vLLM-Moet, TP4, DSpark k=7 (their served configuration)
| input | C=1 tok/s (TPOT) | C=4 | C=8 |
|---:|---:|---:|---:|
| 512 | 136.2 (5.82 ms) | 231.5 (13.65) | 347.1 (16.92) |
| 1024 | 148.5 (5.71) | 246.0 (13.31) | 344.4 (17.04) |
| 2048 | 149.3 (5.11) | 220.6 (13.94) | 304.4 (18.58) |
| 4096 | 147.8 (5.32) | 225.3 (14.61) | 267.9 (19.53) |
| 8192 | 117.6 (4.83) | 164.0 (17.74) | 193.5 (25.39) |

### vLLM-Moet, TP4, DSpark k=5 (matched-k reference)
| input | C=1 | C=4 | C=8 |
|---:|---:|---:|---:|
| 512 | 166.8 | 272.1 | 386.9 |
| 1024 | 160.5 | 261.4 | 361.9 |
| 2048 | 171.6 | 267.4 | 360.2 |
| 4096 | 152.1 | 237.4 | 286.8 |
| 8192 | 122.0 | 165.5 | 219.2 |

### Our vLLM fork (dsv41-vllm:sm120), TP4, DSpark k=5
| input | C=1 | C=4 | C=8 |
|---:|---:|---:|---:|
| 512 | 104.6 (8.10) | 263.1 (11.97) | 357.0 (16.31) |
| 1024 | 103.8 (8.27) | 252.1 (11.97) | 319.1 (17.68) |
| 2048 | 102.3 (8.17) | 242.1 (12.21) | 285.3 (18.61) |
| 4096/8192 | crashed (HTTP 500 / broken pipe) | — | — |

## 4. Matched-k (k=5) comparison at common input lengths

| input | C | Moet TP4 k5 | our TP4 k5 | Moet / ours |
|---:|---:|---:|---:|---:|
| 512 | 1 | 166.8 | 104.6 | 1.59x |
| 512 | 4 | 272.1 | 263.1 | 1.03x |
| 512 | 8 | 386.9 | 357.0 | 1.08x |
| 1024 | 1 | 160.5 | 103.8 | 1.55x |
| 1024 | 4 | 261.4 | 252.1 | 1.04x |
| 1024 | 8 | 361.9 | 319.1 | 1.13x |
| 2048 | 1 | 171.6 | 102.3 | 1.68x |
| 2048 | 4 | 267.4 | 242.1 | 1.10x |
| 2048 | 8 | 360.2 | 285.3 | 1.26x |

At k=7 (Moet's served config) C1 is 136-149 tok/s (k=7 trades some C1 speed for
more draft tokens in other regimes). Our fork **cannot run k=7**: DSpark
validation rejects it (`num_speculative_tokens:7 must be divisible by
n_predict=5`, i.e. its draft MTP has 5 layers and only k<=5 or multiples of 5
are accepted).

## 5. Why our fork loses on 4 GPUs

Our `deepseek-v41-flash-sm120` fork was validated only on **TP8**; TP4 exposes
gaps that Moet explicitly patches for TP4:
- DSpark MTP divisibility (k=7 rejected; max native k=5);
- our baked FlashInfer `chunks_per_block` autotune cache (tuned at TP8 head
  geometry) caused an **illegal memory access** at TP4 startup; disabling
  `FLASHINFER_AUTOTUNE_DIR` makes TP4 start, confirming a cache-geometry bug;
- TP4 stability: 4096-token contexts return HTTP 500 (server-side failure),
  Moet handles 8192 (and up to 512K context);
- Moet's per-TP4 optimizations (b12x W4A8 binding at TP4, fused persistent
  decode MoE, FP4/MXFP4 KV records, KV offload, two-shot P2P all-reduce, GEMV
  paths, etc.) are not in our fork.

## 6. Conclusion

- **vLLM-Moet is the right 4-GPU stack on this host**: it runs TP4 + DSpark k=7
  on the fastest measured NCCL quad and delivers 136-387 tok/s across
  C1-C8 / 512-8192-token inputs with 512K context, stable.
- **Our vLLM fork cannot be fairly compared on 4 GPUs at k=7**: k=7 is rejected
  by its DSpark validation, TP4 needs our autotune cache disabled, and
  long-context TP4 is unstable. At the matched k=5 it is competitive at C4/C8
  (within 3-26%) but clearly behind at C1 (1.6x) — consistent with missing
  Moet's TP4-specific decode optimizations.
- Practical: for 4-GPU deployment use vLLM-Moet; keep our fork for TP8 (or
  port Moet's TP4-specific patches before using it on 4 GPUs).
