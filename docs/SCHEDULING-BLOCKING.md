# Staggered-arrival blocking, and why 4+4 disaggregation does not fit

## Architecture (measured)

DeepSeek-V4.1-Flash is a 552B **asymmetric Causal Encoder-Decoder** MoE: the
encoder handles prefill (~8B active), the decoder handles decode (~16B active).
There is **no separable encoder/decoder weight set** — the asymmetry is a
compute-path difference over a **shared** stack. Weight breakdown from the
checkpoint (`results/weight-breakdown.json`):

| group | size | note |
|---|---:|---|
| `layers.*` MoE decoder | 295.6 GB | shared stack (384 experts, 6 active) |
| `engram` memory | 203.1 GB | n-gram/retrieval tables (layers 1,14; vocab 16M) |
| `mtp`/dspark | 7.9 GB | multi-token-prediction / speculative |
| embed/head | 2.6 GB | |
| **vision encoder (ViT)** | **1.0 GB** | the actual encoder is tiny |
| **total** | **510.3 GB** | |

## Why staggered requests block

Encoder-prefill and decoder-decode are different forward passes over different
active weights; the engine serializes them. Requests arriving **together**
prefill together then decode together. Requests arriving **staggered** force each
newcomer's prefill to pause the running decode batch — head-of-line blocking.
Contributing scheduler state: speculative decoding on (blocks mixed-chunk overlap,
lengthens each decode step), `num-continuous-decode-steps=4` (delays admitting new
prefills), `fcfs` + `schedule-conservativeness=1.0`, and KV pressure at 1M context
(retraction/preemption).

## Why 4+4 PD disaggregation does not fit here

In SGLang PD disaggregation **both** the prefill pool and the decode pool load the
**full** model; the split is by phase, KV is transferred prefill->decode. So a
4-GPU pool must hold all 510 GB:

```
$ ./serve-disagg.sh plan --prefill 4 --decode 4
  topology                               GPUs/pool   wt/GPU   KV/GPU  fits
  aggregated TP8 (current, 1 pool)               8    63.8G    24.8G  yes
  disagg 4+4 (each pool = 4 GPUs)                4   127.6G   -39.0G  NO
  disagg 8+8 two-node (each pool = 8)            8    63.8G    24.8G  yes
```

510.3 / 4 = **127.6 GB/GPU** > 95.6 GB VRAM. It only fits if the 203 GB Engram
tables move to host RAM (`--engram-cpu` -> 76.8 GB/GPU, ~12 GB KV left — tight,
adds per-token latency at layers 1,14, and is unverified in SGLang). Even then,
the KV transport (mooncake/nixl) needs **InfiniBand/RDMA**, absent on a single
PCIe-PHB node. Viable disaggregation = **8+8 across two nodes with IB**.

## What to do

- **Single 8-GPU node (the fix that runs here):**
  `./serve-disagg.sh aggregated restart`
  Relaunches the validated best config with anti-blocking knobs: DSPARK off,
  `--enable-mixed-chunk`, `--num-continuous-decode-steps 1`,
  `--chunked-prefill-size 1024`, `--schedule-conservativeness 0.8`. Trades some
  single-stream decode throughput for far less staggered-arrival blocking; A/B it.
- **Two nodes + IB (real disaggregation):**
  `PREFILL_GPUS=8 DECODE_GPUS=8 DISAGG_IB_DEVICE=mlx5_0 ./serve-disagg.sh disagg --emit`
  then launch `prefill` / `decode` / `lb` roles. Both pools **inherit the full
  validated stack** (flashinfer_mxfp4 MoE, fp4 indexer, SM120 block-FP8, PHB, 1M
  context, parsers); speculation (DSPARK) stays on the decode pool only.
