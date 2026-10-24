# Route test: PCIe-optimised all-to-all instead of DeepEP

Goal: check whether replacing vLLM's DP-attention allgather+reducescatter
(34% of kernel time) with a PCIe-safe all-to-all (NCCL `all_to_all_single` /
ring / one-shot) is worth implementing on this 8x RTX PRO 6000 PCIe node.

## 1. What exists in the stack already

- vLLM all2all backends: `naive` **removed** (server logs
  "The 'naive' all2all backend has been removed. Falling back to
  'allgather_reducescatter'"), so it silently runs AG+RS; DP8 benchmark with
  `--all2all-backend naive` reproduced the AG+RS numbers exactly
  (C1 46.5, C4 158.5, C8 293.6, C16 544.0, C32 985.5 tok/s vs baseline
  46.4/157.6/287.9/547.2/1002.4).
- `MoEPrepareAndFinalizeNaiveDPEP*` uses Torch **AR/RS** ("Uses Torch AR/RS or
  AR for dispatch/combine"), not `all_to_all_single`.
- DeepEP (the intended fast path) cannot run on this PCIe-only node
  (`CPU recv timeout` due to missing NVLink ordering; see
  DEEPEP-SM120-PORT-REPORT.md).

So there is no existing PCIe-safe a2a EP backend to switch on; testing the
route means measuring the collective itself.

## 2. Collective microbenchmarks (8 ranks, NCCL, PCIe)

Model shape: hidden 5120 bf16, topk 6, 384 experts (48/rank). Per-rank token
counts M=1..16 correspond to DP decode batch sizes (DP splits requests across 8
ranks, so M ≈ concurrency/8 × spec tokens).

### One-way dispatch (`all_to_all_single` variable splits vs `all_gather`)
| M/rank | a2a dispatch | all-gather | ratio |
|---:|---:|---:|---:|
| 1 | 65.8 µs | 121.7 µs | 0.54 |
| 4 | 26.9 | 93.0 | 0.29 |
| 8 | 43.3 | 181.9 | 0.24 |
| 16 | 59.5 | 323.2 | 0.18 |

### Full two-way layer comm (dispatch + combine) vs AG + reduce-scatter
| M/rank | a2a dispatch+combine | allgather+reduce-scatter | ratio |
|---:|---:|---:|---:|
| 1 | 62.5 µs | 68.1 µs | 0.92 |
| 2 | 173.6 | 278.7 | 0.62 |
| 4 | 125.0 | 110.7 | 1.13 |
| 8 | 75.4 | 370.2 | 0.20 |
| 16 | 286.6 | 452.4 | 0.63 |

Observations:
- A single NCCL all-to-all dispatch is clearly cheaper than an all-gather of all
  tokens (2-5x at M>=4), because all-gather replicates every token to all 8
  ranks while a2a only sends each token where its experts live.
- But the **two-way dispatch+combine** picture is noisy and mostly
  break-even at decode-sized messages: NCCL's small/uneven all-to-all has a
  high fixed latency (~60-170 µs floor), so the theoretical traffic saving does
  not translate into a reliable win at M=1-4. At M=8-16 it can be 1.6-5x better,
  i.e. only for saturated decode.
- This is a collective-level test; a real kernel (ring/one-shot with
  PCIe-safe ordering) would have to beat NCCL's a2a first, which NCCL already
  implements on PCIe, then be integrated through vLLM's modular MoE
  prepare/finalize API (routing permutation, scales, expert layout) — a
  multi-day kernel + integration project.

## 3. Verdict

- The route is **not worth implementing** as "port/replace with a PCIe-safe
  all-to-all" on this hardware/stack:
  - at DP's actual low-concurrency batch sizes (M≈1-4 per rank) the two-way
    collective gain is ~0% or slightly negative;
  - DP's C1-C16 deficit is also architectural (one request lives on one GPU,
    attention/dense replicated), so even free comm would leave DP behind TP8
    until C≈32;
  - vLLM has no a2a EP backend left to switch on, and DeepEP cannot run here.
- Practical recommendation stands: TP8 for latency/mixed serving; DP attention
  only for saturated throughput (C≥32, +14-16%) or KV-capacity-heavy serving
  (≈8x aggregate KV).
- If low-concurrency DP ever becomes a hard requirement, the remaining option
  is a purpose-built PCIe ring/one-shot kernel that first beats NCCL a2a in a
  microbenchmark; the measurements above suggest the upside is small at the
  token counts that matter, so this should be de-prioritized.

## Artifacts
- Microbench: `tools/bench_pcie_a2a.py` (two variants used during the test:
  one-way dispatch and two-way dispatch+combine).
- Also tested `--all2all-backend naive`: falls back to AG+RS (identical perf).
