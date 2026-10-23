# Kernel tuning round 2 (post-NCU): sparse-MLA chunks_per_block + MoE prefill alignment

Follow-up to `NCU-REPORT.md`. NCU showed (a) sparse-MLA decode is latency-bound
with <1 wave and (b) MoE prefill is compute/L2-bound with low occupancy.

## 1. FlashInfer SM120 sparse-MLA `chunks_per_block` autotuning — WIRED

FlashInfer's SM120 DSv4 decode kernel exposes a per-shape `chunks_per_block`
tactic (1..num_splits; `_sparse_mla_sm120.py:750` "AutoTuner-driven
chunks_per_block tuning"). Kernel-level graph-captured measurements:

| shape | heuristic | best explicit cpb | delta |
|---|---:|---:|---:|
| T=2..16, H=16, topk=512 | 18.46 µs | cpb=1 18.45 µs | 0% |
| T=32, H=16, topk=512 | 22.57 µs | heuristic already optimal | 0% |
| **T=64, H=16, topk=512 (single-SWA segment)** | **45.10 µs** | **cpb=4 32.81 µs** | **−27%** |
| T=64, H=16, production mixed ABI (SWA 128 + extra 512) | 42.81 µs | autotuned 38.96 µs | **−9%** |

The production vLLM path uses the mixed ABI (128 SWA columns + 512 compressed
entries, `extra_sparse_indices`, kv_layout="NHD"), so a cache was generated with
that exact signature via FlashInfer's AutoTuner (`autotune(True, cache=...)`):

- `results/sm120-kernel-ab/fi_autotune/sparse_mla_sm120_decode_dsv4.json`
  (24 entries: T=2..64, H=16/64, topk=512) — produced by
  `tools/gen_fi_cache_prod.py`, verified with `tools/verify_fi_cache_prod.py`.
- `serve-vllm.sh` mounts it at `/opt/fi_autotune` and sets
  `FLASHINFER_AUTOTUNE_DIR`; FlashInfer loads it lazily (mtime-gated), so the
  serving path picks the tuned tactic with no code change.
- Numerics: different split order ⇒ rel-L2 ≈1.7e-3 vs heuristic (inside the
  attention tolerance); correctness smoke on the served endpoint passed.

End-to-end (512-token prompts, DSpark K3, warmed, rounds=2, vs the pre-tuning
server on the same box):
| workload | baseline | tuned |
|---|---:|---:|
| C4 TPOT / tok/s | 10.18 ms / 311 | **9.94 ms / 318** |
| C16 TPOT / tok/s | 17.60 ms / 627 | **17.03 ms / 613** |
| C32 TPOT / tok/s | 27.62 ms / 773 | **27.24 ms / 780** |

The kernel-level T=64 win is solid (9–27% depending on ABI); the end-to-end
delta is small and partly inside run-to-run noise (±5%), so treat it as a
no-regression change with a proven kernel improvement on the C16 decode bucket.
Recommendation: keep the cache mounted (it only affects the T=64 bucket; remove
`FI_AUTOTUNE_DIR` to fall back to the heuristic).

## 2. MoE prefill alignment sweep — production rule already optimal

Standalone DeepGEMM grouped-MoE sweep (E=56, alignment 64 vs 128), FP4 and FP8:

| case / T | align 64 | align 128 | winner |
|---|---:|---:|---|
| fp4 fc1 T=1024 | 772 µs | 824 µs | 64 |
| fp4 fc1 T=2048 | 1156 | 1104 | 128 (+4.5%) |
| fp4 fc1 T=4096 | 2059 | 1928 | 128 (+6.4%) |
| fp4 fc1 T=8192 | 3955 | 3553 | 128 (+10.2%) |
| fp4 fc2 T=8192 | 2073 | 1858 | 128 (+10.4%) |
| fp8 fc1 T=8192 | 4524 | 3955 | 128 (+12.6%) |
| fp8 fc2 T=8192 | 2297 | 2069 | 128 (+9.9%) |

So alignment 64 is right for small/decode M and **128 is materially better for
prefill M≥2048** (up to +10–13%). The deployed rule already uses 64 for small M
and 128 for large M (`deep_gemm_moe_small_m_alignment=64` only affects the
small-M threshold), so no change is needed; the sweep confirms the production
config and gives the crossover (~1024–2048 rows).

## 3. Notes / next
- NCU's sparse-MLA decode "latency-bound with idle SMs" is partly inherent
  (launch/latency floor); the only exploitable knob found is `chunks_per_block`,
  now tuned for the production ABI.
- Remaining unprofiled families: MHC TileLang (3.4–5.7%) and dense b12x/CUTLASS
  GEMM (1.8–6.1%); both are small enough that expected gains are low.
- Do not regenerate the autotune cache on a different image/GPU without
  re-running `tools/gen_fi_cache_prod.py` (cache metadata is environment-gated).
