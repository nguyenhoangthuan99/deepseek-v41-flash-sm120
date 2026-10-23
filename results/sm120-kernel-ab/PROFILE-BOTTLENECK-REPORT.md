# SM120 vLLM kernel profiling — bottleneck report and tuning plan

Profiled the production serving stack of `deepseek-v41-flash-sm120` on VM106
(8× RTX PRO 6000, SM120, PCIe-only) with the torch profiler on every rank.

- Server: TP8, max_model_len 1,048,576, block 64, DSpark K3, FULL_DECODE_ONLY
  CUDA graphs, `linear_backend=flashinfer_b12x`, DeepGEMM MoE alignment 64,
  NCCL_P2P_LEVEL=PHB, `SPARSE_MLA_ENGINE=fast` (vLLM native FlashInfer kernel).
- Method: one warmup batch per workload (profiler off), then `POST
  /start_profile` → same workload → `POST /stop_profile`; per-rank Chrome
  traces archived; kernel time aggregated over rank0/rank1 events with
  `cat=kernel`. Raw windows: `results/sm120-kernel-ab/profile_windows.json`,
  full family/top-kernel tables: `profile_summary.md` / `profile_summary.json`.
- Caveat: kernel-time shares treat asynchronous NCCL kernels as serialized busy
  time; since AllReduce runs concurrently with compute, its share is an
  upper-bound attribution of GPU *occupancy*, but on this PCIe-only node it is
  also the dominant critical-path cost (independently observed as 53–55% of
  GPU busy time in the repo's earlier torch-profiler runs).

## Measured workloads (warmed, profiler window)

| workload | wall | kernels (rank0) | kernel time (rank0) |
|---|---:|---:|---:|
| prefill 32k, C1, 1 out | 2.61 s | 5,992 | 2,420 ms |
| prefill 8k, C8, 1 out | 5.25 s | 12,050 | 5,024 ms |
| mixed 16k, C4, 64 out | 5.61 s | 64,109 | 5,352 ms |
| decode C1, 512 in, 256 out | 2.07 s | 169,099 | 1,994 ms |
| decode C8, 512 in, 128 out | 2.77 s | 130,577 | 2,559 ms |
| decode C32, 512 in, 64 out | 3.90 s | 95,280 | 3,508 ms |

## GPU kernel-time by family (% of summed kernel time)

| family | prefill 32k C1 | prefill 8k C8 | mixed 16k C4 | decode C1 | decode C8 | decode C32 |
|---|---:|---:|---:|---:|---:|---:|
| **NCCL (AllReduce/AllGather)** | **66.7%** | **68.0%** | **63.4%** | **58.1%** | **49.5%** | **54.9%** |
| **MoE grouped GEMM (DeepGEMM)** | 8.3% | 8.3% | 10.6% | 11.1% | **25.9%** | **23.0%** |
| **sparse MLA (FlashInfer)** | 12.0% | 11.3% | 12.1% | 13.0% | 9.8% | 9.1% |
| dense GEMM (b12x / CUTLASS mxfp8) | 1.9% | 1.8% | 2.3% | 6.1% | 4.2% | 3.4% |
| MHC + norm (TileLang) | 5.7% | 5.6% | 5.7% | 3.7% | 4.1% | 3.4% |
| engram/embed/gather | 2.3% | 2.2% | 2.0% | 0.7% | 0.8% | 1.3% |
| KV write / compress | 0.6% | 0.5% | 0.6% | 1.0% | 0.7% | 0.5% |
| indexer/topk | 0.3% | 0.2% | 0.4% | 0.8% | 0.5% | 0.4% |
| sampler | ~0% | ~0% | ~0% | 0.1% | 0.1% | 0.1% |

## Top individual kernels

1. `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` — **47–67%** of kernel time in
   every workload (single largest consumer by far).
2. `deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<... 768, 5120 ...>` (MoE FC1) —
   3.1–15.1%; `... 5120, 384 ...` (MoE FC2) — 0.8–6.1%.
3. FlashInfer `sparse_mla_decode_dsv4_kernel` (decode, 1.9–2.8%) and
   `sparse_mla_prefill_mg_dual_kernel` (prefill/mixed, 3.2–4.1%).
4. MHC TileLang: `mhc_post_tilelang_kernel` 1.9–3.8%,
   `mhc_pre_big_fuse_with_norm_tilelang_kernel` 1.1–1.9%,
   `mhc_fused_tilelang_kernel` up to 2.5%; `sm120_tf32_hc_prenorm_gemm` 1.2–1.9%.
5. Dense: FlashInfer b12x blockscaled mxfp8 kernels 0.9–2.1% each, plus several
   small CUTLASS WMMA/bf16 kernels (~1–2% each) — many small kernel launches,
   not one hot kernel.
6. Minor: `ncclDevKernel_AllGather_RING_LL` 1.0–3.1%, `_fwd_kernel_ep_gather`
   1.5–1.6%, `_silu_mul_quant_fp8_packed_kernel` 1.1–1.2%,
   `_mask_candidates_kernel` 0.7–1.1%, `_fwd_kernel_ep_scatter_2` 0.7%,
   `sm120_fp8_mqa_logits` 0.7%.

## Which kernels are worth tuning (and which are not)

| target | evidence | tuning levers | expected upside |
|---|---|---|---|
| **NCCL allreduce** | 47–68% of kernel time; earlier A/B: Simple +14% C32 but −24% C1/−17% C4 | `NCCL_PROTO` (Simple/LL/LL128), `NCCL_ALGO` (Ring/Tree), channel counts (`MIN/MAX_NCHANNELS`), `NSOCKS_PERTHREAD`/`SOCKET_NTHREADS`, and per-workload env selection | **highest** (several % to >10% end-to-end if the right protocol is chosen per workload) |
| **MoE grouped GEMM** | 23–26% of decode C8/C32, 8.3% prefill | `deep_gemm_moe_small_m_alignment` (32/64/128 per batch), DeepGEMM tile/autotune configs, fp8/fp4 path selection, `silu_mul_quant` fusion | high for saturated decode |
| **sparse MLA (FlashInfer SM120)** | 12% prefill, 9–13% decode | FlashInfer per-shape dispatch/autotuner overrides (`chunks_per_block`), prefill vs decode path, `max_num_batched_tokens`, graph bucket sizes | moderate |
| **MHC TileLang** | 3.4–5.7% | TileLang block/warp parameters, prenorm TF32 GEMM tile choice, collapse/norm fusion selection | moderate (small but consistent) |
| dense GEMM (b12x/CUTLASS) | 1.8–6.1%, spread over many small kernels | per-shape backend/recipe selection, larger-M paths, reduce launch count | low |
| engram / KV-write / indexer / sampler | ≤2.3% each, ≤1% most | — | not worth it |

## Tuning plan

1. **Phase 1 — NCCL (biggest lever).** Warmed endpoint sweep over
   `{default, NCCL_PROTO=Simple, NCCL_PROTO=LL128, NCCL_ALGO=Tree}` ×
   `{C1,4,8,16,32}` × `{512, 4096, 16384, 32768}` tokens; pick per-workload
   winner, then decide a single default (or expose a documented env recommendation
   because env is process-wide, not per-request).
2. **Phase 2 — MoE.** Sweep `deep_gemm_moe_small_m_alignment ∈ {32,64,128}` and
   DeepGEMM tile configs across per-rank M buckets (decode 3–256 rows, prefill
   512–8192) on one GPU microbench, wire a shape-based alignment table into
   `--kernel-config`.
3. **Phase 3 — sparse MLA.** FlashInfer SM120 autotuner/dispatch overrides per
   (tokens, heads, topk, context) on one GPU; extend the existing shape table.
4. **Phase 4 — MHC + dense GEMM.** TileLang MHC parameter sweep; b12x/CUTLASS
   backend-per-shape selection. Keep only changes that win beyond measurement
   spread.
5. Re-profile the final configuration and re-run the warmed VM106-vs-VM100
   endpoint benchmark.

Dense-GEMM micro-tuning and the <2% kernels are explicitly de-prioritized: their
share of GPU time does not justify the risk.
