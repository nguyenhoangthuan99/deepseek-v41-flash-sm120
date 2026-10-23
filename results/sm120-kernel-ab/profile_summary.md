# vLLM SM120 kernel profile (torch profiler, TP8)

## decode_c1
- traces: 9  rank0 kernels: 169099  rank0 total kernel time: 1993.5 ms  (mean over up to 4 ranks: 1984.0 ms)

| family | ms | share |
|---|---:|---:|
| nccl | 1152.6 | 58.1% |
| sparse_mla_attn | 258.9 | 13.0% |
| moe | 220.7 | 11.1% |
| dense_gemm | 121.3 | 6.1% |
| other | 85.1 | 4.3% |
| norm_mhc | 73.1 | 3.7% |
| elementwise_copy | 21.3 | 1.1% |
| kv_cache_write | 19.3 | 1.0% |
| indexer_topk | 15.1 | 0.8% |
| engram_embed | 13.8 | 0.7% |
| sampler | 2.8 | 0.1% |

| top kernel | ms | share |
|---|---:|---:|
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 1134.54 | 57.19% |
| `void deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0u, 768u, 5120u, 128u, 32u, 384u…` | 109.71 | 5.53% |
| `void deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0u, 5120u, 384u, 128u, 32u, 384u…` | 61.75 | 3.11% |
| `void flashinfer::sparse_mla_sm120::sparse_mla_decode_dsv4_kernel<` | 43.04 | 2.17% |
| `mhc_fused_tilelang_kernel` | 36.51 | 1.84% |
| `mhc_pre_big_fuse_with_norm_tilelang_kernel` | 33.41 | 1.68% |
| `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm120_b12xD…` | 32.43 | 1.63% |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_12…` | 32.26 | 1.63% |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_s161616gemm_bf16_32x32_64x1_tn…` | 30.86 | 1.56% |
| `_ZN7cutlass13device_kernelIN10flashinfer4gemm54DeviceGemmMxfp8GemmSm120___nv_…` | 26.32 | 1.33% |
| `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm120_b12xD…` | 26.24 | 1.32% |
| `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm120_b12xD…` | 25.20 | 1.27% |
| `_ZN7cutlass13device_kernelIN10flashinfer4gemm53DeviceGemmMxfp8GemmSm120___nv_…` | 21.22 | 1.07% |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64…` | 20.41 | 1.03% |

## decode_c32
- traces: 9  rank0 kernels: 95280  rank0 total kernel time: 3507.5 ms  (mean over up to 4 ranks: 3367.5 ms)

| family | ms | share |
|---|---:|---:|
| nccl | 1847.3 | 54.9% |
| moe | 774.9 | 23.0% |
| sparse_mla_attn | 306.2 | 9.1% |
| dense_gemm | 115.2 | 3.4% |
| norm_mhc | 113.7 | 3.4% |
| other | 104.9 | 3.1% |
| engram_embed | 43.7 | 1.3% |
| elementwise_copy | 27.8 | 0.8% |
| kv_cache_write | 18.3 | 0.5% |
| indexer_topk | 13.4 | 0.4% |
| sampler | 2.1 | 0.1% |

| top kernel | ms | share |
|---|---:|---:|
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 1744.45 | 51.80% |
| `void deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0u, 768u, 5120u, 128u, 32u, 384u…` | 372.91 | 11.07% |
| `void deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0u, 5120u, 384u, 128u, 32u, 384u…` | 187.16 | 5.56% |
| `ncclDevKernel_AllGather_RING_LL` | 102.80 | 3.05% |
| `void deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0u, 768u, 5120u, 128u, 32u, 384u…` | 73.81 | 2.19% |
| `mhc_post_tilelang_kernel` | 64.70 | 1.92% |
| `void sparse_mla_prefill_mg_dual_kernel<` | 59.62 | 1.77% |
| `void deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0u, 5120u, 384u, 128u, 32u, 384u…` | 54.29 | 1.61% |
| `void sparse_mla_prefill_mg_dual_kernel<` | 41.91 | 1.24% |
| `mhc_pre_big_fuse_with_norm_tilelang_kernel` | 41.18 | 1.22% |
| `void deep_gemm::sm120_tf32_hc_prenorm_gemm_impl<24u, 20480u, 128u, 32u, 64u, …` | 39.73 | 1.18% |
| `_silu_mul_quant_fp8_packed_kernel` | 38.26 | 1.14% |
| `_ZN7cutlass13device_kernelIN10flashinfer4gemm55DeviceGemmMxfp8GemmSm120___nv_…` | 38.20 | 1.13% |
| `_mask_candidates_kernel` | 37.67 | 1.12% |

## decode_c8
- traces: 9  rank0 kernels: 130577  rank0 total kernel time: 2559.4 ms  (mean over up to 4 ranks: 2540.8 ms)

| family | ms | share |
|---|---:|---:|
| nccl | 1258.1 | 49.5% |
| moe | 658.3 | 25.9% |
| sparse_mla_attn | 248.8 | 9.8% |
| dense_gemm | 107.2 | 4.2% |
| norm_mhc | 104.8 | 4.1% |
| other | 89.5 | 3.5% |
| elementwise_copy | 21.6 | 0.9% |
| engram_embed | 20.0 | 0.8% |
| kv_cache_write | 17.0 | 0.7% |
| indexer_topk | 12.7 | 0.5% |
| sampler | 2.6 | 0.1% |

| top kernel | ms | share |
|---|---:|---:|
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 1208.34 | 47.56% |
| `void deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0u, 768u, 5120u, 128u, 32u, 384u…` | 384.45 | 15.13% |
| `void deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0u, 5120u, 384u, 128u, 32u, 384u…` | 155.25 | 6.11% |
| `mhc_fused_tilelang_kernel` | 64.58 | 2.54% |
| `void deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0u, 768u, 5120u, 128u, 32u, 384u…` | 51.24 | 2.02% |
| `ncclDevKernel_AllGather_RING_LL` | 49.76 | 1.96% |
| `void flashinfer::sparse_mla_sm120::sparse_mla_decode_dsv4_kernel<` | 47.62 | 1.87% |
| `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm120_b12xD…` | 36.65 | 1.44% |
| `void deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0u, 5120u, 384u, 128u, 32u, 384u…` | 33.89 | 1.33% |
| `_silu_mul_quant_fp8_packed_kernel` | 30.65 | 1.21% |
| `mhc_pre_big_fuse_with_norm_tilelang_kernel` | 27.85 | 1.10% |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x…` | 26.21 | 1.03% |
| `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm120_b12xD…` | 24.49 | 0.96% |
| `_mask_candidates_kernel` | 18.83 | 0.74% |

## mixed_16k_c4
- traces: 9  rank0 kernels: 64109  rank0 total kernel time: 5352.4 ms  (mean over up to 4 ranks: 5382.4 ms)

| family | ms | share |
|---|---:|---:|
| nccl | 3415.0 | 63.4% |
| sparse_mla_attn | 653.5 | 12.1% |
| moe | 572.9 | 10.6% |
| norm_mhc | 306.6 | 5.7% |
| dense_gemm | 123.9 | 2.3% |
| engram_embed | 110.2 | 2.0% |
| other | 97.1 | 1.8% |
| elementwise_copy | 46.5 | 0.9% |
| kv_cache_write | 33.2 | 0.6% |
| indexer_topk | 22.6 | 0.4% |
| sampler | 1.0 | 0.0% |

| top kernel | ms | share |
|---|---:|---:|
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 3358.34 | 62.40% |
| `void sparse_mla_prefill_mg_dual_kernel<` | 192.59 | 3.58% |
| `mhc_post_tilelang_kernel` | 184.19 | 3.42% |
| `void sparse_mla_prefill_mg_dual_kernel<` | 170.78 | 3.17% |
| `void deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0u, 768u, 5120u, 128u, 32u, 384u…` | 159.82 | 2.97% |
| `void deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0u, 5120u, 384u, 128u, 32u, 384u…` | 122.66 | 2.28% |
| `_ZN7cutlass13device_kernelIN10flashinfer4gemm55DeviceGemmMxfp8GemmSm120___nv_…` | 122.11 | 2.27% |
| `void deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0u, 768u, 5120u, 128u, 32u, 384u…` | 109.59 | 2.04% |
| `mhc_pre_big_fuse_with_norm_tilelang_kernel` | 102.34 | 1.90% |
| `void deep_gemm::sm120_tf32_hc_prenorm_gemm_impl<24u, 20480u, 128u, 32u, 64u, …` | 90.54 | 1.68% |
| `_fwd_kernel_ep_gather` | 82.14 | 1.53% |
| `ncclDevKernel_AllGather_RING_LL` | 56.61 | 1.05% |
| `_ZN7cutlass13device_kernelIN10flashinfer4gemm54DeviceGemmMxfp8GemmSm120___nv_…` | 54.87 | 1.02% |
| `void deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0u, 5120u, 384u, 128u, 32u, 384u…` | 42.78 | 0.79% |

## prefill_32k_c1
- traces: 9  rank0 kernels: 5992  rank0 total kernel time: 2420.3 ms  (mean over up to 4 ranks: 2422.7 ms)

| family | ms | share |
|---|---:|---:|
| nccl | 1616.8 | 66.7% |
| sparse_mla_attn | 290.1 | 12.0% |
| moe | 201.8 | 8.3% |
| norm_mhc | 137.0 | 5.7% |
| engram_embed | 55.7 | 2.3% |
| dense_gemm | 45.1 | 1.9% |
| other | 33.6 | 1.4% |
| elementwise_copy | 21.6 | 0.9% |
| kv_cache_write | 13.4 | 0.6% |
| indexer_topk | 7.5 | 0.3% |
| sampler | 0.1 | 0.0% |

| top kernel | ms | share |
|---|---:|---:|
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 1593.54 | 65.78% |
| `void sparse_mla_prefill_mg_dual_kernel<` | 98.26 | 4.06% |
| `mhc_post_tilelang_kernel` | 91.52 | 3.78% |
| `void sparse_mla_prefill_mg_dual_kernel<` | 87.84 | 3.63% |
| `void deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0u, 768u, 5120u, 128u, 32u, 384u…` | 74.69 | 3.08% |
| `_ZN7cutlass13device_kernelIN10flashinfer4gemm55DeviceGemmMxfp8GemmSm120___nv_…` | 60.91 | 2.51% |
| `void deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0u, 5120u, 384u, 128u, 32u, 384u…` | 56.34 | 2.33% |
| `void deep_gemm::sm120_tf32_hc_prenorm_gemm_impl<24u, 20480u, 128u, 32u, 64u, …` | 45.39 | 1.87% |
| `mhc_pre_big_fuse_with_norm_tilelang_kernel` | 44.27 | 1.83% |
| `_fwd_kernel_ep_gather` | 39.13 | 1.62% |
| `_ZN7cutlass13device_kernelIN10flashinfer4gemm54DeviceGemmMxfp8GemmSm120___nv_…` | 26.55 | 1.10% |
| `ncclDevKernel_AllGather_RING_LL` | 23.29 | 0.96% |
| `_fwd_kernel_ep_scatter_2` | 17.87 | 0.74% |
| `void deep_gemm::sm120_fp8_mqa_logits<32u, 128u, false, 4u, 128u, 2u, 3u, 188u…` | 16.96 | 0.70% |

## prefill_8k_c8
- traces: 9  rank0 kernels: 12050  rank0 total kernel time: 5024.2 ms  (mean over up to 4 ranks: 5013.2 ms)

| family | ms | share |
|---|---:|---:|
| nccl | 3410.8 | 68.0% |
| sparse_mla_attn | 564.7 | 11.3% |
| moe | 414.5 | 8.3% |
| norm_mhc | 279.9 | 5.6% |
| engram_embed | 108.1 | 2.2% |
| dense_gemm | 91.6 | 1.8% |
| other | 66.4 | 1.3% |
| elementwise_copy | 41.2 | 0.8% |
| kv_cache_write | 26.3 | 0.5% |
| indexer_topk | 9.6 | 0.2% |
| sampler | 0.2 | 0.0% |

| top kernel | ms | share |
|---|---:|---:|
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 3364.16 | 67.11% |
| `void sparse_mla_prefill_mg_dual_kernel<` | 189.86 | 3.79% |
| `mhc_post_tilelang_kernel` | 183.96 | 3.67% |
| `void deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0u, 768u, 5120u, 128u, 32u, 384u…` | 165.80 | 3.31% |
| `void sparse_mla_prefill_mg_dual_kernel<` | 165.50 | 3.30% |
| `void deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0u, 5120u, 384u, 128u, 32u, 384u…` | 127.51 | 2.54% |
| `_ZN7cutlass13device_kernelIN10flashinfer4gemm55DeviceGemmMxfp8GemmSm120___nv_…` | 117.81 | 2.35% |
| `mhc_pre_big_fuse_with_norm_tilelang_kernel` | 93.46 | 1.86% |
| `void deep_gemm::sm120_tf32_hc_prenorm_gemm_impl<24u, 20480u, 128u, 32u, 64u, …` | 90.42 | 1.80% |
| `_fwd_kernel_ep_gather` | 79.01 | 1.58% |
| `_ZN7cutlass13device_kernelIN10flashinfer4gemm54DeviceGemmMxfp8GemmSm120___nv_…` | 55.36 | 1.10% |
| `ncclDevKernel_AllGather_RING_LL` | 46.62 | 0.93% |
| `_fwd_kernel_ep_scatter_2` | 36.21 | 0.72% |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_3…` | 35.47 | 0.71% |
