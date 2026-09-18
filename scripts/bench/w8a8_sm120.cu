// SM120 W8A8 block-scaled GEMM: C = (A ⊙ As) @ (B ⊙ Bs)^T
//   A  (M,K) row-major fp8e4m3, As (M, K/32) fp32 per-token per-32-col scales
//   B  (N,K) row-major fp8e4m3, Bs (N/32, K/32) fp32 per 32x32 block
//   Cpart (SPLITK, M, N) fp32 partials; reduced by the caller.
//
// One mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 atom consumes one
// 32-wide scale group, so each group partial is scaled by As[row]*Bs[nblk][kg]
// in registers and accumulated in fp32 — same numerics contract as the tuned
// Triton kernel (per-group fp32 accumulation, split-K partials).
//
// mma m16n8k32 8-bit thread mapping (PTX ISA):
//   lane -> group g = lane/4 (0..7), t = lane%4
//   A (row-major m16 x k32): a0=[g][4t..4t+3] a1=[g+8][4t..] a2=[g][16+4t..] a3=[g+8][16+4t..]
//   B (col .col n8 x k32):   b0=[n=g][k=4t..4t+3] b1=[n=g][k=16+4t..]
//   C: d0=[g][2t] d1=[g][2t+1] d2=[g+8][2t] d3=[g+8][2t+1]
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

#ifndef BM
#define BM 64
#endif
#ifndef BN
#define BN 64
#endif
#ifndef WM
#define WM 2
#endif
#ifndef WN
#define WN 2
#endif
#ifndef STAGES
#define STAGES 2
#endif

#define WARPS (WM * WN)
#define KG 32
#define ATOM_M 16
#define ATOM_N 8
#define TM (BM / (WM * ATOM_M))
#define TN (BN / (WN * ATOM_N))

static_assert(TM >= 1 && BM % (WM * ATOM_M) == 0, "BM tiling");
static_assert(TN >= 1 && BN % (WN * ATOM_N) == 0, "BN tiling");
static_assert(STAGES >= 2, "pipeline needs >=2 stages");

#define CP_ASYNC_16(dst, src) \
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src))
#define CP_COMMIT() asm volatile("cp.async.commit_group;\n")
#define CP_WAIT(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))

__global__ void __launch_bounds__(WARPS * 32)
w8a8_block_gemm(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                const float* __restrict__ As, const float* __restrict__ Bs,
                float* __restrict__ Cpart, int M, int N, int K, int splitk) {
    const int kgroups_total = K / KG;
    const int kg_per_split = (kgroups_total + splitk - 1) / splitk;
    const int kg_begin = blockIdx.z * kg_per_split;
    const int kg_end = min(kg_begin + kg_per_split, kgroups_total);

    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int wm = warp / WN, wn = warp % WN;
    const int g = lane / 4, t = lane % 4;

    __shared__ __align__(16) uint8_t sA[STAGES][BM][KG];
    __shared__ __align__(16) uint8_t sB[STAGES][BN][KG];

    const int threads = WARPS * 32;

    auto load_stage = [&](int stage, int kg) {
        const uint8_t* gA = A + (size_t)kg * KG;
        const uint8_t* gB = B + (size_t)kg * KG;
        uint32_t base_a = (uint32_t)__cvta_generic_to_shared(&sA[stage][0][0]);
        uint32_t base_b = (uint32_t)__cvta_generic_to_shared(&sB[stage][0][0]);
        for (int idx = threadIdx.x; idx < BM * 2; idx += threads) {
            int row = idx >> 1, half = idx & 1;
            int grow = block_m + row;
            if (grow < M)
                CP_ASYNC_16(base_a + row * KG + half * 16,
                            gA + (size_t)grow * K + half * 16);
        }
        for (int idx = threadIdx.x; idx < BN * 2; idx += threads) {
            int row = idx >> 1, half = idx & 1;
            int grow = block_n + row;
            if (grow < N)
                CP_ASYNC_16(base_b + row * KG + half * 16,
                            gB + (size_t)grow * K + half * 16);
        }
        CP_COMMIT();
    };

    float acc[TM][TN][4] = {};

    for (int s = 0; s < STAGES - 1 && kg_begin + s < kg_end; ++s)
        load_stage(s, kg_begin + s);

    for (int kg = kg_begin; kg < kg_end; ++kg) {
        const int stage = (kg - kg_begin) % STAGES;
        const int next = kg + STAGES - 1;
        if (next < kg_end) {
            load_stage((next - kg_begin) % STAGES, next);
            CP_WAIT(STAGES - 1);
        } else {
            CP_WAIT(0);
        }
        __syncthreads();

#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const int arow = wm * TM * ATOM_M + i * ATOM_M;
            uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sA[stage][arow + g][4 * t]);
            uint32_t a1 = *reinterpret_cast<const uint32_t*>(&sA[stage][arow + g + 8][4 * t]);
            uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sA[stage][arow + g][16 + 4 * t]);
            uint32_t a3 = *reinterpret_cast<const uint32_t*>(&sA[stage][arow + g + 8][16 + 4 * t]);
            const int r0 = block_m + arow + g;
            const float s0 = (r0 < M) ? As[(size_t)r0 * kgroups_total + kg] : 0.f;
            const float s1 = (r0 + 8 < M) ? As[(size_t)(r0 + 8) * kgroups_total + kg] : 0.f;
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int brow = wn * TN * ATOM_N + j * ATOM_N;
                uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sB[stage][brow + g][4 * t]);
                uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sB[stage][brow + g][16 + 4 * t]);
                float d0, d1, d2, d3;
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                      "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
                const float bs = Bs[(size_t)((block_n + brow) / KG) * kgroups_total + kg];
                acc[i][j][0] += d0 * (s0 * bs);
                acc[i][j][1] += d1 * (s0 * bs);
                acc[i][j][2] += d2 * (s1 * bs);
                acc[i][j][3] += d3 * (s1 * bs);
            }
        }
        __syncthreads();
    }

    float* Cout = Cpart + (size_t)blockIdx.z * M * N;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r0 = block_m + wm * TM * ATOM_M + i * ATOM_M + g;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int c0 = block_n + wn * TN * ATOM_N + j * ATOM_N + 2 * t;
            if (c0 + 1 < N) {
                if (r0 < M) {
                    Cout[(size_t)r0 * N + c0] = acc[i][j][0];
                    Cout[(size_t)r0 * N + c0 + 1] = acc[i][j][1];
                }
                if (r0 + 8 < M) {
                    Cout[(size_t)(r0 + 8) * N + c0] = acc[i][j][2];
                    Cout[(size_t)(r0 + 8) * N + c0 + 1] = acc[i][j][3];
                }
            } else if (c0 < N) {
                if (r0 < M) Cout[(size_t)r0 * N + c0] = acc[i][j][0];
                if (r0 + 8 < M) Cout[(size_t)(r0 + 8) * N + c0] = acc[i][j][2];
            }
        }
    }
}

extern "C" int launch_w8a8(const void* A, const void* B, const void* As,
                           const void* Bs, void* Cpart, int M, int N, int K,
                           int splitk, void* stream) {
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, splitk);
    w8a8_block_gemm<<<grid, WARPS * 32, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)A, (const uint8_t*)B, (const float*)As,
        (const float*)Bs, (float*)Cpart, M, N, K, splitk);
    return (int)cudaGetLastError();
}
