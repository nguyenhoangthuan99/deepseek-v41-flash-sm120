// cuBLASLt MXFP8 (e4m3 payload, e8m0 1x32 block scales) GEMM: D = A @ B^T, bf16 out.
// The V4.1-Flash 32x32 weight blocks expand losslessly to 1x32 rows, so this
// computes the same product as the serving W8A8 kernels (modulo scale-format
// rounding handled by the caller, which requantizes payloads exactly).
//
// Requires CUDA >= 12.8 cublasLt with CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0.
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

#define CHECK(x)                                                    \
    do {                                                            \
        auto st_ = (x);                                             \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                         \
            std::snprintf(err_msg, sizeof(err_msg), "%s -> %d", #x, \
                          (int)st_);                                \
            return -(int)st_;                                       \
        }                                                           \
    } while (0)

static char err_msg[256];
extern "C" const char* last_error() { return err_msg; }

static cublasLtHandle_t handle = nullptr;
static void* workspace = nullptr;
static const size_t WORKSPACE = 64ull << 20;

// A (M,K) row-major e4m3, B (N,K) row-major e4m3 (used as B^T), scales are
// packed UE8M0 bytes: sa (M, K/32) row-major, sb (N, K/32) row-major.
// cublasLt column-major view: D(N,M) = B(N,K) * A^T(K,M) with OP_T on A.
extern "C" int mxfp8_gemm(const void* A, const void* B, const void* sa,
                          const void* sb, void* D, int M, int N, int K,
                          void* stream) {
    if (!handle) {
        CHECK(cublasLtCreate(&handle));
        if (cudaMalloc(&workspace, WORKSPACE) != cudaSuccess) return -1000;
    }
    cublasLtMatmulDesc_t op;
    CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    // Column-major: compute D_col(N, M) = B_col(K,N)^T * A_col(K,M)
    CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA,
                                         &opT, sizeof(opT)));
    CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB,
                                         &opN, sizeof(opN)));
    cublasLtMatmulMatrixScale_t vec32 =
        CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    CHECK(cublasLtMatmulDescSetAttribute(
        op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &vec32, sizeof(vec32)));
    CHECK(cublasLtMatmulDescSetAttribute(
        op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &vec32, sizeof(vec32)));
    const void* pa = sb;  // A of the column-major problem is our B
    const void* pb = sa;
    CHECK(cublasLtMatmulDescSetAttribute(
        op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &pa, sizeof(pa)));
    CHECK(cublasLtMatmulDescSetAttribute(
        op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &pb, sizeof(pb)));

    cublasLtMatrixLayout_t la, lb, ld;
    // Our row-major B (N,K) == column-major (K,N) with ld=K; transposed above.
    CHECK(cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, K, N, K));
    CHECK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, K, M, K));
    CHECK(cublasLtMatrixLayoutCreate(&ld, CUDA_R_16BF, N, M, N));

    float alpha = 1.f, beta = 0.f;
    cublasLtMatmulPreference_t pref;
    CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WORKSPACE,
        sizeof(WORKSPACE)));
    cublasLtMatmulHeuristicResult_t heur;
    int found = 0;
    CHECK(cublasLtMatmulAlgoGetHeuristic(handle, op, la, lb, ld, ld, pref, 1,
                                         &heur, &found));
    if (!found) {
        std::snprintf(err_msg, sizeof(err_msg), "no heuristic for %dx%dx%d",
                      M, N, K);
        return -2000;
    }
    CHECK(cublasLtMatmul(handle, op, &alpha, B, la, A, lb, &beta, D, ld, D,
                         ld, &heur.algo, workspace, WORKSPACE,
                         (cudaStream_t)stream));
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(ld);
    cublasLtMatmulDescDestroy(op);
    return 0;
}
