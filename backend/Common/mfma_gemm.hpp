/**
 * @file mfma_gemm.hpp
 * @brief Hand-written fp64 MFMA (matrix-core) GEMM for the LDG residual interp/integrate GEMMs.
 *
 * Drop-in replacement for the rocBLAS Dgemm('N','N') calls inside Node2Gauss / Gauss2Node
 * (backend==3, HIP). Computes  C[M,N] = A[M,K] * B[K,N]  in column-major, where A is one of the
 * small constant DG shape operators (shapegt/shapfgt/shapegw/shapfgw, M,K ~ 27..108) and N is large.
 *
 * Why: on CDNA3 (MI300A/gfx942) the rocBLAS Dgemm for these skinny shapes underfills the 228 CUs and
 * leaves the matrix cores idle in the launch tail. A rocWMMA fp64 16x16x4 MFMA kernel with one
 * wavefront (64 lanes) per 16x16 output tile fills the machine and is bit-exact vs rocBLAS for these
 * shapes (measured relerr 0..1e-16). Standalone microbench on the real solve's sizes: interp
 * (K=27) 1.16-1.32x, integrate (K=108) 0.97-1.14x, all bit-exact.
 *
 * Correctness for M,K,N not multiples of 16/4:
 *  - the operator A is PRE-PADDED once to MP x KP (MP=ceil(M/16)*16, KP=ceil(K/4)*4) with zeros, and
 *    cached by pointer, so the A-fragment loads are always in-bounds and the padded rows/cols
 *    contribute exact zeros;
 *  - B loads direct for interior tiles and is staged through zeroed LDS only on the boundary tile;
 *  - C is stored masked (gi<M && gj<N).
 *
 * fp64 only. USE_FLOAT builds and any non-double T fall back to rocBLAS (enabled() returns false).
 * Runtime escape hatch: EXASIM_MFMA=0 disables and reverts to rocBLAS without recompiling.
 */
#pragma once
#ifdef HAVE_HIP
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdlib>
#include <map>
#include <utility>
#include "exasim_stream.hpp"

namespace exasim_mfma {

static constexpr int TM = 16, TN = 16, TK = 4;

// C[M,N] = Apad[MP,KP] * B[K,N], column-major. Apad is the operator pre-padded to MP x KP.
__global__ __launch_bounds__(64)
void mfma_gemm_nn_f64(double* __restrict C, const double* __restrict Apad,
                      const double* __restrict B, int M, int K, int N, int MP, int ldc) {
    using namespace rocwmma;
    const int m0 = blockIdx.x * TM, n0 = blockIdx.y * TN, lane = threadIdx.x;
    fragment<accumulator, TM, TN, TK, double> acc; fill_fragment(acc, 0.0);
    const bool nFull = (n0 + TN <= N);
    __shared__ double Bs[TK * TN];
    for (int k0 = 0; k0 < K; k0 += TK) {
        fragment<matrix_a, TM, TN, TK, double, col_major> a;
        fragment<matrix_b, TM, TN, TK, double, col_major> b;
        load_matrix_sync(a, Apad + m0 + (long)MP * k0, MP);   // padded operator: always in-bounds
        if (nFull && (k0 + TK <= K)) {
            load_matrix_sync(b, B + k0 + (long)K * n0, K);    // interior: direct, staging-free
            mma_sync(acc, a, b, acc);
        } else {
            int kk = lane & 3, j = lane >> 2, gk = k0 + kk, gj = n0 + j;
            Bs[kk + TK * j] = (gk < K && gj < N) ? B[gk + (long)K * gj] : 0.0;
            __syncthreads();
            load_matrix_sync(b, Bs, TK);
            mma_sync(acc, a, b, acc);
            __syncthreads();
        }
    }
    __shared__ double Cs[TM * TN];
    store_matrix_sync(Cs, acc, TM, mem_col_major);
    __syncthreads();
    for (int e = lane; e < TM * TN; e += 64) {
        int i = e & 15, j = e >> 4, gi = m0 + i, gj = n0 + j;
        if (gi < M && gj < N) C[gi + (long)ldc * gj] = Cs[i + TM * j];
    }
}

// Runtime + type gate. MFMA path is fp64-only; EXASIM_MFMA=0 disables it.
template <class T>
inline bool enabled() {
#ifdef USE_FLOAT
    return false;
#else
    if (sizeof(T) != sizeof(double)) return false;
    static const bool on = [](){ const char* e = std::getenv("EXASIM_MFMA"); return !(e && e[0] == '0'); }();
    return on;
#endif
}

// Return a device pointer to the operator A (M x K, col-major ld=M) padded to MP x KP with zeros.
// Cached by (pointer, M, K): the DG shape operators are constant device arrays allocated once, so the
// pad+cache cost is paid on the first residual only. *MPout returns the padded leading dimension.
inline const double* padded_operator(const double* A, int M, int K, int* MPout) {
    const int MP = ((M + TM - 1) / TM) * TM;
    const int KP = ((K + TK - 1) / TK) * TK;
    *MPout = MP;
    static std::map<std::pair<const double*, long>, double*> cache;
    const long key2 = (long)M * 100000L + (long)K;
    auto key = std::make_pair(A, key2);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    double* Apad = nullptr;
    hipMalloc(&Apad, (size_t)MP * KP * sizeof(double));
    hipMemset(Apad, 0, (size_t)MP * KP * sizeof(double));
    // copy the real M x K block (col-major ld=M) into the top-left of Apad (col-major ld=MP)
    hipMemcpy2D(Apad, (size_t)MP * sizeof(double), A, (size_t)M * sizeof(double),
                (size_t)M * sizeof(double), (size_t)K, hipMemcpyDeviceToDevice);
    cache[key] = Apad;
    return Apad;
}

// C[M,N] = A[M,K] * B[K,N], col-major, ldc = leading dim of C. A = operator (gets padded+cached).
inline void gemm_nn(double* C, const double* A, const double* B, int M, int K, int N, int ldc) {
    int MP; const double* Apad = padded_operator(A, M, K, &MP);
    dim3 grid((M + TM - 1) / TM, (N + TN - 1) / TN);
    hipLaunchKernelGGL(mfma_gemm_nn_f64, grid, dim3(64), 0, exasim_stream::compute(), C, Apad, B, M, K, N, MP, ldc);
}

} // namespace exasim_mfma
#endif // HAVE_HIP
