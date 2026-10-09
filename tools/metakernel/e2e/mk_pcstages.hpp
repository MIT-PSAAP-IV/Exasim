// mk_pcstages.hpp -- replacement kernels for the LDG block-Jacobian preconditioner build (ldgblockjacobian.cpp).
// Include after the Exasim unity TU. Sizes are the case's (mk_case_sizes.hpp).
//
//   mkpc::bminv(...)   uEquationSchurBlockLDG's B.MinvC (D +=) and B.MinvE (F -=) for all (mcomp, ncomp, d) of one element
//                      block in two kernels (production: 2 x 3 x 81 strided-batched 27x27x27 / 27x54x27 GEMMs + 486 scatters).
//                      One wave per (element, mcomp, ncomp); the contraction over k runs on the fp64 matrix cores; each entry
//                      gets production's sequence D += w_x; D += w_y; D += w_z (and F -= w_x; F -= w_y; F -= w_z).
//   mkpc::fg(...)      uEquationSchurBlockLDG's D += F.G, with G >99% zeros: per column the nonzero rows of G are found with a
//                      wave ballot in ascending row order (deterministic), then D[:, col] += sum_r F[:, r] G[r, col].
//   mkpc::inverse(...) hipComputeInverse without its per-call hipMalloc / pointer-array uploads / hipFree: the same
//                      getrfBatched + getriBatched + copy-back, with the workspace cached per (matrix, batch).
#pragma once
#include "mk_jacstages.hpp"
#include "mk_gmstages.hpp"
#include <map>
#include <tuple>
#include <vector>
#include <cstring>

namespace mkpc {
constexpr int NPE = MK_NPE, NCU = MK_NCU, NDF = 54, N = NPE * NCU, MM = NDF * NCU;
typedef double d4 __attribute__((ext_vector_type(4)));

// D part: D[(i + npe m) + n (j + npe c) + n^2 e] += w_d[i][j],  w_d = B_d(m,c)[i,k] MinvC_d(e)[k,j]
extern "C" __global__ void __launch_bounds__(64) mkpc_bminvc(double* D, const double* B, const double* C, const int ne,
                                                              const size_t cstride, const int nd)
{
    const int l = threadIdx.x, lr = l % 16, lk = l / 16;
    const int e = blockIdx.x % ne, t = blockIdx.x / ne, m = t % NCU, c = t / NCU;
    const size_t M = (size_t)NPE*NPE, L = M*ne, K = L*NCU, Bd = M*NCU*NCU*ne;
    d4 acc[3][2][2];
    // d unrolled at compile time (nd <= 3, guarded): a runtime index into acc[] forced it to scratch (416 B spill)
#pragma unroll
    for (int d = 0; d < 3; d++) {
        if (d >= nd) break;
        const double* A = B + Bd*d + L*m + K*c + M*e;             // A[i + npe k]
        const double* Cm = C + cstride*d + M*e;                     // Cm[k + npe j]
#pragma unroll
        for (int rt = 0; rt < 2; rt++)
#pragma unroll
            for (int ct = 0; ct < 2; ct++) acc[d][rt][ct] = (d4){0.0, 0.0, 0.0, 0.0};
#pragma unroll
        for (int ks = 0; ks < (NPE + 3) / 4; ks++) {
            const int k = 4*ks + lk; const bool kin = k < NPE;
            double av[2], bv[2];
#pragma unroll
            for (int rt = 0; rt < 2; rt++) { const int i = 16*rt + lr; av[rt] = (kin && i < NPE) ? A[i + NPE*k] : 0.0; }
#pragma unroll
            for (int ct = 0; ct < 2; ct++) { const int j = 16*ct + lr; bv[ct] = (kin && j < NPE) ? Cm[k + NPE*j] : 0.0; }
#pragma unroll
            for (int rt = 0; rt < 2; rt++)
#pragma unroll
                for (int ct = 0; ct < 2; ct++) acc[d][rt][ct] = __builtin_amdgcn_mfma_f64_16x16x4f64(av[rt], bv[ct], acc[d][rt][ct], 0, 0, 0);
        }
    }
#pragma unroll
    for (int rt = 0; rt < 2; rt++)
#pragma unroll
        for (int ct = 0; ct < 2; ct++)
#pragma unroll
            for (int r = 0; r < 4; r++) {
                const int i = 16*rt + 4*r + lk, j = 16*ct + lr;   // MFMA f64 16x16x4 D layout (measured): row 4r + l/16, col l%16
                if (i < NPE && j < NPE) {
                    const size_t idx = (size_t)(i + NPE*m) + (size_t)N*(j + NPE*c) + (size_t)N*N*e;
                    double v = D[idx];
#pragma unroll
                    for (int d = 0; d < 3; d++) if (d < nd) v += acc[d][rt][ct][r];
                    D[idx] = v;
                }
            }
}

// F part: F[(i + npe m) + n (j + ndf c) + n mm e] -= w_d[i][j],  w_d = B_d(m,c)[i,k] MinvE_d(e)[k,j], j < ndf
extern "C" __global__ void __launch_bounds__(64) mkpc_bminve(double* F, const double* B, const double* E, const int ne,
                                                              const size_t estride, const int nd)
{
    const int l = threadIdx.x, lr = l % 16, lk = l / 16;
    const int e = blockIdx.x % ne, t = blockIdx.x / ne, m = t % NCU, c = t / NCU;
    const size_t M = (size_t)NPE*NPE, L = M*ne, K = L*NCU, Bd = M*NCU*NCU*ne, ME = (size_t)NPE*NDF;
    double v[2][4][4];
#pragma unroll
    for (int rt = 0; rt < 2; rt++)
#pragma unroll
        for (int ct = 0; ct < 4; ct++)
#pragma unroll
            for (int r = 0; r < 4; r++) { const int i = 16*rt + 4*r + lk, j = 16*ct + lr;
                v[rt][ct][r] = (i < NPE && j < NDF) ? F[(size_t)(i + NPE*m) + (size_t)N*(j + NDF*c) + (size_t)N*MM*e] : 0.0; }
    for (int d = 0; d < nd; d++) {
        const double* A = B + Bd*d + L*m + K*c + M*e;
        const double* Em = E + estride*d + ME*e;                   // Em[k + npe j]
        d4 acc[2][4];
#pragma unroll
        for (int rt = 0; rt < 2; rt++)
#pragma unroll
            for (int ct = 0; ct < 4; ct++) acc[rt][ct] = (d4){0.0, 0.0, 0.0, 0.0};
#pragma unroll
        for (int ks = 0; ks < (NPE + 3) / 4; ks++) {
            const int k = 4*ks + lk; const bool kin = k < NPE;
            double av[2], bv[4];
#pragma unroll
            for (int rt = 0; rt < 2; rt++) { const int i = 16*rt + lr; av[rt] = (kin && i < NPE) ? A[i + NPE*k] : 0.0; }
#pragma unroll
            for (int ct = 0; ct < 4; ct++) { const int j = 16*ct + lr; bv[ct] = (kin && j < NDF) ? Em[k + NPE*j] : 0.0; }
#pragma unroll
            for (int rt = 0; rt < 2; rt++)
#pragma unroll
                for (int ct = 0; ct < 4; ct++) acc[rt][ct] = __builtin_amdgcn_mfma_f64_16x16x4f64(av[rt], bv[ct], acc[rt][ct], 0, 0, 0);
        }
#pragma unroll
        for (int rt = 0; rt < 2; rt++)
#pragma unroll
            for (int ct = 0; ct < 4; ct++)
#pragma unroll
                for (int r = 0; r < 4; r++) v[rt][ct][r] -= acc[rt][ct][r];
    }
#pragma unroll
    for (int rt = 0; rt < 2; rt++)
#pragma unroll
        for (int ct = 0; ct < 4; ct++)
#pragma unroll
            for (int r = 0; r < 4; r++) { const int i = 16*rt + 4*r + lk, j = 16*ct + lr;
                if (i < NPE && j < NDF) F[(size_t)(i + NPE*m) + (size_t)N*(j + NDF*c) + (size_t)N*MM*e] = v[rt][ct][r]; }
}

// D[:, col] += F G[:, col] with G's nonzeros gathered per column (ascending r, wave ballot); one wave per (element, column)
extern "C" __global__ void __launch_bounds__(64) mkpc_fg(double* D, const double* F, const double* G, int* overflow)
{
    __shared__ int rows[64]; __shared__ double vals[64];
    const int l = threadIdx.x, col = blockIdx.x % N; const size_t e = blockIdx.x / N;
    const double* Gc = G + (size_t)MM*col + (size_t)MM*N*e;
    int cnt = 0;
    for (int base = 0; base < MM; base += 64) {
        const int r = base + l; const double g = r < MM ? Gc[r] : 0.0;
        const unsigned long long mask = __ballot(g != 0.0);
        const int pos = cnt + __popcll(mask & ((1ull << l) - 1ull));
        if (g != 0.0 && pos < 64) { rows[pos] = r; vals[pos] = g; }
        cnt += __popcll(mask);
    }
    __syncthreads();
    if (cnt > 64) { if (l == 0) atomicAdd(overflow, 1); return; }
    const double* Fe = F + (size_t)N*MM*e;
    for (int i = l; i < N; i += 64) {
        double s = 0.0;
        for (int q = 0; q < cnt; q++) s += Fe[i + (size_t)N*rows[q]] * vals[q];
        const size_t idx = i + (size_t)N*col + (size_t)N*N*e;
        D[idx] = s + D[idx];       // rocBLAS: alpha*(F G) + beta*D with alpha = beta = 1
    }
}

inline int* overflow_flag() { static int* p = nullptr; if (!p) { hipMalloc(&p, sizeof(int)); hipMemset(p, 0, sizeof(int)); } return p; }

// uEquationSchurBlockLDG with the two replacements (layouts D/F unchanged: production's own kernels)
inline void schur(resstruct& res, commonstruct& common, Int jth, bool gen_bminv, bool gen_fg, cublasHandle_t handle, Int backend,
                  bool prelaid = false)       // prelaid: D and F already in the n x n / n x m layouts (the producer wrote them so)
{
    const Int ncu = common.components.ncu, nd = common.grid.nd, npe = common.grid.npe, npf = common.grid.npf, nfe = common.meshsizes.nfe;
    const Int e1 = common.eblks[3*jth]-1, e2 = common.eblks[3*jth+1], ne = e2-e1, n = npe*ncu, m = npf*nfe*ncu;
    dstype *D = res.D, *F = res.F, *G = res.G;
    if (!prelaid) {
    schurMatrixD(res.H, res.D, npe, ncu, ne);
    ArrayCopy(D, res.H, n*n*ne);
    LDGSchurMatrixF(res.H, F, npe, ncu, npf, nfe, ne);
    ArrayCopy(F, res.H, n*m*ne);
    }
    dstype scalar = 1.0;
    if (common.timeparams.wave == 1) scalar = 1.0/common.timestate.dtfactor;
    if (common.components.ncq > 0) {
        if (gen_bminv && scalar == 1.0 && npe == NPE && ncu == NCU && npf*nfe == NDF) {
            const size_t cs = (size_t)npe*npe*common.meshsizes.ne, es = (size_t)npe*npf*nfe*common.meshsizes.ne;
            hipLaunchKernelGGL(mkpc_bminvc, dim3(ne*NCU*NCU), dim3(64), 0, mk_stream(), D, (const double*)res.B, (const double*)&res.C[npe*npe*e1], (int)ne, cs, (int)nd);
            hipLaunchKernelGGL(mkpc_bminve, dim3(ne*NCU*NCU), dim3(64), 0, mk_stream(), F, (const double*)res.B, (const double*)&res.E[npe*npf*nfe*e1], (int)ne, es, (int)nd);
            MK_HIPCHECK("mkpc_bminv");
        } else {
            for (Int d = 0; d < nd; d++) {
                LDGSchurMatrixBMinvC_GEMM(handle, D, &res.B[npe*npe*ncu*ncu*ne*d], &res.C[npe*npe*common.meshsizes.ne*d + npe*npe*e1], res.H,
                                          scalar, npe, ncu, ne, backend); }
            for (Int d = 0; d < nd; d++) {
                LDGSchurMatrixBMinvE_GEMM(handle, F, &res.B[npe*npe*ncu*ncu*ne*d], &res.E[npe*npf*nfe*common.meshsizes.ne*d + npe*npf*nfe*e1], res.H,
                                          scalar, npe, ncu, npf, nfe, ne, backend); }
        }
    }
    if (gen_fg && n == N && m == MM) {
        hipLaunchKernelGGL(mkpc_fg, dim3(ne*N), dim3(64), 0, mk_stream(), D, (const double*)F, (const double*)G, overflow_flag());
        MK_HIPCHECK("mkpc_fg");
    } else {
        PGEMNMStridedBached(handle, n, n, m, one, F, n, G, m, one, D, n, ne, backend);
    }
}

inline int overflow_count() { int h = 0; hipMemcpy(&h, overflow_flag(), sizeof(int), hipMemcpyDeviceToHost); return h; }

// hipComputeInverse with cached workspace (bitwise-identical: the same library calls on the same data)
inline void inverse(cublasHandle_t handle, dstype* A, dstype* C, Int n, Int batch)
{
    struct W { dstype **Ap = nullptr, **Cp = nullptr; Int *ipiv = nullptr, *info = nullptr; };
    static std::map<std::tuple<const void*, const void*, Int, Int>, W> cache;
    auto key = std::make_tuple((const void*)A, (const void*)C, n, batch);
    auto it = cache.find(key);
    if (it == cache.end()) {
        W w; hipMalloc(&w.ipiv, (size_t)n*batch*sizeof(Int)); hipMalloc(&w.info, (size_t)batch*sizeof(Int));
        hipMalloc(&w.Ap, (size_t)batch*sizeof(dstype*)); hipMalloc(&w.Cp, (size_t)batch*sizeof(dstype*));
        std::vector<dstype*> a(batch), c(batch);
        for (Int i = 0; i < batch; i++) { a[i] = A + (size_t)n*n*i; c[i] = C + (size_t)n*n*i; }
        hipMemcpy(w.Ap, a.data(), batch*sizeof(dstype*), hipMemcpyHostToDevice);
        hipMemcpy(w.Cp, c.data(), batch*sizeof(dstype*), hipMemcpyHostToDevice);
        it = cache.emplace(key, w).first;
    }
    W& w = it->second;
    hipblasDgetrfBatched(handle, n, w.Ap, n, w.ipiv, w.info, batch);
    hipblasDgetriBatched(handle, n, w.Ap, n, w.ipiv, w.Cp, n, w.info, batch);
    ArrayCopy(A, C, n*n*batch);
}

// ---- batched in-place inverse: blocked Gauss-Jordan with partial pivoting (the same pivot sequence as getrf's), one
// workgroup per 243x243 matrix, column-major. Pivoting is implicit (rows flagged, never swapped); panel of 16 columns in
// LDS (unblocked GJ there), the rest of the matrix updated with a rank-16 MFMA product:
//   A[i, c] <- [i not a pivot row of this panel] A[i, c] + sum_j Pnew[i, j] Aold[p_j, c]      (c outside the panel)
// Result W satisfies inv(A)[i, j] = W[p_i, q_j], q = p^-1; written to C, then copied back to A (production's contract:
// getri writes C, ArrayCopy(A, C)). Ties in the pivot search go to the smallest row (iamax semantics).
template <int GJT, int GJB>
__global__ void __launch_bounds__(GJT) mkpc_gjinv(double* Aall, double* Call, int skip)
{
    // Ar (old pivot rows) lives in C, which is scratch until the final permuted write
    __shared__ double P[GJB * N];       // new panel, P[j*N + r]
    __shared__ short pan[N];            // panel index a row was pivoted in (-1: not yet)
    __shared__ short piv[N];
    __shared__ double rv[GJT / 64]; __shared__ int ri[GJT / 64]; __shared__ int sp; __shared__ double sd;
    const int t = threadIdx.x, l = t % 64, w = t / 64, lr = l % 16, lk = l / 16;
    double* A = Aall + (size_t)N*N*blockIdx.x; double* C = Call + (size_t)N*N*blockIdx.x; double* Ar = C;
    for (int r = t; r < N; r += GJT) { pan[r] = -1; piv[r] = (short)r; }
    for (int k0 = 0, pb = 0; k0 < N; k0 += GJB, pb++) {
        const int bw = min(GJB, N - k0);
        __syncthreads();
        for (int x = t; x < GJB*N; x += GJT) { const int j = x / N, r = x % N; P[x] = j < bw ? A[r + (size_t)N*(k0 + j)] : 0.0; }
        __syncthreads();
        for (int jj = 0; jj < (skip == 2 ? 0 : bw); jj++) {
            double bv = -1.0; int bi = N;
            for (int r = t; r < N; r += GJT) { const double v = pan[r] < 0 ? fabs(P[jj*N + r]) : -1.0; if (v > bv) { bv = v; bi = r; } }
            for (int o = 32; o > 0; o >>= 1) { const double ov = __shfl_xor(bv, o); const int oi = __shfl_xor(bi, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; } }
            if (l == 0) { rv[w] = bv; ri[w] = bi; }
            __syncthreads();
            if (t == 0) { double v = rv[0]; int i = ri[0];
                for (int q = 1; q < GJT / 64; q++) if (rv[q] > v || (rv[q] == v && ri[q] < i)) { v = rv[q]; i = ri[q]; }
                sp = i; sd = P[jj*N + i]; pan[i] = pb; piv[k0 + jj] = i; }
            __syncthreads();
            const int p = sp; const double d = sd;     // not P[jj, p]: the pivot thread overwrites it below
            if (t < N) {                      // one row per thread (N <= GJT)
                const int r = t;
                if (r == p) {
                    for (int j = 0; j < GJB; j++) { const double v = P[j*N + p]; P[j*N + r] = (j == jj) ? 1.0 / d : v / d; }
                }
            }
            // the other rows then use the scaled pivot row (P[jj,p] = 1/d)
            __syncthreads();
            if (t < N && t != p) {
                const int r = t; const double f = P[jj*N + r];
                for (int j = 0; j < GJB; j++) P[j*N + r] = (j == jj) ? -f / d : P[j*N + r] - f * P[j*N + p];
            }
            __syncthreads();
        }
        for (int x = t; x < GJB*N; x += GJT) { const int j = x / N, c = x % N;
            Ar[x] = (j < bw && (c < k0 || c >= k0 + bw)) ? A[piv[k0 + j] + (size_t)N*c] : 0.0; }
        __syncthreads();
        constexpr int NT = (N + 15) / 16;
        for (int tile = w; tile < (skip == 1 ? 0 : NT*NT); tile += GJT / 64) {
            const int c0 = 16 * (tile / NT), i0 = 16 * (tile % NT);
            if (c0 >= k0 && c0 < k0 + GJB) continue;              // the panel's own columns: written from P below
            d4 acc = (d4){0.0, 0.0, 0.0, 0.0};
#pragma unroll
            for (int s = 0; s < GJB / 4; s++) {
                const int j = 4*s + lk, c = c0 + lr, i = i0 + lr;
                const double a = c < N ? Ar[j*N + c] : 0.0, b = i < N ? P[j*N + i] : 0.0;
                acc = __builtin_amdgcn_mfma_f64_16x16x4f64(a, b, acc, 0, 0, 0);
            }
            const int i = i0 + lr;
            if (i < N) {
                const bool keep = pan[i] != pb;
#pragma unroll
                for (int r = 0; r < 4; r++) { const int c = c0 + 4*r + lk;
                    if (c < N) { double* x = &A[i + (size_t)N*c]; *x = (keep ? *x : 0.0) + acc[r]; } }
            }
        }
        for (int x = t; x < bw*N; x += GJT) { const int j = x / N, r = x % N; A[r + (size_t)N*(k0 + j)] = P[x]; }
    }
    __syncthreads();
    for (int k = t; k < N; k += GJT) pan[piv[k]] = (short)k;
    __syncthreads();
    for (int x = t; x < N*N; x += GJT) { const int i = x % N, j = x / N; C[x] = A[piv[i] + (size_t)N*pan[j]]; }
    __syncthreads();
    for (int x = t; x < N*N; x += GJT) A[x] = C[x];
}

// max over entries of |X A - I| for the first `count` matrices of a batch (X = inverse, A = the matrix before inversion)
extern "C" __global__ void __launch_bounds__(256) mkpc_invresid(const double* X, const double* A, int count, unsigned long long* out)
{
    const size_t e = blockIdx.x / N; const int col = blockIdx.x % N;
    if ((int)e >= count) return;
    const double* Xe = X + (size_t)N*N*e; const double* Ae = A + (size_t)N*N*e;
    double m = 0.0;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        double s = 0.0; for (int k = 0; k < N; k++) s += Xe[i + (size_t)N*k] * Ae[k + (size_t)N*col];
        m = fmax(m, fabs(s - (i == col ? 1.0 : 0.0)));
    }
    atomicMax(out, (unsigned long long)__double_as_longlong(m));     // non-negative doubles order as their bits
}

inline void gjinverse(dstype* A, dstype* C, Int n, Int batch)
{
    if (n != N) { printf("MKERR gjinverse: n=%d, built for %d\n", (int)n, N); return; }
    static const int T = [] { const char* e = getenv("MK_GJT"); return e ? atoi(e) : 1024; }();
    static const int Bw = [] { const char* e = getenv("MK_GJB"); return e ? atoi(e) : 16; }();
    static const int skip = [] { const char* e = getenv("MK_GJSKIP"); return e ? atoi(e) : 0; }();
    static const int pad = [] { const char* e = getenv("MK_GJPAD"); return e ? atoi(e) : 0; }();   // extra dynamic LDS (co-residency probe)
#define MKGJ(TT, BB) if (T == TT && Bw == BB) hipLaunchKernelGGL((mkpc_gjinv<TT, BB>), dim3(batch), dim3(TT), pad, mk_stream(), A, C, skip); else
    MKGJ(256, 16) MKGJ(512, 16) MKGJ(1024, 16) MKGJ(256, 32) MKGJ(512, 32) MKGJ(1024, 32)
    { printf("MKERR gjinverse: no instance for T=%d B=%d\n", T, Bw); return; }
#undef MKGJ
    MK_HIPCHECK("mkpc_gjinv");
}
inline double invresid(const dstype* X, const dstype* A, int count)
{
    static unsigned long long* d = nullptr; if (!d) hipMalloc(&d, sizeof(*d));
    hipMemset(d, 0, sizeof(*d));
    hipLaunchKernelGGL(mkpc_invresid, dim3(count*N), dim3(256), 0, mk_stream(), X, A, count, d);
    unsigned long long h = 0; hipMemcpy(&h, d, sizeof(h), hipMemcpyDeviceToHost);
    double v; std::memcpy(&v, &h, 8); return v;
}

// the generated stages as an Exasim PrecondStageTable (precondstages.hpp); sizes other than the generated ones fall back
#ifdef __PRECONDSTAGES
inline void t_schur(ResidualStageContext& c, Int jth) { schur(c.res, c.common, jth, true, true, c.handle, c.backend); }
inline void t_schur_prelaid(ResidualStageContext& c, Int jth) { schur(c.res, c.common, jth, true, true, c.handle, c.backend, true); }
inline void t_inverse(ResidualStageContext& c, dstype* A, Int n, Int batch) {
    if (n == N) gjinverse(A, c.res.H, n, batch);
    else Inverse(c.handle, A, c.res.H, c.res.ipiv, n, batch, c.backend);
    if (mkgm::use_f32()) mkgm::f32_slice((const double*)c.res.K, (const double*)A, (long)n * n, (long)batch, (long)c.common.meshsizes.ne1); }
inline const PrecondStageTable* table() {
    static PrecondStageTable t = [] { PrecondStageTable x; x.name = "metakernel-pc"; x.schur = t_schur; x.inverse = t_inverse;
        const char* e = getenv("MK_PC_ELEM"); if (!(e && e[0] == '0')) x.elem = mkjac::elem;     // MK_PC_ELEM=0: production elem
        const char* f = getenv("MK_PC_FACE"); if (!(f && f[0] == '0')) x.elemface = mkjac::elemface;   // MK_PC_FACE=0: production
        const char* ap = getenv("MK_PC_APPLY"); if (!(ap && ap[0] == '0')) x.apply = mkgm::apply;   // MK_PC_APPLY=0: production
        const char* cg = getenv("MK_PC_CGS"); if (!(cg && cg[0] == '0')) x.cgs = mkgm::cgs;         // MK_PC_CGS=0: production
        const char* cr = getenv("MK_PC_CROSS"); if (!(cr && cr[0] == '0')) x.cross = mkjac::cross;   // MK_PC_CROSS=0: production
        const char* g = getenv("MK_PC_FUSED");        // default: element + element-face fused (face terms in the GEMM epilogue)
        if (!(g && g[0] == '0') && x.elem && x.elemface) {     // MK_PC_FUSED=1: v1 (gathered face matrices); default v2
            const bool v1 = g && g[0] == '1';
            const char* h = getenv("MK_PC_SLAYOUT");   // default on with v2: D/F written in Schur layout, Schur skips its relayout
            const bool sl = !v1 && !(h && h[0] == '0');
            x.elemface = mkjac::elemface_noop;
            if (v1) x.elem = mkjac::elem_fused;
            else if (sl) { x.elem = mkjac::elem_fused2_sl; x.schur = t_schur_prelaid; }
            else x.elem = mkjac::elem_fused2_nosl; }
        return x; }();
    return &t; }
#endif
}  // namespace mkpc
