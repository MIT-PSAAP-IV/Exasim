// mk_gmstages.hpp -- GMRES-side replacements (precondstages.hpp v4 entries apply / cgs). Include after the Exasim unity TU.
//
//   mkgm::apply(ctx, x)  the LDG block-Jacobian apply x <- K_e x_e for every element, in place: one workgroup per element,
//                        x_e staged in LDS, K_e (n x n, column-major) streamed once with coalesced column reads.
//                        Production: ArrayCopy(Ru, x) + rocBLAS batched GEMV. Same products; the sum over columns is split
//                        into 4 interleaved partial sums (rounding level, like the other GEMM replacements).
//   mkgm::cgs(...)       the GMRES classical Gram-Schmidt step with the coefficients kept in device memory: the same hipBLAS
//                        gemv / gemv / dot as production (bitwise the same values), MPI_Allreduce on the device buffers
//                        (GPU-aware MPI), and ONE device->host copy of [h, |w|^2] instead of host-resident H read by the GPU.
#pragma once
#include <cmath>

namespace mkgm {
constexpr int GN = MK_NPE * MK_NCU;          // block size
constexpr int GT = 256;
static_assert(GN <= GT, "one row per thread");

extern "C" __global__ void __launch_bounds__(GT) mkgm_bgemv(const double* __restrict K, double* __restrict x)
{
    __shared__ double xs[GN];
    const int t = threadIdx.x; const size_t e = blockIdx.x;
    const double* Ke = K + (size_t)GN*GN*e; double* xe = x + (size_t)GN*e;
    if (t < GN) xs[t] = xe[t];
    __syncthreads();
    double y = 0.0;
    if (t < GN) {
        double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
        int j = 0;
#pragma unroll 4
        for (; j + 3 < GN; j += 4) {
            a0 += Ke[t + (size_t)GN*j] * xs[j];
            a1 += Ke[t + (size_t)GN*(j+1)] * xs[j+1];
            a2 += Ke[t + (size_t)GN*(j+2)] * xs[j+2];
            a3 += Ke[t + (size_t)GN*(j+3)] * xs[j+3];
        }
        for (; j < GN; j++) a0 += Ke[t + (size_t)GN*j] * xs[j];
        y = (a0 + a1) + (a2 + a3);
    }
    __syncthreads();
    if (t < GN) xe[t] = y;
}


// variants for the sweep (same products; partial-sum grouping differs): U = columns in flight per thread, NT = nontemporal
// loads of K (K is 2.9 GB per rank, far beyond the Infinity Cache, so it streams every apply), EPB = elements per workgroup
template <int U, bool NT, int EPB>
__global__ void __launch_bounds__(GT * EPB) mkgm_bgemv_v(const double* __restrict K, double* __restrict x, const int ne)
{
    __shared__ double xs[EPB][GN];
    const int t = threadIdx.x % GT, w = threadIdx.x / GT; const size_t e = (size_t)blockIdx.x * EPB + w;
    const bool live = e < (size_t)ne;
    const double* Ke = K + (size_t)GN*GN*e; double* xe = x + (size_t)GN*e;
    if (live && t < GN) xs[w][t] = xe[t];
    __syncthreads();
    double y = 0.0;
    if (live && t < GN) {
        double acc[4] = {0.0, 0.0, 0.0, 0.0};
        int j = 0;
        for (; j + U <= GN; j += U) {
            double kv[U];
#pragma unroll
            for (int u = 0; u < U; u++) kv[u] = NT ? __builtin_nontemporal_load(&Ke[t + (size_t)GN*(j+u)]) : Ke[t + (size_t)GN*(j+u)];
#pragma unroll
            for (int u = 0; u < U; u++) acc[u % 4] += kv[u] * xs[w][j+u];
        }
        for (; j < GN; j++) acc[0] += Ke[t + (size_t)GN*j] * xs[w][j];
        y = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    }
    __syncthreads();
    if (live && t < GN) xe[t] = y;
}
// fp32 copy of the block inverses (MK_PC_F32=1): the apply streams half the bytes; products and sums stay fp64.
// This changes the preconditioner (not the residual): GMRES still converges to the same tolerance.
inline bool use_f32() { static const bool v = [] { const char* e = getenv("MK_PC_F32"); return e && e[0] == '1'; }(); return v; }
struct KF { float* p = nullptr; size_t cap = 0; const void* src = nullptr; bool ok = false; long done = 0; };
inline KF& kf() { static KF k; return k; }
__global__ void mkgm_tof32(const double* __restrict K, float* __restrict F, const size_t n) {
    for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < n; i += (size_t)gridDim.x * 256) F[i] = (float)K[i]; }
// the inverse runs in element-block slices: convert each slice at its offset; the copy is used once a build covers all
inline void f32_slice(const double* K, const double* A, long n2, long batch, long ne) {
    KF& k = kf(); const size_t tot = (size_t)n2 * ne;
    if (tot > k.cap) { if (k.p) hipFree(k.p); hipMalloc(&k.p, tot * sizeof(float)); k.cap = tot; }
    const long off = (long)(A - K);
    if (off == 0 || k.src != (const void*)K) { k.done = 0; k.src = K; }   // a new build starts at the first slice
    k.ok = false;
    if (off < 0 || off % n2 != 0 || off / n2 + batch > ne) return;
    hipLaunchKernelGGL(mkgm_tof32, dim3(4096), dim3(256), 0, mk_stream(), A, k.p + off, (size_t)n2 * batch); MK_HIPCHECK("mkgm_tof32");
    k.done += batch; k.ok = (k.done == ne); }
extern "C" __global__ void __launch_bounds__(GT) mkgm_bgemv_f32(const float* __restrict K, double* __restrict x)
{
    __shared__ double xs[GN];
    const int t = threadIdx.x; const size_t e = blockIdx.x;
    const float* Ke = K + (size_t)GN*GN*e; double* xe = x + (size_t)GN*e;
    if (t < GN) xs[t] = xe[t];
    __syncthreads();
    double y = 0.0;
    if (t < GN) {
        double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
        int j = 0;
#pragma unroll 4
        for (; j + 3 < GN; j += 4) {
            a0 += (double)Ke[t + (size_t)GN*j] * xs[j];
            a1 += (double)Ke[t + (size_t)GN*(j+1)] * xs[j+1];
            a2 += (double)Ke[t + (size_t)GN*(j+2)] * xs[j+2];
            a3 += (double)Ke[t + (size_t)GN*(j+3)] * xs[j+3];
        }
        for (; j < GN; j++) a0 += (double)Ke[t + (size_t)GN*j] * xs[j];
        y = (a0 + a1) + (a2 + a3);
    }
    __syncthreads();
    if (t < GN) xe[t] = y;
}
inline int apply_variant() { static const int v = [] { const char* e = getenv("MK_GMV"); return e ? atoi(e) : 0; }(); return v; }
inline void apply_v(int v, const double* K, double* x, int ne)
{
    switch (v) {
    case 1: hipLaunchKernelGGL((mkgm_bgemv_v<8, false, 1>), dim3(ne), dim3(GT), 0, mk_stream(), K, x, ne); break;
    case 2: hipLaunchKernelGGL((mkgm_bgemv_v<16, false, 1>), dim3(ne), dim3(GT), 0, mk_stream(), K, x, ne); break;
    case 3: hipLaunchKernelGGL((mkgm_bgemv_v<8, true, 1>), dim3(ne), dim3(GT), 0, mk_stream(), K, x, ne); break;
    case 4: hipLaunchKernelGGL((mkgm_bgemv_v<16, true, 1>), dim3(ne), dim3(GT), 0, mk_stream(), K, x, ne); break;
    case 5: hipLaunchKernelGGL((mkgm_bgemv_v<8, true, 2>), dim3((ne + 1)/2), dim3(2*GT), 0, mk_stream(), K, x, ne); break;
    case 6: hipLaunchKernelGGL((mkgm_bgemv_v<16, true, 2>), dim3((ne + 1)/2), dim3(2*GT), 0, mk_stream(), K, x, ne); break;
    default: hipLaunchKernelGGL(mkgm_bgemv, dim3(ne), dim3(GT), 0, mk_stream(), K, x); break;
    }
    MK_HIPCHECK("mkgm_bgemv_v");
}

inline void apply(ResidualStageContext& c, dstype* x)
{
    const Int n = c.common.grid.npe*c.common.components.ncu, ne = c.common.meshsizes.ne1;
    if (n != GN) {     // not the generated size: production
        ArrayCopy(c.common.cublasHandle, c.res.Ru, x, c.common.sizes.ndof1, c.backend);
        PGEMNMStridedBached(c.common.cublasHandle, n, 1, n, one, c.res.K, n, c.res.Ru, n, zero, x, n, ne, c.backend);
        return; }
    if (use_f32()) { static int nf = 0, n64 = 0; const bool f = kf().ok && kf().src == (const void*)c.res.K; (f ? nf : n64)++;
        if (nf + n64 == 200) printf("MKINFO MK_PC_F32: %d of the first 200 applies used the fp32 blocks\n", nf); }
    if (use_f32() && kf().ok && kf().src == (const void*)c.res.K) {
        hipLaunchKernelGGL(mkgm_bgemv_f32, dim3(ne), dim3(GT), 0, mk_stream(), (const float*)kf().p, x); MK_HIPCHECK("mkgm_bgemv_f32"); return; }
    apply_v(apply_variant(), (const double*)c.res.K, x, (int)ne);
}

inline double* dbuf(Int n)
{
    static double* p = nullptr; static Int cap = 0;
    if (n > cap) { if (p) hipFree(p); hipMalloc(&p, n*sizeof(double)); cap = n; }
    return p;
}

inline void cgs(cublasHandle_t handle, dstype* V, dstype* H, dstype* temp, Int N, Int m, Int backend)
{
    (void)temp; (void)backend;
    double* dh = dbuf(m + 1);
    hipblasDgemv(handle, HIPBLAS_OP_T, N, m, &one, V, N, &V[m*N], inc1, &zero, dh, inc1);          // h = V^T w
#ifdef HAVE_MPI
    hipStreamSynchronize(mk_stream());
    MPI_Allreduce(MPI_IN_PLACE, dh, m, mpi_type<dstype>(), MPI_SUM, EXASIM_COMM_WORLD);
#endif
    hipblasDgemv(handle, HIPBLAS_OP_N, N, m, &minusone, V, N, dh, inc1, &one, &V[m*N], inc1);      // w -= V h
    hipblasPointerMode_t pm; hipblasGetPointerMode(handle, &pm);
    hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);
    hipblasDdot(handle, N, &V[m*N], inc1, &V[m*N], inc1, &dh[m]);                                  // |w|^2
    hipblasSetPointerMode(handle, pm);
#ifdef HAVE_MPI
    hipStreamSynchronize(mk_stream());
    MPI_Allreduce(MPI_IN_PLACE, &dh[m], 1, mpi_type<dstype>(), MPI_SUM, EXASIM_COMM_WORLD);
#endif
    hipMemcpy(H, dh, (m + 1)*sizeof(double), hipMemcpyDeviceToHost);
    H[m] = sqrt(H[m]);
    ArrayMultiplyScalar(&V[m*N], one/H[m], N);
}
}  // namespace mkgm
