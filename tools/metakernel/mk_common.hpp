// meta-kernel common: case sizes, the argument block every generated kernel gets, laundering, the GEMM stages
#pragma once
#include <map>
// The case's sizes come from mk_case_sizes.hpp on the include path (next to the case's generated kernels):
//   constexpr int MK_NC, MK_NCW, MK_NCO, MK_NCU, MK_ND, MK_NCX, MK_NPE, MK_NGE, MK_NPR, MK_NPF, MK_NGF;
// (nc, ncw, nco, ncu, nd, ncx, npe, nge, property-stage width, npf, ngf). The harness checks them against the
// Exasim case at run time and falls back to production on a mismatch.
#if __has_include("mk_case_sizes.hpp")
#include "mk_case_sizes.hpp"
#else
#error "metakernel: put the case's mk_case_sizes.hpp on the include path (see tools/metakernel/README.md)"
#endif
constexpr int MK_NF = MK_NCU * MK_ND, MK_NRG = MK_NCU * (MK_ND + 1);
static_assert(MK_NPF == MK_NGF, "face kernels assume npf == ngf");

struct MkIn {
    const dstype *x, *o, *udg, *wdg, *sdg, *jac, *Xx, *gt, *gw, *uinf, *par;   // inputs (x, o, sdg, jac, Xx at gauss points)
    dstype *g_u, *g_w, *g_s, *g_sg, *g_pr, *g_f, *g_rg, *R;                   // stage buffers [comp*ng + i]; rg in GEMM layout
    dstype *un, *wn; const int* eind;                                           // gather scratch for the GEMM interp
    const dstype* udg_abs;                                                      // udg base the (absolute) eind indices refer to
    dstype time, dtf; int ng, ne;
    // producer-side face values (MK_FACE_PRODUCER): per (global element, face slot) the destination of this element's side
    // of an interior face in the face stage's interpolated-column buffer (nullptr = no face), its stride, and the 9 face
    // nodes as element-local node numbers in the face's own order; pe1 = this block's first global element
    const dstype* fgt = nullptr; dstype* const* pbase = nullptr; const int* pnga = nullptr; const int* pnode = nullptr; int pe1 = 0;
    const int* pinv = nullptr;   // per (element, slot, element node p): the face node q at p, or -1 (MFMA producer)
};
// face block arguments (one production fblks block): geometry/og at face Gauss points are block-local [i + nga*c]
struct MkFaceIn {
    const dstype *xg, *nl, *jac, *og1, *og2;          // faceg / og1 / og2 at this block's offsets
    const dstype *udg, *wdg, *uh;                     // global nodal arrays
    const int *find1, *find2, *facecon;               // findudg1/2 at npf*nc*f1 (absolute udg indices), facecon (global)
    const dstype *gt, *gw, *uinf, *par, *tau;         // shapfgt, shapfgw, model parameters, stabilization
    dstype *Rh, *gf1, *gug;                           // Rh at npf*ncu*f1; per-point F1 buffer; interpolated face columns (split)
    dstype *gat;                                      // gathered face-node columns (split, interp="gemm")
    dstype time; int f1, nf, ib;
};
// q-path element block arguments (generated GetQ minus RqFace)
constexpr int MK_NCQ = MK_NCU * MK_ND;
struct MkQIn {
    dstype* udg; const dstype *Xx, *gt, *gw, *Rh, *Minv;   // Xx at this block's elemg offset (block-local [g + nge el + nga (m + nd j)])
    const int *nodeptr, *contrib;                          // ordered face contributions per element node (production's gather order)
    const int *efaces, *lcontrib;                          // per element: its <= 6 faces (-1 pad); contrib re-coded as (slot*npf + q)<<1 | side
    int e1, neb;
    int curved;   // 1: per-element Minv (ArrayGemmBatch1); 0: R /= jac0_e (ApplyJacInv), then master Minv0 (ArrayMatrixMultiplication1)
    // q v7 (face integrals in-kernel): uh, face shape functions, face geometry; per element its faces in ascending id
    // (qfs, -1 pad) and per (face slot, element node) the face-node index and side ((q << 1) | side, -1 = not on it);
    // per face its offset into faceg and its block's ngf * nf stride
    const dstype *uh = nullptr, *fgt = nullptr, *fgw = nullptr, *faceg = nullptr;
    const int *qfs = nullptr, *qcode = nullptr; const size_t* fofs = nullptr; const int* fnga = nullptr;
};
// opaque copies of every base pointer / size: no address arithmetic can be hoisted or shared across this point
__device__ __forceinline__ MkIn mk_launder(const MkIn& a) {
    MkIn b = a;
    asm volatile("" : "+s"(b.x), "+s"(b.o), "+s"(b.udg), "+s"(b.wdg), "+s"(b.sdg), "+s"(b.jac), "+s"(b.Xx), "+s"(b.gt), "+s"(b.gw), "+s"(b.uinf), "+s"(b.par));
    asm volatile("" : "+s"(b.g_u), "+s"(b.g_w), "+s"(b.g_s), "+s"(b.g_sg), "+s"(b.g_pr), "+s"(b.g_f), "+s"(b.g_rg), "+s"(b.R), "+s"(b.ng), "+s"(b.ne));
    return b;
}
#define MK_CFENCE() asm volatile("" ::: "memory")
// plain-HIP kernels launch on the Kokkos instance stream, so they order with the Kokkos kernels and the MFMA GEMMs
// mk_cur(): the stream the current block's chain runs on (set by mkstages::fanout); nullptr = the Kokkos instance stream
inline hipStream_t& mk_cur() { static hipStream_t s = nullptr; return s; }
static hipStream_t mk_stream() { return mk_cur() ? mk_cur() : Kokkos::HIP().hip_stream(); }
// Kokkos execution instance on mk_stream() (cached per stream, never destroyed: they must outlive Kokkos::finalize order)
inline Kokkos::HIP mk_exec() {
    if (!mk_cur()) return Kokkos::HIP();
    static auto* m = new std::map<hipStream_t, Kokkos::HIP>();
    auto it = m->find(mk_cur()); if (it == m->end()) it = m->emplace(mk_cur(), Kokkos::HIP(mk_cur())).first;
    return it->second; }
// the MFMA GEMM (Exasim's v2 kernel) launched on mk_stream(); exasim_mfma::gemm_nn always uses its own compute stream
static void mk_gemm_nn(double* C, const double* A, const double* B, int M, int K, int N, int ldc) {
    int MP; const double* Ap = exasim_mfma::padded_operator(A, M, K, &MP);
    const int ks = (K + 3) / 4; const dim3 g((N + 15) / 16), b(64);
    if (ks <= 3) hipLaunchKernelGGL(exasim_mfma::mfma_gemm_nn_f64_v2<3>, g, b, 0, mk_stream(), C, Ap, B, M, K, N, MP, ldc);
    else if (ks <= 7) hipLaunchKernelGGL(exasim_mfma::mfma_gemm_nn_f64_v2<7>, g, b, 0, mk_stream(), C, Ap, B, M, K, N, MP, ldc);
    else hipLaunchKernelGGL(exasim_mfma::mfma_gemm_nn_f64_v2<exasim_mfma::V2_KSMAX>, g, b, 0, mk_stream(), C, Ap, B, M, K, N, MP, ldc); }
#define MK_HIPCHECK(name) do { hipError_t mk_e_ = hipGetLastError(); if (mk_e_ != hipSuccess) { \
    printf("MKERR launch %s: %s\n", name, hipGetErrorString(mk_e_)); std::abort(); } } while (0)

// production-form GEMM stages (gather + MFMA interpolation; MFMA integration)
static void mk_gemm_interp(const MkIn& a) {
    const int ne = a.ne; const int* pe = a.eind; const dstype *pu = a.udg_abs, *pw = a.wdg; dstype *pun = a.un, *pwn = a.wn;
    Kokkos::parallel_for("gather", Kokkos::RangePolicy<Kokkos::HIP>(mk_exec(), 0, (size_t)MK_NPE * MK_NC * ne), KOKKOS_LAMBDA(const size_t i) { pun[i] = pu[pe[i]]; });
    mk_gemm_nn(a.g_u, a.gt, pun, MK_NGE, MK_NPE, ne * MK_NC, MK_NGE);
    Kokkos::parallel_for("gatherw", Kokkos::RangePolicy<Kokkos::HIP>(mk_exec(), 0, (size_t)MK_NPE * MK_NCW * ne), KOKKOS_LAMBDA(const size_t idx) {
        const int p = idx % MK_NPE; const size_t r = idx / MK_NPE; const int e = r % ne, k = r / ne; pwn[idx] = pw[p + MK_NPE * k + (size_t)MK_NPE * MK_NCW * e]; });
    mk_gemm_nn(a.g_w, a.gt, pwn, MK_NGE, MK_NPE, ne * MK_NCW, MK_NGE);
}
static void mk_gemm_integrate(const MkIn& a) {
    mk_gemm_nn(a.R, a.gw, a.g_rg, MK_NPE, MK_NGE * (MK_ND + 1), MK_NCU * a.ne, MK_NPE);
}
