// mk_jacstages.hpp -- generated replacements for the LDG block-Jacobian element / element-face stages
// (precondstages.hpp entries elem / elemface). Include after the Exasim unity TU and mk_common.hpp.
//
//   mkjac::elem(pctx, j)   uEquationElemBlock for element block j. The pointwise part (gather, interp, w Newton, Source,
//                          Tdfunc, Flux, w chain, ApplyXxJac) is production's calls verbatim, minus the dead Ru path;
//                          Gauss2Node + (*= -1) for D and B is one kernel (mkjac_negmm): one wave per 16 columns of rg,
//                          the rg fragment held in registers across all 46 row tiles of the 729 x 108 operator, -acc
//                          stored directly (production: one wave per 16x16 tile re-reading rg, then a full RMW for -1).
//                          The product is formed transposed (C^T = rg^T S^T) so the stores are coalesced; each entry is
//                          the same MFMA f64 16x16x4 chain over the same k groups, so it is bitwise production's.
#pragma once
#include <map>
#include <vector>

namespace mkjac {
// optional section timers for the face stage (MK_JACT=1): 0 gather+interp, 1 interior w Newton, 2 flux..interior GEMMs,
// 3 (unused), 4 boundary non-w work, 5 boundary w Newton, 6 assembly
inline bool jt_on() { static const bool on = [] { const char* e = getenv("MK_JACT"); return e && e[0] == '1'; }(); return on; }
inline double* jt() { static double t[24] = {0}; return t; }   // [0,8) face stage, [8,16) element stage
#define MKJT_START() Kokkos::Timer mkjt_; if (mkjac::jt_on()) { Kokkos::fence(); hipDeviceSynchronize(); mkjt_.reset(); }
#define MKJT(k) if (mkjac::jt_on()) { Kokkos::fence(); hipDeviceSynchronize(); mkjac::jt()[k] += mkjt_.seconds() * 1e3; mkjt_.reset(); }
typedef double d4 __attribute__((ext_vector_type(4)));
constexpr int KG = MK_NGE * (MK_ND + 1);        // 108: flattened (gauss point, value/derivative)
constexpr int KS = KG / 4;                      // 27 MFMA k-steps
static_assert(KG % 4 == 0, "K must be a multiple of 4 for the register-resident rg fragment");

// C[m + ldc*n] = -(sum_k S[m + MP*k] rg[k + KG*n]),  m < M (= npe^2), n < N;  one wave per 16 columns.
// With Ftm != nullptr (fused face assembly): C = (-acc) + face terms, added in ascending face order from the FaceTab
// (cnt/ent) -- production's D = -elem, then D += Dtmp per face (atomic, arbitrary order), in one store.
extern "C" __global__ void __launch_bounds__(64) mkjac_negmm(double* __restrict C, const double* __restrict Spad,
        const double* __restrict rg, const int M, const int N, const int MP, const int ldc,
        const double* __restrict Ftm, const int* __restrict cnt, const int* __restrict ent, const int fstride)
{
    const int l = threadIdx.x, lr = l % 16, lk = l / 16, n0 = 16 * blockIdx.x, nc = n0 + lr;
    double a[KS];                                   // rg^T fragment: row n = n0 + l%16, k = 4s + l/16
#pragma unroll
    for (int s = 0; s < KS; s++) a[s] = nc < N ? rg[(4*s + lk) + (size_t)KG * nc] : 0.0;
    for (int m0 = 0; m0 < M; m0 += 16) {
        d4 acc = (d4){0.0, 0.0, 0.0, 0.0};
#pragma unroll
        for (int s = 0; s < KS; s++) acc = __builtin_amdgcn_mfma_f64_16x16x4f64(a[s], Spad[m0 + lr + (size_t)MP * (4*s + lk)], acc, 0, 0, 0);
        const int m = m0 + lr;                      // D^T layout: row (n) 4r + l/16, col (m) l%16
        if (m < M) {
            const int c = Ftm ? cnt[m] : 0;
#pragma unroll
            for (int r = 0; r < 4; r++) { const int n = n0 + 4*r + lk;
                if (n < N) { double x = -acc[r];
                    for (int q = 0; q < c; q++) x += Ftm[ent[4*m + q] + (size_t)fstride * n];
                    C[m + (size_t)ldc * n] = x; } }
        }
    }
}

inline void negmm(double* C, const double* S, const double* rg, int M, int K, int N,
                  const double* Ftm = nullptr, const int* cnt = nullptr, const int* ent = nullptr, int fstride = 0)
{
    int MP; const double* Spad = exasim_mfma::padded_operator(S, M, K, &MP);
    hipLaunchKernelGGL(mkjac_negmm, dim3((N + 15) / 16), dim3(64), 0, mk_stream(), C, Spad, rg, M, N, MP, M, Ftm, cnt, ent, fstride);
    MK_HIPCHECK("mkjac_negmm");
}

// uEquationElemBlock<AbiAdapter> with the replacements above (production line numbers: backend/Discretization/uequation.hpp)

// Face GEMM in the element epilogue. Pf_l[tp, g] = S_f[(a + npf b), g] when t = perm[a + npf l] and p = perm[b + npf l]
// (0 otherwise), padded to 736 x 12; mask[tp] = the faces holding (t, p). The per-face product Pf_l . FRG_l is the same
// MFMA 16x16x4 chain (k groups {0-3},{4-7},{8, pad}) on the same operands as production's face GEMM -> Dtmp, so each face
// term is production's value; they are added to -elem in ascending l, as the fused gather did.
struct FaceOp { double* Pf = nullptr; int* mask = nullptr; int* tmask = nullptr; };   // tmask[m0/16]: faces touching a 16-row tile
constexpr int PM = 736, PK = 12;
inline FaceOp& faceop(const int* perm_dev, const double* Sf_dev, int npe, int npf, int nfe, int ngf)
{
    static std::map<std::pair<const int*, const double*>, FaceOp> cache;
    auto key = std::make_pair(perm_dev, Sf_dev); auto it = cache.find(key); if (it != cache.end()) return it->second;
    const int ndf = npf*nfe; std::vector<int> perm(ndf); std::vector<double> Sf(npf*npf*ngf);
    hipMemcpy(perm.data(), perm_dev, ndf*sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(Sf.data(), Sf_dev, Sf.size()*sizeof(double), hipMemcpyDeviceToHost);
    std::vector<double> Pf((size_t)nfe*PM*PK, 0.0); std::vector<int> mask(npe*npe, 0);
    for (int l = 0; l < nfe; l++) for (int a = 0; a < npf; a++) for (int b = 0; b < npf; b++) {
        const int tp = perm[a + npf*l] + npe*perm[b + npf*l]; mask[tp] |= 1 << l;
        for (int g = 0; g < ngf; g++) Pf[(size_t)l*PM*PK + tp + PM*g] = Sf[(a + npf*b) + npf*npf*g]; }
    std::vector<int> tmask(PM/16, 0); for (int tp = 0; tp < npe*npe; tp++) tmask[tp/16] |= mask[tp];
    FaceOp op; hipMalloc(&op.Pf, Pf.size()*sizeof(double)); hipMalloc(&op.mask, mask.size()*sizeof(int)); hipMalloc(&op.tmask, tmask.size()*sizeof(int));
    hipMemcpy(op.tmask, tmask.data(), tmask.size()*sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(op.Pf, Pf.data(), Pf.size()*sizeof(double), hipMemcpyHostToDevice); hipMemcpy(op.mask, mask.data(), mask.size()*sizeof(int), hipMemcpyHostToDevice);
    return cache.emplace(key, op).first->second;
}
constexpr int NFE = 6, NGF = MK_NGF;
static_assert(NGF <= PK, "face K padding");
// C[m + ldc n] = (-(S . rg)[m, n]) + sum_{l in mask[m], ascending} (Pf_l . FRG_l)[m, n]
// column n = el + ne*(mm + ncu*nn); FRG column = mm + ncu*(coff + nn)  (coff = 0: D / u-columns, ncu: B / q-columns)
template <bool SL>
__global__ void __launch_bounds__(64) mkjac_negmm_face(double* __restrict C, const double* __restrict Spad,
        const double* __restrict rg, const int M, const int N, const int MP, const int ldc,
        const double* __restrict FRG, const double* __restrict Pf, const int* __restrict mask, const int* __restrict tmask,
        const int ne, const int ncu, const int coff, const long long nga)
{
    const int l = threadIdx.x, lr = l % 16, lk = l / 16, n0 = 16 * blockIdx.x, nc = n0 + lr;
    double a[KS];
#pragma unroll
    for (int s = 0; s < KS; s++) a[s] = nc < N ? rg[(4*s + lk) + (size_t)KG * nc] : 0.0;
    double bf[NFE][PK / 4];                         // FRG^T fragments: row n = n0 + l%16, k = g = 4s + l/16
    {   const int el = nc % ne, mn = nc / ne, mm = mn % ncu, nn = mn / ncu;
        const size_t fc = (size_t)nga * (mm + ncu*(coff + nn));
#pragma unroll
        for (int f = 0; f < NFE; f++)
#pragma unroll
            for (int s = 0; s < PK / 4; s++) { const int g = 4*s + lk;
                bf[f][s] = (nc < N && g < NGF) ? FRG[g + NGF*(f + NFE*(size_t)el) + fc] : 0.0; } }
    for (int m0 = 0; m0 < M; m0 += 16) {
        d4 acc = (d4){0.0, 0.0, 0.0, 0.0};
#pragma unroll
        for (int s = 0; s < KS; s++) acc = __builtin_amdgcn_mfma_f64_16x16x4f64(a[s], Spad[m0 + lr + (size_t)MP * (4*s + lk)], acc, 0, 0, 0);
        d4 af[NFE];
#pragma unroll
        for (int f = 0; f < NFE; f++) { af[f] = (d4){0.0, 0.0, 0.0, 0.0};
#pragma unroll
            for (int s = 0; s < PK / 4; s++)
                af[f] = __builtin_amdgcn_mfma_f64_16x16x4f64(bf[f][s], Pf[(size_t)f*PM*PK + m0 + lr + PM*(4*s + lk)], af[f], 0, 0, 0); }
        const int m = m0 + lr;
        if (m < M) {
            const int msk = mask[m];
            // Schur layout (SL): D[(t + npe mm) + nn_ (p + npe nn) + nn_^2 el], m = t + npe p, n = el + ne (mm + ncu nn)
            const int tt = m % MK_NPE, pp = m / MK_NPE; constexpr int NN = MK_NPE * MK_NCU;
#pragma unroll
            for (int r = 0; r < 4; r++) { const int n = n0 + 4*r + lk;
                if (n < N) { double x = -acc[r];
#pragma unroll
                    for (int f = 0; f < NFE; f++) if (msk & (1 << f)) x += af[f][r];
                    if (SL) { const int el = n % ne, mn = n / ne, mm = mn % ncu, nn = mn / ncu;
                        C[(size_t)(tt + MK_NPE*mm) + (size_t)NN*(pp + MK_NPE*nn) + (size_t)NN*NN*el] = x; }
                    else C[m + (size_t)ldc * n] = x; } }
        }
    }
}
inline void negmm_face(double* C, const double* S, const double* rg, int M, int K, int N, const FaceOp& op, const double* FRG,
                       int ne, int ncu, int coff, long long nga, bool schur_layout = false)
{
    int MP; const double* Spad = exasim_mfma::padded_operator(S, M, K, &MP);
    if (schur_layout) hipLaunchKernelGGL(mkjac_negmm_face<true>, dim3((N + 15) / 16), dim3(64), 0, mk_stream(), C, Spad, rg, M, N, MP, M, FRG, op.Pf, op.mask, op.tmask, ne, ncu, coff, nga);
    else hipLaunchKernelGGL(mkjac_negmm_face<false>, dim3((N + 15) / 16), dim3(64), 0, mk_stream(), C, Spad, rg, M, N, MP, M, FRG, op.Pf, op.mask, op.tmask, ne, ncu, coff, nga);
    MK_HIPCHECK("mkjac_negmm_face");
}
struct FaceTab;
inline const int* ft_cnt(const FaceTab* f);
inline const int* ft_ent(const FaceTab* f);
inline bool elem_ok(const commonstruct& common) { return common.grid.nge*(common.grid.nd+1) == KG && common.timeparams.wave == 0; }
inline void elem_impl(PrecondStageContext& c, Int jth, const FaceTab* ft, const FaceOp* fo = nullptr, bool sl = false);
inline void elem(PrecondStageContext& c, Int jth) { elem_impl(c, jth, nullptr); }
inline void elem_impl(PrecondStageContext& c, Int jth, const FaceTab* ft, const FaceOp* fo, bool sl)
{
    using M = exasim::detail::AbiAdapter;
    auto& sol = c.sol; auto& res = c.res; auto& app = c.app; auto& master = c.master; auto& mesh = c.mesh; auto& tmp = c.tmp;
    auto& common = c.common; auto handle = c.handle; const Int backend = c.backend;
    const Int nc = common.components.nc, ncu = common.components.ncu, ncq = common.components.ncq, nco = common.components.nco,
              ncx = common.components.ncx, ncw = common.components.ncw, nd = common.grid.nd, npe = common.grid.npe, nge = common.grid.nge;
    if (nge*(nd+1) != KG || common.timeparams.wave != 0) {      // sizes/flags this stage was generated for
        uEquationElemBlock<M>(sol, res, app, master, mesh, tmp, common, handle, jth, backend); return; }
    const Int e1 = common.eblks[3*jth]-1, e2 = common.eblks[3*jth+1], ne = e2-e1, nga = nge*ne;
    const Int n1 = nga*ncx, n2 = nga*(ncx+nd*nd), n3 = 0, n4 = nga*ncu*nd, n5 = n4 + nga*ncu, n6 = n5 + nga*nc,
              n7 = n6 + nga*ncw, n8 = n7 + nga*ncu*nd*nc, n9 = n8 + nga*ncu*nd*ncw, n10 = n9 + nga*ncu*nc, n11 = n10 + nga*ncu*ncw;
    const Int nm = nge*e1*(ncx+nd*nd+1);
    dstype *xg = &sol.elemg[nm], *Xx = &sol.elemg[nm+n1], *jac = &sol.elemg[nm+n2], *og = &sol.odgg[nge*nco*e1];
    dstype *wsrc = &tmp.tempg[n3], *fg = &tmp.tempg[n3], *sg = &tmp.tempg[n4], *uqg = &tmp.tempg[n5], *wg = &tmp.tempg[n6];
    dstype *fg_uq = &tmp.tempg[n7], *fg_w = &tmp.tempg[n8], *sg_uq = &tmp.tempg[n9], *sg_w = &tmp.tempg[n10], *wg_uq = &tmp.tempg[n11];

    MKJT_START();
    GetElemNodes(tmp.tempn, sol.udg, npe, nc, 0, nc, e1, e2);                                            // :106
    Node2Gauss(handle, uqg, tmp.tempn, master.shapegt, nge, npe, ne*nc, backend);
    if (ncw > 0) {                                                                                         // :110
        const Int ncwa = ncw - app.materialdb_nprop;
        GetElemNodes(tmp.tempn, sol.wdg, npe, ncw, 0, ncw, e1, e2);
        Node2Gauss(handle, wg, tmp.tempn, master.shapegt, nge, npe, ne*ncw, backend);
        if (ncwa > 0) {
            GetElemNodes(tmp.tempn, sol.wsrc, npe, ncw, 0, ncwa, e1, e2);
            Node2Gauss(handle, wsrc, tmp.tempn, master.shapegt, nge, npe, ne*ncwa, backend);
        }
        wEquation<M>(wg, wg_uq, xg, uqg, og, wsrc, tmp.tempn, app, common, nga, backend, tmp.tempi);
    }
    MKJT(8);
    ArraySetValue(sg, 0.0, nga*ncu);                                                                       // :131
    ArraySetValue(sg_uq, 0.0, nga*ncu*nc);
    if (ncw > 0) ArraySetValue(sg_w, 0.0, nga*ncu*ncw);
    SourceDriver(sg, sg_uq, sg_w, xg, uqg, og, wg, mesh, master, app, sol, tmp, common, nge, e1, e2, backend);
    if (common.timeparams.tdep) {                         // only the Jacobian part: sg / fg (Ru) are dead here (:138,148,151)
        if (common.timeparams.tdfunc == 1)
            TdfuncDriver(fg_uq, xg, uqg, og, wg, mesh, master, app, sol, tmp, common, nge, e1, e2, backend);
        else
            ArraySetValue(fg_uq, one, nga*ncu);
        ApplyDtcoef(sg_uq, fg_uq, -common.timestate.dtfactor, nga, ncu);                                  // :154
    }
    MKJT(9);
    ArraySetValue(fg, 0.0, nga*ncu*nd);                                                                    // :164
    ArraySetValue(fg_uq, 0.0, nga*ncu*nd*nc);
    if (ncw > 0) ArraySetValue(fg_w, 0.0, nga*ncu*nd*ncw);
    FluxDriver(fg, fg_uq, fg_w, xg, uqg, og, wg, mesh, master, app, sol, tmp, common, nge, e1, e2, backend);
    if (ncw > 0) {
        ArrayGemmBatch2(sg_uq, sg_w, wg_uq, 1.0, ncu, nc, ncw, nga);                                     // :171
        ArrayGemmBatch2(fg_uq, fg_w, wg_uq, 1.0, ncu*nd, nc, ncw, nga);
    }
    MKJT(10);
    // D, B = -(shapegwdotshapeg . rg), fused (production :181-189)
    ApplyXxJac(tmp.tempn, sg_uq, fg_uq, Xx, jac, nge, nd, ncu, ncu, ne);
    const Int npf = common.grid.npf, nfe = common.meshsizes.nfe, nf = nfe*ne;    // face terms (res.H: Dtmp, Btmp), if fused
    const double* Dtmp = ft ? &res.H[0] : nullptr; const double* Btmp = ft ? &res.H[npf*npf*nf*ncu*ncu] : nullptr;
    const int* fcnt = ft ? ft_cnt(ft) : nullptr; const int* fent = ft ? ft_ent(ft) : nullptr;
    const long long ngaf = (long long)common.grid.ngf*nf;
    if (fo) negmm_face(res.D, master.shapegwdotshapeg, tmp.tempn, npe*npe, nge*(nd+1), ncu*ncu*ne, *fo, res.H, ne, ncu, 0, ngaf, sl);
    else negmm(res.D, master.shapegwdotshapeg, tmp.tempn, npe*npe, nge*(nd+1), ncu*ncu*ne, Dtmp, fcnt, fent, npf*npf*nfe);
    MKJT(11);
    if (ncq > 0) {
        ApplyXxJac(tmp.tempn, &sg_uq[nga*ncu*ncu], &fg_uq[nga*ncu*nd*ncu], Xx, jac, nge, nd, ncu, ncq, ne);
        if (fo) negmm_face(res.B, master.shapegwdotshapeg, tmp.tempn, npe*npe, nge*(nd+1), ncu*ncq*ne, *fo, res.H, ne, ncu, ncu, ngaf);
        else negmm(res.B, master.shapegwdotshapeg, tmp.tempn, npe*npe, nge*(nd+1), ncu*ncq*ne, Btmp, fcnt, fent, npf*npf*nfe);
        MKJT(12);
    }
}

// ---- element-face stage: production (ldgblockjacobian.cpp uEquationElemFaceBlockLDG :905-1075) verbatim, then the
// D/B assembly as a gather in fixed face order (no atomics: deterministic) and F written whole in one pass.
struct FaceTab { int* cnt = nullptr; int* ent = nullptr; int* inv = nullptr; };   // device tables, built once per perm
inline FaceTab& facetab(const int* perm_dev, int npe, int npf, int nfe)
{
    static std::map<const int*, FaceTab> cache;
    auto it = cache.find(perm_dev); if (it != cache.end()) return it->second;
    const int ndf = npf*nfe; std::vector<int> perm(ndf); hipMemcpy(perm.data(), perm_dev, ndf*sizeof(int), hipMemcpyDeviceToHost);
    std::vector<int> cnt(npe*npe, 0), ent(npe*npe*4, -1), inv(npe*nfe, -1);
    for (int l = 0; l < nfe; l++) {                       // ascending l: the fixed summation order
        for (int a = 0; a < npf; a++) { inv[perm[a + npf*l] + npe*l] = a;
            for (int b = 0; b < npf; b++) { const int t = perm[a + npf*l], p = perm[b + npf*l], tp = t + npe*p;
                if (cnt[tp] >= 3) { printf("MKERR facetab: >3 faces share a node pair (fused epilogue holds 3)\n"); continue; }
                ent[4*tp + cnt[tp]++] = (a + npf*b) + npf*npf*l; } } }
    FaceTab ft; hipMalloc(&ft.cnt, cnt.size()*sizeof(int)); hipMalloc(&ft.ent, ent.size()*sizeof(int)); hipMalloc(&ft.inv, inv.size()*sizeof(int));
    hipMemcpy(ft.cnt, cnt.data(), cnt.size()*sizeof(int), hipMemcpyHostToDevice); hipMemcpy(ft.ent, ent.data(), ent.size()*sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(ft.inv, inv.data(), inv.size()*sizeof(int), hipMemcpyHostToDevice);
    return cache.emplace(perm_dev, ft).first->second;
}
// D[tp + M*e] += sum over the faces holding (t,p), ascending l, of Dtmp[(a+npf b) + npf^2 (l + nfe e)]
extern "C" __global__ void __launch_bounds__(256) mkjac_asmBD(double* D, const double* Dtmp, const int* cnt, const int* ent,
        const int M, const int npf2nfe, const long long total)
{
    const long long idx = (long long)blockIdx.x * 256 + threadIdx.x; if (idx >= total) return;
    const int tp = (int)(idx % M); const long long e = idx / M;
    const int c = cnt[tp]; if (c == 0) return;
    double v = D[idx];
    for (int q = 0; q < c; q++) v += Dtmp[ent[4*tp + q] + (long long)npf2nfe * e];
    D[idx] = v;
}
// F[t + npe j + npe npf l + npe ndf e] = 0 + Ftmp[(a + npf b) + npf^2 (l + nfe e)] if t = perm[a + npf l] (j = b), else 0
extern "C" __global__ void __launch_bounds__(256) mkjac_asmF(double* F, const double* Ftmp, const int* inv, const int npe,
        const int npf, const int nfe, const long long total)
{
    const long long idx = (long long)blockIdx.x * 256 + threadIdx.x; if (idx >= total) return;
    const int t = (int)(idx % npe); long long r = idx / npe; const int b = (int)(r % npf); r /= npf; const int l = (int)(r % nfe); const long long e = r / nfe;
    const int a = inv[t + npe*l];
    F[idx] = a < 0 ? 0.0 : 0.0 + Ftmp[(a + npf*b) + (long long)npf*npf*(l + (long long)nfe*e)];
}
inline const int* ft_cnt(const FaceTab* f) { return f->cnt; }
inline const int* ft_ent(const FaceTab* f) { return f->ent; }
// face terms of block jth into res.H (Dtmp, Btmp, Ftmp): production :905-1075 verbatim
inline void face_terms(PrecondStageContext& c, Int jth, bool frg = false)
{
    auto& sol = c.sol; auto& res = c.res; auto& app = c.app; auto& master = c.master; auto& mesh = c.mesh; auto& tmp = c.tmp;
    auto& common = c.common; auto handle = c.handle; const Int backend = c.backend; auto& driver_abi = c.driver_abi;
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncq = common.components.ncq;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int ncw = common.components.ncw;
    Int nd = common.grid.nd;
    Int npe = common.grid.npe;
    Int npf = common.grid.npf;
    Int ngf = common.grid.ngf;
    Int nfe = common.meshsizes.nfe;

    Int e1 = common.eblks[3*jth]-1;
    Int e2 = common.eblks[3*jth+1];
    Int ne = e2-e1;
    Int nf = nfe*ne;
    Int nn = npf*nf;
    Int nga = ngf*nf;

    Int n4 = nga*ncu;
    Int n5 = nga*(ncu+nc);
    Int n6 = nga*(ncu+nc+nco);
    Int n7 = nga*(ncu+nc+nco+ncw);
    Int n8 = nga*(ncu+nc+nco+ncw+ncw);
    Int nm = ngf*nfe*e1*(ncx+nd+1);

    dstype *xg = &sol.elemfaceg[nm];
    dstype *nlg = &sol.elemfaceg[nm + nga*ncx];
    dstype *jac = &sol.elemfaceg[nm + nga*(ncx+nd)];

    dstype *uhg = &tmp.tempg[0];
    dstype *udg = &tmp.tempg[n4];
    dstype *odg = &tmp.tempg[n5];
    dstype *wsrc = &tmp.tempg[n6];
    dstype *wdg = &tmp.tempg[n7];

    dstype *fg     = &tmp.tempg[n8];
    dstype *fg_uq  = &tmp.tempg[n8 + nga*ncu*nd];
    dstype *fh_uh  = &tmp.tempg[n8 + nga*ncu*nd + nga*ncu*nd*nc];
    dstype *fg_w   = &tmp.tempg[n8 + nga*ncu*nd + nga*ncu*nd*nc + nga*ncu*ncu];
    dstype *wdg_uq = &tmp.tempg[n8 + nga*ncu*nd + nga*ncu*nd*nc + nga*ncu*ncu + nga*ncu*nd*ncw];

    MKJT_START();
    GetElementFaceNodesLDG(tmp.tempn, sol.uh, mesh.elemcon, npf*nfe, ncu, npf, e1, e2);
    GetElementFaceNodes(&tmp.tempn[nn*ncu], sol.udg, mesh.perm, npf*nfe, nc, npe, nc, e1, e2);
    if (nco > 0)
        GetElementFaceNodes(&tmp.tempn[nn*(ncu+nc)], sol.odg, mesh.perm, npf*nfe, nco, npe, nco, e1, e2);
    if ((ncw > 0) && (common.timeparams.wave == 0)) {
        GetElementFaceNodes(&tmp.tempn[nn*(ncu+nc+nco)], sol.wsrc, mesh.perm, npf*nfe, ncw, npe, ncw, e1, e2);
        GetElementFaceNodes(&tmp.tempn[nn*(ncu+nc+nco+ncw)], sol.wdg, mesh.perm, npf*nfe, ncw, npe, ncw, e1, e2);
    }

    Node2Gauss(handle, tmp.tempg, tmp.tempn, master.shapfgt,
            ngf, npf, nf*(ncu+nc+nco+ncw+ncw), backend);

    MKJT(0);
    if ((ncw > 0) && (common.timeparams.wave == 0)) {
        wEquation<exasim::detail::AbiAdapter>(wdg, wdg_uq, xg, udg, odg, wsrc, tmp.tempn, app, common, nga, backend, tmp.tempi);
    }

    MKJT(1);
    ArraySetValue(fg, 0.0, nga*ncu*nd);
    ArraySetValue(fg_uq, 0.0, nga*ncu*nd*nc);
    if (ncw > 0)
        ArraySetValue(fg_w, 0.0, nga*ncu*nd*ncw);

    FluxDriver(fg, fg_uq, fg_w, xg, udg, odg, wdg, driver_abi,
            mesh, master, app, sol, tmp, common, ngf*nfe, e1, e2, backend);

    ArraySetValue(fh_uh, 0.0, nga*ncu*ncu);
    LDGFluxDerivativeDotNormal(tmp.tempn, fg_uq, nlg, 0.5, nga, ncu, nd, nc);
    ArrayCopy(fg_uq, tmp.tempn, nga*ncu*nc);

    if ((ncw > 0) && (common.timeparams.wave == 0)) {
        LDGFluxDerivativeDotNormal(tmp.tempn, fg_w, nlg, 0.5, nga, ncu, nd, ncw);
        ArrayCopy(fg_w, tmp.tempn, nga*ncu*ncw);
        ArrayGemmBatch2(fg_uq, fg_w, wdg_uq, one, ncu, nc, ncw, nga);
    }

    LDGAddTraceStabilizationDerivatives(fg_uq, fh_uh, app.tau, common.components.ntau,
            nga, ncu, nc);

    columnwiseMultiply(fg_uq, fg_uq, jac, nga, ncu*nc);
    columnwiseMultiply(fh_uh, fh_uh, jac, nga, ncu*ncu);

    MKJT(2);
    dstype *Dtmp = &res.H[0];
    dstype *Btmp = &res.H[npf*npf*nf*ncu*ncu];
    dstype *Ftmp = &res.H[npf*npf*nf*ncu*ncu + npf*npf*nf*ncu*ncq];

    if (frg) ArrayCopy(res.H, fg_uq, nga*ncu*nc);          // FRG (the D/B GEMMs are formed in the element epilogue)
    else {
    Gauss2Node(handle, Dtmp, fg_uq, master.shapfgwdotshapfg,
            ngf, npf*npf, nf*ncu*ncu, backend);

    if (ncq > 0) {
        Gauss2Node(handle, Btmp, &fg_uq[nga*ncu*ncu],
                master.shapfgwdotshapfg, ngf, npf*npf, nf*ncu*ncq, backend);
    }
    }

    Gauss2Node(handle, Ftmp, fh_uh, master.shapfgwdotshapfg,
            ngf, npf*npf, nf*ncu*ncu, backend);

    MKJT(3);
    for (Int ibc = 0; ibc < common.meshsizes.maxnbc; ibc++) {
        Int n = ibc + common.meshsizes.maxnbc*jth;
        Int start = common.nboufaces[n];
        Int nfaces = common.nboufaces[n + 1] - start;
        if (nfaces > 0) {
            Int ngb = nfaces*ngf;
            dstype *xgb = &tmp.tempg[n8];
            dstype *ugb = &tmp.tempg[n8 + ngb*ncx];
            dstype *ogb = &tmp.tempg[n8 + ngb*ncx + ngb*nc];
            dstype *wgb = &tmp.tempg[n8 + ngb*ncx + ngb*nc + ngb*nco];
            dstype *uhb = &tmp.tempg[n8 + ngb*ncx + ngb*nc + ngb*nco + ngb*ncw];
            dstype *nlb = &tmp.tempg[n8 + ngb*ncx + ngb*nc + ngb*nco + ngb*ncw + ngb*ncu];
            dstype *wsb = &tmp.tempg[n8 + ngb*ncx + ngb*nc + ngb*nco + ngb*ncw + ngb*ncu + ngb*nd];
            dstype *fhb = &tmp.tempg[n8 + ngb*ncx + ngb*nc + ngb*nco + ngb*ncw + ngb*ncu + ngb*nd + ngb*ncw];
            dstype *fhb_uq = &tmp.tempg[n8 + ngb*ncx + ngb*nc + ngb*nco + ngb*ncw + ngb*ncu + ngb*nd + ngb*ncw + ngb*ncu];
            dstype *fhb_w = &tmp.tempg[n8 + ngb*ncx + ngb*nc + ngb*nco + ngb*ncw + ngb*ncu + ngb*nd + ngb*ncw + ngb*ncu + ngb*ncu*nc];
            dstype *fhb_uh = &tmp.tempg[n8 + ngb*ncx + ngb*nc + ngb*nco + ngb*ncw + ngb*ncu + ngb*nd + ngb*ncw + ngb*ncu + ngb*ncu*nc + ngb*ncu*ncw];
            dstype *wgb_uq = &tmp.tempg[n8 + ngb*ncx + ngb*nc + ngb*nco + ngb*ncw + ngb*ncu + ngb*nd + ngb*ncw + ngb*ncu + ngb*ncu*nc + ngb*ncu*ncw + ngb*ncu*ncu];

            GetBoundaryNodes(xgb, xg, &mesh.boufaces[start], ngf, nfe, ne, ncx, nfaces);
            GetBoundaryNodes(ugb, udg, &mesh.boufaces[start], ngf, nfe, ne, nc, nfaces);
            GetBoundaryNodes(ogb, odg, &mesh.boufaces[start], ngf, nfe, ne, nco, nfaces);
            GetBoundaryNodes(wgb, wdg, &mesh.boufaces[start], ngf, nfe, ne, ncw, nfaces);
            GetBoundaryNodes(wsb, wsrc, &mesh.boufaces[start], ngf, nfe, ne, ncw, nfaces);
            GetBoundaryNodes(uhb, uhg, &mesh.boufaces[start], ngf, nfe, ne, ncu, nfaces);
            GetBoundaryNodes(nlb, nlg, &mesh.boufaces[start], ngf, nfe, ne, nd, nfaces);

            if ((ncw > 0) && (common.timeparams.wave == 0)) {
            MKJT(4);
                ArrayCopy(res.F, ugb, ngb*nc);
                ArrayCopy(res.F, uhb, ngb*ncu);
                wEquation<exasim::detail::AbiAdapter>(wgb, wgb_uq, xgb, res.F, ogb, wsb, &res.F[ngb*nc],
                        app, common, ngb, backend, tmp.tempi);
            }

            MKJT(5);
            ArraySetValue(fhb, 0.0, ngb*ncu);
            ArraySetValue(fhb_uq, 0.0, ngb*ncu*nc);
            ArraySetValue(fhb_uh, 0.0, ngb*ncu*ncu);
            if (ncw > 0) ArraySetValue(fhb_w, 0.0, ngb*ncu*ncw);

            FbouJacDriver(fhb, fhb_uq, fhb_w, fhb_uh, xgb, ugb, ogb,
                          wgb, uhb, nlb, driver_abi, mesh, master, app,
                          sol, tmp, common, ngb, ibc+1, backend);

            if ((ncw > 0) && (common.timeparams.wave == 0)) {
                ArrayGemmBatch2(fhb_uh, fhb_w, wgb_uq, one, ncu, ncu, ncw, ngb);
                ArraySetValue(wgb_uq, 0.0, ngb*ncw*ncu);
                ArrayGemmBatch2(fhb_uq, fhb_w, wgb_uq, one, ncu, nc, ncw, ngb);
            }

            dstype *jacb = &tmp.tempg[n8];
            GetBoundaryNodes(jacb, jac, &mesh.boufaces[start], ngf, nfe, ne, 1, nfaces);
            columnwiseMultiply(fhb_uq, fhb_uq, jacb, ngb, ncu*nc);
            columnwiseMultiply(fhb_uh, fhb_uh, jacb, ngb, ncu*ncu);

            dstype *Rb = res.F;
            if (frg) PutBoundaryNodes(res.H, fhb_uq, &mesh.boufaces[start], ngf, nfe, ne, ncu*nc, nfaces);
            else {
            Gauss2Node(handle, Rb, fhb_uq, master.shapfgwdotshapfg,
                    ngf, npf*npf, nfaces*ncu*ncu, backend);
            PutBoundaryNodes(Dtmp, Rb, &mesh.boufaces[start],
                    npf*npf, nfe, ne, ncu*ncu, nfaces);

            if (ncq > 0) {
                Gauss2Node(handle, Rb, &fhb_uq[ngb*ncu*ncu],
                        master.shapfgwdotshapfg, ngf, npf*npf,
                        nfaces*ncu*ncq, backend);
                PutBoundaryNodes(Btmp, Rb, &mesh.boufaces[start],
                        npf*npf, nfe, ne, ncu*ncq, nfaces);
            }
            }

            Gauss2Node(handle, Rb, fhb_uh, master.shapfgwdotshapfg,
                    ngf, npf*npf, nfaces*ncu*ncu, backend);
            PutBoundaryNodes(Ftmp, Rb, &mesh.boufaces[start],
                    npf*npf, nfe, ne, ncu*ncu, nfaces);
        }
    }    
    // assembly (production :1077-1082: atomic_add of Dtmp/Btmp into D/B, zero F + atomic scatter of Ftmp)
    MKJT(4);
}
// F in Schur layout: Fs[(t + npe r) + nn_ (j + ndf s) + nn_ mm_ el] (j = b + npf l), i.e. LDGSchurMatrixF of production's F
extern "C" __global__ void __launch_bounds__(256) mkjac_asmF_schur(double* F, const double* Ftmp, const int* inv, const int npe,
        const int npf, const int nfe, const int ncu, const int ne, const long long total)
{
    const long long idx = (long long)blockIdx.x * 256 + threadIdx.x; if (idx >= total) return;
    const int nn_ = npe*ncu, ndf = npf*nfe, mm_ = ndf*ncu;
    const int row = (int)(idx % nn_); const long long q = idx / nn_; const int col = (int)(q % mm_); const long long el = q / mm_;
    const int t = row % npe, r = row / npe, j = col % ndf, s = col / ndf, b = j % npf, l = j / npf;
    const long long e = el + (long long)ne*(r + ncu*s);
    const int a = inv[t + npe*l];
    F[idx] = a < 0 ? 0.0 : 0.0 + Ftmp[(a + npf*b) + (long long)npf*npf*(l + (long long)nfe*e)];
}
inline void asmF(PrecondStageContext& c, Int jth, const FaceTab& ft, bool sl = false)
{
    auto& res = c.res; auto& common = c.common;
    const Int ncu = common.components.ncu, npe = common.grid.npe, npf = common.grid.npf, nfe = common.meshsizes.nfe;
    const Int e1 = common.eblks[3*jth]-1, e2 = common.eblks[3*jth+1], ne = e2-e1, nf = nfe*ne;
    const double* Ftmp = &res.H[npf*npf*nf*ncu*ncu + npf*npf*nf*ncu*common.components.ncq];
    const long long tF = (long long)npe*npf*nfe*ne*ncu*ncu;
    if (sl) hipLaunchKernelGGL(mkjac_asmF_schur, dim3((tF + 255)/256), dim3(256), 0, mk_stream(), res.F, Ftmp, ft.inv, npe, npf, nfe, ncu, ne, tF);
    else hipLaunchKernelGGL(mkjac_asmF, dim3((tF + 255)/256), dim3(256), 0, mk_stream(), res.F, Ftmp, ft.inv, npe, npf, nfe, tF);
    MK_HIPCHECK("mkjac_asmF");
}
// the element-face stage on its own: face terms, then D/B += faces (gather, ascending l) and F written whole
inline void elemface(PrecondStageContext& c, Int jth)
{
    face_terms(c, jth);
    MKJT_START();
    auto& res = c.res; auto& mesh = c.mesh; auto& common = c.common;
    const Int ncu = common.components.ncu, ncq = common.components.ncq, npe = common.grid.npe, npf = common.grid.npf, nfe = common.meshsizes.nfe;
    const Int e1 = common.eblks[3*jth]-1, e2 = common.eblks[3*jth+1], ne = e2-e1, nf = nfe*ne;
    const double* Dtmp = &res.H[0]; const double* Btmp = &res.H[npf*npf*nf*ncu*ncu];
    FaceTab& ft = facetab(mesh.perm, npe, npf, nfe);
    {   const long long tD = (long long)npe*npe*ne*ncu*ncu;
        hipLaunchKernelGGL(mkjac_asmBD, dim3((tD + 255)/256), dim3(256), 0, mk_stream(), res.D, Dtmp, ft.cnt, ft.ent, npe*npe, npf*npf*nfe, tD); }
    if (ncq > 0) { const long long tB = (long long)npe*npe*ne*ncu*ncq;
        hipLaunchKernelGGL(mkjac_asmBD, dim3((tB + 255)/256), dim3(256), 0, mk_stream(), res.B, Btmp, ft.cnt, ft.ent, npe*npe, npf*npf*nfe, tB); }
    asmF(c, jth, ft);
    MK_HIPCHECK("mkjac_asm");
    MKJT(6);
}
// element + element-face stages fused: face terms first (they do not read D/B), F written, then the element stage whose
// GEMM epilogue adds the face terms -- one store of D and B. Registered as `elem`, with `elemface` a no-op.
inline void elem_fused(PrecondStageContext& c, Int jth)
{
    if (!elem_ok(c.common)) {       // not the generated sizes: production order and functions
        uEquationElemBlock<exasim::detail::AbiAdapter>(c.sol, c.res, c.app, c.master, c.mesh, c.tmp, c.common, c.handle, jth, c.backend);
        uEquationElemFaceBlockLDG(c.sol, c.res, c.app, c.driver_abi, c.master, c.mesh, c.tmp, c.common, c.handle, jth, c.backend);
        return; }
    face_terms(c, jth);
    FaceTab& ft = facetab(c.mesh.perm, c.common.grid.npe, c.common.grid.npf, c.common.meshsizes.nfe);
    asmF(c, jth, ft);
    elem_impl(c, jth, &ft);
}
inline void elemface_noop(PrecondStageContext&, Int) {}
// v2: the face GEMMs for D/B also move into the element epilogue (face_terms keeps Gauss-level FRG; F as before)
inline void elem_fused2(PrecondStageContext& c, Int jth, bool sl)
{
    auto& common = c.common;
    if (!elem_ok(common) || common.meshsizes.nfe != NFE || common.grid.ngf != NGF || common.grid.npe*common.grid.npe > PM) {
        elem_fused(c, jth); if (sl) { printf("MKERR elem_fused2: Schur layout requested on the fallback path\n"); } return; }
    face_terms(c, jth, true);
    FaceTab& ft = facetab(c.mesh.perm, common.grid.npe, common.grid.npf, common.meshsizes.nfe);
    asmF(c, jth, ft, sl);
    FaceOp& fo = faceop(c.mesh.perm, c.master.shapfgwdotshapfg, common.grid.npe, common.grid.npf, common.meshsizes.nfe, common.grid.ngf);
    elem_impl(c, jth, nullptr, &fo, sl);
}
inline void elem_fused2(PrecondStageContext& c, Int jth) { elem_fused2(c, jth, false); }
inline void elem_fused2_nosl(PrecondStageContext& c, Int jth) { elem_fused2(c, jth, false); }
inline void elem_fused2_sl(PrecondStageContext& c, Int jth) { elem_fused2(c, jth, true); }


// cross-face GEMM + scatter, fused, reading bufq directly (no LDGPackFaceQForCrossGEMM):
//   Af[row, tnode] = sum_mid B[row, mid] Ef[mid, tnode],  B[row, mid] = bufq[a + npf b + npf^2 (d + nd (m + ncu (c + ncu f)))]
//   row = a + npf (m + ncu c), mid = b + npf d;  then A[...] += Af as LDGScatterCrossFaceGEMMBlock (atomic, production's indices).
// One wave per (face, 16-row tile), MFMA f64 16x16x4 over mid (27 -> 28). Replaces the rocBLAS strided-batched GEMM: same
// products, different summation grouping (rounding level, like the Schur BMinv kernels).
extern "C" __global__ void __launch_bounds__(64) mkjac_cross(double* __restrict A, const double* __restrict bufq,
        const double* __restrict Ef, const int* __restrict facecon, const int* __restrict f2e, const int sideResidual,
        const int f1, const int npe, const int npf, const int ncu, const int nd, const int ne)
{
    const int nrow = npf*ncu*ncu, nmid = npf*nd, nlocu = npe*ncu, ntile = (nrow + 15) / 16;
    const int flocal = blockIdx.x / ntile, r0 = 16 * (blockIdx.x % ntile), f = f1 + flocal;
    if (f2e[4*f + 2] < 0) return;                                   // boundary face: no cross term (uniform per wave)
    const int l = threadIdx.x, lr = l % 16, lk = l / 16;
    const int row = r0 + lr, a = row % npf, s_ = row / npf, m = s_ % ncu, c = s_ / ncu;
    d4 acc = (d4){0.0, 0.0, 0.0, 0.0};
    for (int k0 = 0; k0 < nmid; k0 += 4) {
        const int mid = k0 + lk, b = mid % npf, d = mid / npf;
        const double av = (row < nrow && mid < nmid) ? bufq[a + npf*b + (size_t)npf*npf*(d + nd*(m + ncu*(c + ncu*(size_t)flocal)))] : 0.0;
        const double bv = (lr < npf && mid < nmid) ? Ef[mid + (size_t)nmid*(lr + npf*(size_t)flocal)] : 0.0;
        acc = __builtin_amdgcn_mfma_f64_16x16x4f64(av, bv, acc, 0, 0, 0);
    }
    const int tnode = lr, so = sideResidual - 1;                    // D layout: row 4r + l/16, col (tnode) l%16
    if (tnode >= npf) return;
    const int kt = facecon[2*(tnode + npf*f) + so], unode = kt % npe, uelem = (kt - unode) / npe;
#pragma unroll
    for (int r = 0; r < 4; r++) {
        const int rp = r0 + 4*r + lk; if (rp >= nrow) continue;
        const int aa = rp % npf, ss = rp / npf, mm = ss % ncu, cc = ss / ncu;
        const int kr = facecon[2*(aa + npf*f) + so], rownode = kr % npe, rowelem = (kr - rownode) / npe;
        if (rowelem < 0 || rowelem >= ne || rowelem != uelem) continue;
        atomicAdd(&A[(rownode + npe*mm) + (size_t)nlocu*(unode + npe*cc) + (size_t)nlocu*nlocu*rowelem], acc[r]);
    }
}
// ---- cross-face stage: production RuFaceCrossDerivOptimized (ldgblockjacobian.cpp) verbatim + section timers (MK_JACT)
// slots (both sides, all face blocks): 16 gather+interp, 17 flux Jacobian, 18 dot+GEMM+pack, 19 slot map+Ef+scale, 20 GEMM+scatter
inline void cross(PrecondStageContext& c, dstype* A)
{
    auto& sol = c.sol; auto& res = c.res; auto& app = c.app; auto& master = c.master; auto& mesh = c.mesh; auto& tmp = c.tmp;
    auto& common = c.common; auto& driver_abi = c.driver_abi;
    static const bool cross_v2 = [] { const char* e = getenv("MK_CROSS_V"); return !(e && e[0] == '1'); }();   // MK_CROSS_V=1: verbatim

    if (common.components.ncq <= 0)
        return;

    Int backend = common.backend;
    Int npe = common.grid.npe;
    Int npf = common.grid.npf;
    Int ngf = common.grid.ngf;
    Int ncu = common.components.ncu;
    //Int ncq = common.components.ncq;
    Int nc = common.components.nc;
    Int nco = common.components.nco;
    Int ncw = common.components.ncw;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int ne = common.meshsizes.ne1;

    dstype scalar = 1.0;
    if (common.timeparams.wave == 1)
        scalar = 1.0/common.timestate.dtfactor;

    MKJT_START();
    for (Int jblk = 0; jblk < common.meshsizes.nbf; jblk++) {
        Int f1 = common.fblks[3*jblk]-1;
        Int f2 = common.fblks[3*jblk+1];
        Int ib = common.fblks[3*jblk+2];
        if (ib != 0)
            continue;

        Int nfb = f2 - f1;
        Int nn = npf*nfb;
        Int nga = ngf*nfb;
        Int M = nga*ncu;
        Int N = nga*ncu*nd;
        Int nm = ngf*f1*(ncx+nd+1);

        Int fncols = ncu + 2*nc + 2*ncw;
        dstype *fn = tmp.tempn;
        dstype *fg = tmp.tempg;
        Int fluxSize = N;
        Int fluxWSize = max((Int) 1, fluxSize*ncw);
        dstype *flux = &fg[nga*fncols];
        dstype *flux_udg = &flux[fluxSize];
        dstype *fw = &flux_udg[fluxSize*nc];

        Int szFq = ngf*nd*ncu*ncu*nfb;
        Int szBufq = npf*npf*nd*ncu*ncu*nfb;
        Int szEf = npf*nd*npf*nfb;
        //Int szAf = npf*ncu*ncu*npf*nfb;

        dstype *fq = res.H;
        dstype *B = fq;
        dstype *bufq = &res.H[max(szFq, szBufq)];
        dstype *Ef = bufq;
        dstype *Af = &bufq[max(szBufq, szEf)];
        if (cross_v2) Ef = Af;          // production builds Ef over bufq (safe only after the pack copied bufq out); v2 reads bufq
        //(void)szAf;

        GetElemNodes(fn, sol.uh, npf, ncu, 0, ncu, f1, f2);
        GetArrayAtIndex(&fn[nn*ncu], sol.udg, &mesh.findudg1[npf*nc*f1], nn*nc);
        if (ncw > 0)
            GetFaceNodes(&fn[nn*(ncu+nc)], sol.wdg, mesh.facecon,
                    npf, ncw, npe, ncw, f1, f2, 1);
        GetArrayAtIndex(&fn[nn*(ncu+nc+ncw)], sol.udg,
                &mesh.findudg2[npf*nc*f1], nn*nc);
        if (ncw > 0)
            GetFaceNodes(&fn[nn*(ncu+2*nc+ncw)], sol.wdg, mesh.facecon,
                    npf, ncw, npe, ncw, f1, f2, 2);

        Node2Gauss(common.cublasHandle, fg, fn, master.shapfgt,
                ngf, npf, nfb*fncols, backend);
        MKJT(16);

        dstype *ug1 = &fg[nga*ncu];
        dstype *wg1 = (ncw > 0) ? &fg[nga*(ncu+nc)] : nullptr;
        dstype *ug2 = &fg[nga*(ncu+nc+ncw)];
        dstype *wg2 = (ncw > 0) ? &fg[nga*(ncu+2*nc+ncw)] : nullptr;
        dstype *xg = &sol.faceg[nm];
        dstype *nlg = &sol.faceg[nm + nga*ncx];
        dstype *jac = &sol.faceg[nm + nga*(ncx+nd)];
        dstype *og1 = (nco > 0) ? &sol.og1[ngf*nco*f1] : nullptr;
        dstype *og2 = (nco > 0) ? &sol.og2[ngf*nco*f1] : nullptr;

        ArraySetValue(fw, 0.0, fluxWSize);
        FluxDriver(flux, flux_udg, fw, xg, ug1, og1, wg1, driver_abi,
                   mesh, master, app, sol, tmp, common, ngf, f1, f2, backend);
        MKJT(17);
        LDGFluxQDerivativeDotNormalJac(fq, flux_udg, nlg, jac, 0.5,
                ngf, nfb, ncu, nd, nc);
        Gauss2Node(common.cublasHandle, bufq, fq, master.shapfgwdotshapfg,
                ngf, npf*npf, nfb*nd*ncu*ncu, backend);
        if (!cross_v2) LDGPackFaceQForCrossGEMM(B, bufq, npf, ncu, nd, nfb);
        MKJT(18);
        LDGBuildFaceSlotQMap(res.ipiv, mesh.f2e, mesh.elemcon, 1,
                f1, nfb, npf, common.meshsizes.nfe);
        LDGBuildFaceEForCrossBlockOptimized(Ef, res.E, mesh.f2e,
                mesh.perm, res.ipiv, 1, f1, nfb, npe, npf, common.meshsizes.nfe,
                nd, common.meshsizes.ne);
        ArrayMultiplyScalar(common.cublasHandle, Ef, 0.5*scalar,
                szEf, backend);
        MKJT(19);
        if (cross_v2) {
            const int ntile = (npf*ncu*ncu + 15) / 16;
            hipLaunchKernelGGL(mkjac_cross, dim3(nfb*ntile), dim3(64), 0, mk_stream(), A, bufq, Ef, mesh.facecon, mesh.f2e, 2,
                               (int)f1, (int)npe, (int)npf, (int)ncu, (int)nd, (int)ne);
            MK_HIPCHECK("mkjac_cross");
        } else {
        PGEMNMStridedBached(common.cublasHandle, npf*ncu*ncu, npf,
                npf*nd, one, B, npf*ncu*ncu, Ef, npf*nd, 0.0,
                Af, npf*ncu*ncu, nfb, backend);
        LDGScatterCrossFaceGEMMBlock(A, Af, mesh.facecon, mesh.f2e,
                2, f1, nfb, npe, npf, ncu, ne);
        }
        MKJT(20);

        ArraySetValue(fw, 0.0, fluxWSize);
        FluxDriver(flux, flux_udg, fw, xg, ug2, og2, wg2, driver_abi,
                   mesh, master, app, sol, tmp, common, ngf, f1, f2, backend);
        MKJT(17);
        LDGFluxQDerivativeDotNormalJac(fq, flux_udg, nlg, jac, 0.5,
                ngf, nfb, ncu, nd, nc);
        Gauss2Node(common.cublasHandle, bufq, fq, master.shapfgwdotshapfg,
                ngf, npf*npf, nfb*nd*ncu*ncu, backend);
        if (!cross_v2) LDGPackFaceQForCrossGEMM(B, bufq, npf, ncu, nd, nfb);
        MKJT(18);
        LDGBuildFaceSlotQMap(res.ipiv, mesh.f2e, mesh.elemcon, 2,
                f1, nfb, npf, common.meshsizes.nfe);
        LDGBuildFaceEForCrossBlockOptimized(Ef, res.E, mesh.f2e,
                mesh.perm, res.ipiv, 2, f1, nfb, npe, npf, common.meshsizes.nfe,
                nd, common.meshsizes.ne);
        ArrayMultiplyScalar(common.cublasHandle, Ef, 0.5*scalar*minusone,
                szEf, backend);
        MKJT(19);
        if (cross_v2) {
            const int ntile = (npf*ncu*ncu + 15) / 16;
            hipLaunchKernelGGL(mkjac_cross, dim3(nfb*ntile), dim3(64), 0, mk_stream(), A, bufq, Ef, mesh.facecon, mesh.f2e, 1,
                               (int)f1, (int)npe, (int)npf, (int)ncu, (int)nd, (int)ne);
            MK_HIPCHECK("mkjac_cross");
        } else {
        PGEMNMStridedBached(common.cublasHandle, npf*ncu*ncu, npf,
                npf*nd, one, B, npf*ncu*ncu, Ef, npf*nd, 0.0,
                Af, npf*ncu*ncu, nfb, backend);
        LDGScatterCrossFaceGEMMBlock(A, Af, mesh.facecon, mesh.f2e,
                1, f1, nfb, npe, npf, ncu, ne);
        }
        MKJT(20);

        //(void)ncq;
        //(void)szAf;
    }
}
}  // namespace mkjac
