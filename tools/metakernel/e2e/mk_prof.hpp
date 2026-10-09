// mk_prof.hpp -- profiling-only roctx ranges around every registered stage (MK_PROF_RANGES=1).
// Each range is bracketed by hipDeviceSynchronize, so a kernel's GPU interval lies inside the host range that launched it
// and kernels can be attributed to stages by timestamp containment. The syncs perturb wall time (profiling runs only).
// Wraps: preconditioner build (elem, elemface, schur, cross, inverse), the block-Jacobian apply and the GMRES CGS step,
// and the residual stages (uhat, q, w, elem, face, tail). Production stages that are not table entries run unwrapped
// and are attributed by elimination in the analysis (e.g. GMRES vector ops, the build's state stage, the K copy).
#pragma once
#include <rocprofiler-sdk-roctx/roctx.h>
#include <cstring>

// marker kernels: one launched right after each range push and one right before each pop, so kernels can be attributed
// to ranges by GPU ORDER (k-th mkprof_begin <-> k-th range), independent of the host/GPU clock offset
extern "C" __global__ void mkprof_begin(int) {}
extern "C" __global__ void mkprof_end(int) {}
namespace mkprof {
inline bool on() { static const bool v = [] { const char* e = getenv("MK_PROF_RANGES"); return e && e[0] == '1'; }(); return v; }
inline int& seq() { static int k = 0; return k; }
// stage code = the begin marker's GRID SIZE (visible in the kernel trace) -> attribution needs no marker/host records.
// Keep in sync with STAGES in analyze_counters.py.
inline int code(const char* n)
{
    static const char* tab[] = {"pc.elem", "pc.elemface", "pc.schur", "pc.cross", "pc.inverse", "gm.apply", "gm.cgs",
                                "res.uhat", "res.q", "res.w", "res.elem", "res.face", "res.tail"};
    for (int i = 0; i < (int)(sizeof(tab)/sizeof(tab[0])); i++) if (!strcmp(n, tab[i])) return i + 1;
    return 99;
}
struct R {
    // device syncs on BOTH sides of each marker: stage kernels may run on other streams (hipBLAS/rocBLAS), so the markers
    // must bracket them in GPU order on every stream
    explicit R(const char* n) { hipDeviceSynchronize(); roctxRangePush(n); hipLaunchKernelGGL(mkprof_begin, dim3(code(n)), dim3(1), 0, mk_stream(), seq()); hipDeviceSynchronize(); }
    ~R() { hipDeviceSynchronize(); hipLaunchKernelGGL(mkprof_end, dim3(1), dim3(1), 0, mk_stream(), seq()++); hipDeviceSynchronize(); roctxRangePop(); }
};
inline const PrecondStageTable*& P0() { static const PrecondStageTable* p = nullptr; return p; }
inline const ResidualStageTable*& Q0() { static const ResidualStageTable* q = nullptr; return q; }

inline void p_elem(PrecondStageContext& c, Int j) { R r("pc.elem"); P0()->elem(c, j); }
inline void p_elemface(PrecondStageContext& c, Int j) { R r("pc.elemface"); P0()->elemface(c, j); }
inline void p_schur(ResidualStageContext& c, Int j) { R r("pc.schur"); P0()->schur(c, j); }
inline void p_cross(PrecondStageContext& c, dstype* K) { R r("pc.cross"); P0()->cross(c, K); }
inline void p_inverse(ResidualStageContext& c, dstype* A, Int n, Int b) { R r("pc.inverse"); P0()->inverse(c, A, n, b); }
inline void p_apply(ResidualStageContext& c, dstype* x) { R r("gm.apply"); P0()->apply(c, x); }
inline void p_cgs(cublasHandle_t h, dstype* V, dstype* H, dstype* t, Int N, Int m, Int b) { R r("gm.cgs"); P0()->cgs(h, V, H, t, N, m, b); }

inline void q_uhat(ResidualStageContext& c, int pass) { R r("res.uhat"); Q0()->uhat(c, pass); }
inline void q_q(ResidualStageContext& c, Int a, Int b, int pass) { R r("res.q"); Q0()->q(c, a, b, pass); }
inline void q_w(ResidualStageContext& c, Int a, Int b) { R r("res.w"); Q0()->w(c, a, b); }
inline void q_elem(ResidualStageContext& c, Int a, Int b) { R r("res.elem"); Q0()->elem(c, a, b); }
inline void q_face(ResidualStageContext& c) { R r("res.face"); Q0()->face(c); }
inline void q_tail(ResidualStageContext& c) { R r("res.tail"); Q0()->tail(c); }

// wrapped copies of the tables (entries that are null stay null: production runs, unattributed)
inline const PrecondStageTable* wrap(const PrecondStageTable* t)
{
    if (!on() || !t) return t;
    static PrecondStageTable w; P0() = t; w = *t; w.name = "metakernel-pc+roctx";
    if (t->elem) w.elem = p_elem; if (t->elemface) w.elemface = p_elemface; if (t->schur) w.schur = p_schur;
    if (t->cross) w.cross = p_cross; if (t->inverse) w.inverse = p_inverse; if (t->apply) w.apply = p_apply; if (t->cgs) w.cgs = p_cgs;
    return &w;
}
inline const ResidualStageTable* wrap(const ResidualStageTable* t)
{
    if (!on() || !t) return t;
    static ResidualStageTable w; Q0() = t; w = *t; w.name = "metakernel+roctx";
    if (t->uhat) w.uhat = q_uhat; if (t->q) w.q = q_q; if (t->w) w.w = q_w; if (t->elem) w.elem = q_elem;
    if (t->face) w.face = q_face; if (t->tail) w.tail = q_tail;
    return &w;
}
}  // namespace mkprof
