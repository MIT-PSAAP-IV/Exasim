// Kernel-equivalence test for the multi-domain HDG interface-coupling surface
// (Fint / Fext) on the CONCRETE (templated) model path.
//
// The templated `exasim::fint_kernel<M>` / `exasim::fext_kernel<M>` (in
// <exasim/kernels/boundary.hpp>) must produce byte-identical residual + Jacobian
// buffers to the libpdemodel ABI `HdgFint` / `HdgFext` for the same model math
// and the same injected `uext` array — that is the whole correctness bar for the
// coupling surface. This test proves it:
//
//   abi_*  — the ABI kernels text2code emitted from the model files, checked in
//            under tests/coupling-models/reference/ (see that dir's README),
//   M      — a hand-written concrete Model whose fint/fext + Jacobians encode the
//            same pointwise math, driven through the templated kernels here.
//
// Two models are compared:
//   pde2   — apps/poisson/poisson2d/pdemodel2.txt verbatim (Fint has 2 output
//            components with ncu==1; Fext reads uext). Real app coverage.
//   probe  — same shapes but non-trivial Fint/Fext bodies so the uq- and
//            uhat-Jacobians are non-zero, locking the trace/input-index-outer
//            SoA layout that the trivial pdemodel2 bodies cannot exercise.
//
// Agreement is exact (these are the same closed-form expressions), gated at
// rtol 1e-13 to absorb any floating reassociation. This mirrors the style of
// tests/builtin-models/compare_model4.cpp.
//
// Unlike compare_model4 (which shims Kokkos to a serial loop), this test links
// real Kokkos: both the checked-in ABI kernels and the templated kernels call
// Kokkos::parallel_for, and <exasim/kernels/boundary.hpp> pulls <Kokkos_Core.hpp>.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Pulls Kokkos + the Model contract + dstype (via <exasim/common.h>).
#include <exasim/kernels/boundary.hpp>

// ---- ABI reference kernels (text2code output), each in its own namespace so
// the duplicate HdgFint/HdgFint1/HdgFext symbol names do not collide. ----
namespace abi_pde2 {
#include "reference/pde2_HdgFint.cpp"
#include "reference/pde2_HdgFext.cpp"
}
namespace abi_probe {
#include "reference/probe_HdgFint.cpp"
#include "reference/probe_HdgFext.cpp"
}

// ---------------------------------------------------------------------------
// Concrete models. ncu==1, nd==2 (so Nq==3), ncw==0, nparam==1, nco==0.
// Jacobian local layout is TRACE/INPUT-index-outer: fb[j*nf + o] = d f[o]/d in[j].
// ---------------------------------------------------------------------------

// pdemodel2.txt verbatim: Fint fb=[-uh0, 1-uh0]; Fext fb=[uext0 - uh0].
struct Pde2Coupled : exasim::ModelDefaults<Pde2Coupled> {
    static constexpr int nd = 2, ncu = 1, ncw = 0, nparam = 1;
    static constexpr auto disc = exasim::Discretization::HDG;
    static constexpr bool has_external_coupling = true;
    static constexpr int nfint = 2;   // Fint output_size
    static constexpr int nfext = 1;   // Fext output_size
    static constexpr int ncuext = 1;  // width of injected uext

    KOKKOS_INLINE_FUNCTION static
    void flux(double f[], const double[], const double uq[], const double[],
              const double mu[], const double[], double) { f[0]=mu[0]*uq[1]; f[1]=mu[0]*uq[2]; }
    KOKKOS_INLINE_FUNCTION static
    void initu(double ui[], const double[], const double[], const double[]) { ui[0]=0.0; }

    // Fint: fb[0] = -uh0 ; fb[1] = 1 - uh0
    KOKKOS_INLINE_FUNCTION static
    void fint(double fb[], int, const double[], const double[], const double[],
              const double[], const double uh[], const double[], const double[],
              const double[], const double[], double) {
        fb[0] = -uh[0];
        fb[1] = 1.0 - uh[0];
    }
    KOKKOS_INLINE_FUNCTION static
    void fint_jac_uq(double fb_uq[], int, const double[], const double[], const double[],
                     const double[], const double[], const double[], const double[],
                     const double[], const double[], double) {
        for (int k = 0; k < nfint * (ncu*(1+nd)); ++k) fb_uq[k] = 0.0;
    }
    KOKKOS_INLINE_FUNCTION static
    void fint_jac_uh(double fb_uh[], int, const double[], const double[], const double[],
                     const double[], const double[], const double[], const double[],
                     const double[], const double[], double) {
        // j=0 (only trace), o in {0,1}: d fb[o]/d uh0 = -1
        fb_uh[0] = -1.0;   // [j*nfint + o] = [0*2+0]
        fb_uh[1] = -1.0;   //                 [0*2+1]
    }

    // Fext: fb[0] = uext0 - uh0
    KOKKOS_INLINE_FUNCTION static
    void fext(double fb[], int, const double[], const double[], const double[],
              const double[], const double uh[], const double[], const double uext[],
              const double[], const double[], const double[], double) {
        fb[0] = uext[0] - uh[0];
    }
    KOKKOS_INLINE_FUNCTION static
    void fext_jac_uq(double fb_uq[], int, const double[], const double[], const double[],
                     const double[], const double[], const double[], const double[],
                     const double[], const double[], const double[], double) {
        for (int k = 0; k < nfext * (ncu*(1+nd)); ++k) fb_uq[k] = 0.0;
    }
    KOKKOS_INLINE_FUNCTION static
    void fext_jac_uh(double fb_uh[], int, const double[], const double[], const double[],
                     const double[], const double[], const double[], const double[],
                     const double[], const double[], const double[], double) {
        fb_uh[0] = -1.0;
    }
};

// Non-trivial variant (matches tests/coupling-models/reference/probe_*):
//   Fint: fb[0] = uq1*uq2 + 3*uh0 ; fb[1] = 5*uq0 + 7*uh0
//   Fext: fb[0] = uext0*uq2 - uh0
struct ProbeCoupled : exasim::ModelDefaults<ProbeCoupled> {
    static constexpr int nd = 2, ncu = 1, ncw = 0, nparam = 1;
    static constexpr auto disc = exasim::Discretization::HDG;
    static constexpr bool has_external_coupling = true;
    static constexpr int nfint = 2;
    static constexpr int nfext = 1;
    static constexpr int ncuext = 1;

    KOKKOS_INLINE_FUNCTION static
    void flux(double f[], const double[], const double uq[], const double[],
              const double mu[], const double[], double) { f[0]=mu[0]*uq[1]; f[1]=mu[0]*uq[2]; }
    KOKKOS_INLINE_FUNCTION static
    void initu(double ui[], const double[], const double[], const double[]) { ui[0]=0.0; }

    KOKKOS_INLINE_FUNCTION static
    void fint(double fb[], int, const double[], const double uq[], const double[],
              const double[], const double uh[], const double[], const double[],
              const double[], const double[], double) {
        fb[0] = uq[1]*uq[2] + 3.0*uh[0];
        fb[1] = 5.0*uq[0]   + 7.0*uh[0];
    }
    KOKKOS_INLINE_FUNCTION static
    void fint_jac_uq(double fb_uq[], int, const double[], const double uq[], const double[],
                     const double[], const double[], const double[], const double[],
                     const double[], const double[], double) {
        constexpr int Nq = ncu*(1+nd), nf = nfint;
        for (int k = 0; k < nf*Nq; ++k) fb_uq[k] = 0.0;
        // layout [j*nf + o] = d fb[o]/d uq[j]
        fb_uq[0*nf + 1] = 5.0;    // d fb1/d uq0
        fb_uq[1*nf + 0] = uq[2];  // d fb0/d uq1
        fb_uq[2*nf + 0] = uq[1];  // d fb0/d uq2
    }
    KOKKOS_INLINE_FUNCTION static
    void fint_jac_uh(double fb_uh[], int, const double[], const double[], const double[],
                     const double[], const double[], const double[], const double[],
                     const double[], const double[], double) {
        fb_uh[0] = 3.0;   // d fb0/d uh0
        fb_uh[1] = 7.0;   // d fb1/d uh0
    }

    KOKKOS_INLINE_FUNCTION static
    void fext(double fb[], int, const double[], const double uq[], const double[],
              const double[], const double uh[], const double[], const double uext[],
              const double[], const double[], const double[], double) {
        fb[0] = uext[0]*uq[2] - uh[0];
    }
    KOKKOS_INLINE_FUNCTION static
    void fext_jac_uq(double fb_uq[], int, const double[], const double uq[], const double[],
                     const double[], const double[], const double[], const double uext[],
                     const double[], const double[], const double[], double) {
        constexpr int Nq = ncu*(1+nd), nf = nfext;
        for (int k = 0; k < nf*Nq; ++k) fb_uq[k] = 0.0;
        fb_uq[2*nf + 0] = uext[0];  // d fb0/d uq2
    }
    KOKKOS_INLINE_FUNCTION static
    void fext_jac_uh(double fb_uh[], int, const double[], const double[], const double[],
                     const double[], const double[], const double[], const double[],
                     const double[], const double[], const double[], double) {
        fb_uh[0] = -1.0;
    }
};

// ---------------------------------------------------------------------------
static int nfail = 0;
static void cmp(const char* name, const double* a, const double* b, int n,
                double rtol = 1e-13) {
    double worst = 0.0; int wi = -1;
    for (int k = 0; k < n; ++k) {
        double scale = std::fmax(1.0, std::fmax(std::fabs(a[k]), std::fabs(b[k])));
        double err = std::fabs(a[k] - b[k]) / scale;
        if (err > worst) { worst = err; wi = k; }
    }
    if (worst > rtol) {
        ++nfail;
        printf("FAIL %-26s worst rel err %.3e at slot %d (abi=%.17g cxx=%.17g)\n",
               name, worst, wi, a[wi], b[wi]);
    } else {
        printf("ok   %-26s n=%d  worst rel err %.3e\n", name, n, worst);
    }
}

// Test state: ncu=1, nd=2, Nq=nc=3, ncw=0, nco=0, ncuext=1. Distinct per-point
// values so the SoA index math (k*ng + i) is exercised across i.
static constexpr int NG = 3, NCU = 1, ND = 2, NQ = 3, NC = 3, NCW = 0, NCO = 0, NCX = 2, NUE = 1;

static std::vector<double> xdg, udg, uhg, nlg, uext, tau, param, odg, wdg;
static void fill_inputs() {
    xdg.assign(NCX*NG, 0.0);
    udg.assign(NC*NG, 0.0);
    uhg.assign(NCU*NG, 0.0);
    nlg.assign(ND*NG, 0.0);
    uext.assign(NUE*NG, 0.0);
    odg.assign(1, 0.0);      // nco==0, unused
    wdg.assign(1, 0.0);      // ncw==0, unused
    tau.assign(NCU, 5.0);
    param.assign(1, 1.7);
    for (int i = 0; i < NG; ++i) {
        double s = 0.1 * (i + 1);
        for (int k = 0; k < NCX; ++k) xdg[k*NG+i] = 0.3 + s + 0.05*k;
        for (int k = 0; k < NC;  ++k) udg[k*NG+i] = 0.2 + 0.11*k + s;   // uq0,uq1,uq2
        uhg[0*NG+i]  = 0.37 + s;
        for (int k = 0; k < ND;  ++k) nlg[k*NG+i] = (k==0 ? 0.6 : 0.8);
        uext[0*NG+i] = 0.91 - s;
    }
}

template <class M>
static void run_case(const char* tag,
    void (*abi_fint)(dstype*,dstype*,dstype*,dstype*,const dstype*,const dstype*,const dstype*,
                     const dstype*,const dstype*,const dstype*,const dstype*,const dstype*,
                     const dstype*,const dstype,const int,const int,const int,const int,const int,
                     const int,const int,const int,const int),
    void (*abi_fext)(dstype*,dstype*,dstype*,dstype*,const dstype*,const dstype*,const dstype*,
                     const dstype*,const dstype*,const dstype*,const dstype*,const dstype*,
                     const dstype*,const dstype*,const dstype,const int,const int,const int,const int,
                     const int,const int,const int,const int,const int))
{
    const int ib = 1, mn = 0; const double t = 0.0;
    const int nfi = exasim::nfint_v<M>, nfe = exasim::nfext_v<M>;

    std::vector<double> fj_abi(nfi*NG, -7), fjuq_abi(nfi*NC*NG, -7), fjuh_abi(nfi*NCU*NG, -7), fjw(1, 0);
    std::vector<double> fj_cxx(nfi*NG, 9),  fjuq_cxx(nfi*NC*NG, 9),  fjuh_cxx(nfi*NCU*NG, 9);

    // ---- Fint ----
    abi_fint(fj_abi.data(), fjuq_abi.data(), fjw.data(), fjuh_abi.data(),
             xdg.data(), udg.data(), odg.data(), wdg.data(), uhg.data(), nlg.data(),
             tau.data(), nullptr, param.data(), t, mn, ib, NG, NC, NCU, ND, NCX, NCO, NCW);
    exasim::fint_kernel<M>(fj_cxx.data(), fjuq_cxx.data(), fjw.data(), fjuh_cxx.data(),
             xdg.data(), udg.data(), odg.data(), wdg.data(), uhg.data(), nlg.data(),
             tau.data(), nullptr, param.data(), t, mn, ib, NG, NC, NCU, ND, NCX, NCO, NCW);
    Kokkos::fence();
    char nm[64];
    snprintf(nm, sizeof nm, "%s Fint f", tag);        cmp(nm, fj_abi.data(),   fj_cxx.data(),   nfi*NG);
    snprintf(nm, sizeof nm, "%s Fint f_udg", tag);    cmp(nm, fjuq_abi.data(), fjuq_cxx.data(), nfi*NC*NG);
    snprintf(nm, sizeof nm, "%s Fint f_uhg", tag);    cmp(nm, fjuh_abi.data(), fjuh_cxx.data(), nfi*NCU*NG);

    // ---- Fext ----
    std::vector<double> gj_abi(nfe*NG, -7), gjuq_abi(nfe*NC*NG, -7), gjuh_abi(nfe*NCU*NG, -7);
    std::vector<double> gj_cxx(nfe*NG, 9),  gjuq_cxx(nfe*NC*NG, 9),  gjuh_cxx(nfe*NCU*NG, 9);
    abi_fext(gj_abi.data(), gjuq_abi.data(), fjw.data(), gjuh_abi.data(),
             xdg.data(), udg.data(), odg.data(), wdg.data(), uhg.data(), nlg.data(),
             uext.data(), tau.data(), nullptr, param.data(), t, mn, ib, NG, NC, NCU, ND, NCX, NCO, NCW);
    exasim::fext_kernel<M>(gj_cxx.data(), gjuq_cxx.data(), fjw.data(), gjuh_cxx.data(),
             xdg.data(), udg.data(), odg.data(), wdg.data(), uhg.data(), nlg.data(),
             uext.data(), tau.data(), nullptr, param.data(), t, mn, ib, NG, NC, NCU, ND, NCX, NCO, NCW);
    Kokkos::fence();
    snprintf(nm, sizeof nm, "%s Fext f", tag);        cmp(nm, gj_abi.data(),   gj_cxx.data(),   nfe*NG);
    snprintf(nm, sizeof nm, "%s Fext f_udg", tag);    cmp(nm, gjuq_abi.data(), gjuq_cxx.data(), nfe*NC*NG);
    snprintf(nm, sizeof nm, "%s Fext f_uhg", tag);    cmp(nm, gjuh_abi.data(), gjuh_cxx.data(), nfe*NCU*NG);
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        fill_inputs();
        run_case<Pde2Coupled >("pde2 ", abi_pde2::HdgFint,  abi_pde2::HdgFext);
        run_case<ProbeCoupled>("probe", abi_probe::HdgFint, abi_probe::HdgFext);
    }
    Kokkos::finalize();
    printf(nfail ? "\n%d comparisons FAILED\n" : "\nALL comparisons passed\n", nfail);
    return nfail ? 1 : 0;
}
