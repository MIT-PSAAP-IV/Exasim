// Kernel-equivalence test for the implicit HDG w-equation source (Sourcew) on
// the CONCRETE (templated) model path.
//
// The templated `exasim::hdg_sourcew_kernel<M>` (value + ∂sw/∂udg + ∂sw/∂w) and
// `exasim::hdg_sourcewonly_kernel<M>` (value + ∂sw/∂w) — in
// <exasim/kernels/sourcew.hpp>, routed from backend/Discretization/wequation.hpp
// via EXASIM_LEGACY_W_CALL for ncw>0 models — must produce byte-identical
// residual + Jacobian buffers to the libpdemodel ABI `HdgSourcew` /
// `HdgSourcewonly` for the same model math. That is the whole correctness bar
// for the HDG w-equation Newton block. This test proves it:
//
//   abi_*  — the ABI kernels in text2code's generated style, checked in under
//            tests/w-models/reference/ (see that dir's header comment),
//   WProbe — a hand-written concrete Model whose sourcew + Jacobians encode the
//            same pointwise math, driven through the templated kernels here.
//
// The probe has ncw==2 and non-trivial dependence on BOTH udg and wdg, so it
// locks the input-index-outer SoA layout of f_udg (size ncw*Nq) and f_wdg
// (size ncw*ncw) — the layout model4 (ncw==1) cannot distinguish. Agreement is
// exact (same closed-form expressions); gated at rtol 1e-13 to absorb any
// floating reassociation. Mirrors tests/coupling-models/compare_fint_fext.cpp.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Pulls Kokkos + the Model contract + dstype (via <exasim/common.h>).
#include <exasim/kernels/sourcew.hpp>

// ---- ABI reference kernels (text2code-style), in their own namespace. ----
namespace abi_probe {
#include "reference/probe_HdgSourcew.cpp"
}

// ---------------------------------------------------------------------------
// Concrete model. nd==2, ncu==2, ncw==2, nparam==1, nco==0 => Nq==nc==6.
// Jacobian local layout is INPUT-index-outer: buf[j*ncw + o] = d sw[o]/d in[j].
//   sw[0] = mu0*uq0 + 3*uq4 + w0*w1
//   sw[1] = uq1*uq2 + 5*uq5 + 2*w0 + w1^2
// ---------------------------------------------------------------------------
struct WProbe : exasim::ModelDefaults<WProbe> {
    static constexpr int nd = 2, ncu = 2, ncw = 2, nparam = 1, nco = 0;
    static constexpr auto disc = exasim::Discretization::HDG;

    KOKKOS_INLINE_FUNCTION static
    void flux(double f[], const double[], const double[], const double[],
              const double[], const double[], double) {
        for (int k = 0; k < ncu*nd; ++k) f[k] = 0.0;
    }
    KOKKOS_INLINE_FUNCTION static
    void initu(double ui[], const double[], const double[], const double[]) {
        ui[0] = 0.0; ui[1] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION static
    void sourcew(double sw[], const double[], const double uq[], const double[],
                 const double w[], const double mu[], const double[], double) {
        sw[0] = mu[0]*uq[0] + 3.0*uq[4] + w[0]*w[1];
        sw[1] = uq[1]*uq[2] + 5.0*uq[5] + 2.0*w[0] + w[1]*w[1];
    }
    KOKKOS_INLINE_FUNCTION static
    void sourcew_jac_uq(double s[], const double[], const double uq[], const double[],
                        const double[], const double mu[], const double[], double) {
        constexpr int Nq = ncu*(1+nd);      // 6
        for (int k = 0; k < ncw*Nq; ++k) s[k] = 0.0;
        // layout [j*ncw + o] = d sw[o]/d uq[j]
        s[0*ncw+0] = mu[0];   // d sw0/d uq0
        s[1*ncw+1] = uq[2];   // d sw1/d uq1
        s[2*ncw+1] = uq[1];   // d sw1/d uq2
        s[4*ncw+0] = 3.0;     // d sw0/d uq4
        s[5*ncw+1] = 5.0;     // d sw1/d uq5
    }
    KOKKOS_INLINE_FUNCTION static
    void sourcew_jac_w(double s[], const double[], const double[], const double[],
                       const double w[], const double[], const double[], double) {
        // layout [j*ncw + o] = d sw[o]/d w[j]
        s[0*ncw+0] = w[1];      // d sw0/d w0
        s[0*ncw+1] = 2.0;       // d sw1/d w0
        s[1*ncw+0] = w[0];      // d sw0/d w1
        s[1*ncw+1] = 2.0*w[1];  // d sw1/d w1
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

// ncu=2, nd=2, Nq=nc=6, ncw=2, nco=0, ncx=2.
static constexpr int NG = 4, NCU = 2, ND = 2, NC = 6, NCW = 2, NCO = 0, NCX = 2;

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        std::vector<double> xdg(NCX*NG), udg(NC*NG), wdg(NCW*NG), odg(1, 0.0), param(1, 1.7);
        for (int i = 0; i < NG; ++i) {
            double s = 0.1 * (i + 1);
            for (int k = 0; k < NCX; ++k) xdg[k*NG+i] = 0.3 + s + 0.05*k;
            for (int k = 0; k < NC;  ++k) udg[k*NG+i] = 0.2 + 0.11*k + s;
            for (int k = 0; k < NCW; ++k) wdg[k*NG+i] = 1.3 + 0.4*k + s;
        }
        const int mn = 0; const double t = 0.0;
        const double* NULLP = nullptr;  // typed null so T deduces in the templated kernels

        // ---- HdgSourcew: value + f_udg + f_wdg ----
        std::vector<double> f_abi(NCW*NG, -7),  fudg_abi(NCW*NC*NG, -7),  fwdg_abi(NCW*NCW*NG, -7);
        std::vector<double> f_cxx(NCW*NG, 9),   fudg_cxx(NCW*NC*NG, 9),   fwdg_cxx(NCW*NCW*NG, 9);

        abi_probe::HdgSourcew(f_abi.data(), fudg_abi.data(), fwdg_abi.data(),
                 xdg.data(), udg.data(), odg.data(), wdg.data(), nullptr, param.data(),
                 t, mn, NG, NC, NCU, ND, NCX, NCO, NCW);
        exasim::hdg_sourcew_kernel<WProbe>(f_cxx.data(), fudg_cxx.data(), fwdg_cxx.data(),
                 xdg.data(), udg.data(), odg.data(), wdg.data(), NULLP, param.data(),
                 t, mn, NG, NC, NCU, ND, NCX, NCO, NCW);
        Kokkos::fence();
        cmp("Sourcew f",      f_abi.data(),    f_cxx.data(),    NCW*NG);
        cmp("Sourcew f_udg",  fudg_abi.data(), fudg_cxx.data(), NCW*NC*NG);
        cmp("Sourcew f_wdg",  fwdg_abi.data(), fwdg_cxx.data(), NCW*NCW*NG);

        // ---- HdgSourcewonly: value + f_wdg ----
        std::vector<double> g_abi(NCW*NG, -7),  gwdg_abi(NCW*NCW*NG, -7);
        std::vector<double> g_cxx(NCW*NG, 9),   gwdg_cxx(NCW*NCW*NG, 9);

        abi_probe::HdgSourcewonly(g_abi.data(), gwdg_abi.data(),
                 xdg.data(), udg.data(), odg.data(), wdg.data(), nullptr, param.data(),
                 t, mn, NG, NC, NCU, ND, NCX, NCO, NCW);
        exasim::hdg_sourcewonly_kernel<WProbe>(g_cxx.data(), gwdg_cxx.data(),
                 xdg.data(), udg.data(), odg.data(), wdg.data(), NULLP, param.data(),
                 t, mn, NG, NC, NCU, ND, NCX, NCO, NCW);
        Kokkos::fence();
        cmp("Sourcewonly f",     g_abi.data(),    g_cxx.data(),    NCW*NG);
        cmp("Sourcewonly f_wdg", gwdg_abi.data(), gwdg_cxx.data(), NCW*NCW*NG);
    }
    Kokkos::finalize();
    printf(nfail ? "\n%d comparisons FAILED\n" : "\nALL comparisons passed\n", nfail);
    return nfail ? 1 : 0;
}
