// Kernel-equivalence test for the multi-domain HDG interface-coupling surface
// (Fint / Fext) on the CONCRETE (templated) model path, driven by a
// text2code-GENERATED model — the generated-model analog of
// tests/coupling-models/compare_fint_fext.cpp (which used HAND-WRITTEN models).
//
// This closes Gap 1 of the concrete-model coupling work: text2code must EMIT the
// coupling surface into the generated `PdeModel` (has_external_coupling +
// nfint/nfext/ncuext + concrete fint/fext + Jacobians), so a generated coupled
// model runs pure-template with NO loaded ABI. We prove it byte-for-byte:
//
//   abi_*  — the libpdemodel ABI kernels text2code emitted from the model files,
//            checked in under tests/coupling-models/reference/,
//   gen_*  — the concrete `PdeModel` text2code emitted into `my_model.hpp` (the
//            templated path), checked in under tests/coupling-models/generated/,
//            driven here through the templated exasim::fint_kernel<M> /
//            fext_kernel<M> (same kernels the ABI-vs-hand-written test uses).
//
// Two GENERATED models are compared (both regenerated with the coupling-aware
// text2code — see tests/coupling-models/generated/README.md):
//   pde2   — apps/poisson/poisson2d/pdemodel2.txt verbatim (Fint has 2 output
//            components with ncu==1; Fext reads uext). Real app coverage.
//   probe  — probe_model.txt (pdemodel2 with non-zero uq/uhat derivatives) so
//            the test exercises the trace/input-index-outer Jacobian SoA layout
//            (J[(j*nf+o)*ng+i] = d f[o]/d input[j]) that the trivial pdemodel2
//            bodies cannot distinguish.
//
// The GENERATED concrete kernels must match the ABI kernels to 0.0 — that is the
// whole correctness bar for the emitted coupling surface. Gated at rtol 1e-13 to
// absorb any floating reassociation. Mirrors compare_fint_fext.cpp exactly except
// the models come from `#include`d generated headers instead of hand-written C++.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Pulls Kokkos + the Model contract (exasim::ModelDefaults) + dstype.
#include <exasim/kernels/boundary.hpp>

// ---- ABI reference kernels (text2code libpdemodel output), each in its own
// namespace so the duplicate HdgFint/HdgFint1/HdgFext symbol names do not
// collide. Reused verbatim from compare_fint_fext.cpp. ----
namespace abi_pde2 {
#include "reference/pde2_HdgFint.cpp"
#include "reference/pde2_HdgFext.cpp"
}
namespace abi_probe {
#include "reference/probe_HdgFint.cpp"
#include "reference/probe_HdgFext.cpp"
}

// ---- text2code-GENERATED concrete models. Each generated my_model.hpp defines a
// struct `PdeModel : ModelDefaults<PdeModel>`, so wrap each in its own namespace
// (and pull `exasim::ModelDefaults` into scope, since the generated header names
// it unqualified). The nested `#include <Kokkos_Core.hpp>` in each header is a
// no-op — boundary.hpp already included it at global scope. ----
namespace gen_pde2 {
using exasim::ModelDefaults;
#include "generated/pde2_my_model.hpp"
}
namespace gen_probe {
using exasim::ModelDefaults;
#include "generated/probe_my_model.hpp"
}

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
        printf("FAIL %-26s worst rel err %.3e at slot %d (abi=%.17g gen=%.17g)\n",
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
        // Sanity: text2code must have turned the coupling surface ON.
        static_assert(exasim::has_external_coupling_v<gen_pde2::PdeModel>,
                      "generated pde2 PdeModel must set has_external_coupling=true");
        static_assert(exasim::has_external_coupling_v<gen_probe::PdeModel>,
                      "generated probe PdeModel must set has_external_coupling=true");
        static_assert(exasim::nfint_v<gen_pde2::PdeModel> == 2, "pde2 Fint has 2 outputs");
        static_assert(exasim::nfext_v<gen_pde2::PdeModel> == 1, "pde2 Fext has 1 output");

        run_case<gen_pde2::PdeModel >("pde2 ", abi_pde2::HdgFint,  abi_pde2::HdgFext);
        run_case<gen_probe::PdeModel>("probe", abi_probe::HdgFint, abi_probe::HdgFext);
    }
    Kokkos::finalize();
    printf(nfail ? "\n%d comparisons FAILED\n" : "\nALL comparisons passed\n", nfail);
    return nfail ? 1 : 0;
}
