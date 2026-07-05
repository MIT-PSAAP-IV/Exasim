// Gap-3 regression: DISPATCH-level equivalence for the multi-domain HDG
// interface-coupling drivers (exasim::FintDriver<M> / FextDriver<M>).
//
// The sibling tests (compare_fint_fext.cpp, compare_generated_fint_fext.cpp)
// exercise the templated KERNELS `exasim::fint_kernel<M>` / `fext_kernel<M>`
// DIRECTLY. They do NOT go through the `exasim::FintDriver<M>` /
// `FextDriver<M>` overload set — and that overload set is exactly where Gap-3
// bites.
//
// Gap-3: the concrete assembly sites (backend/Discretization/uequation.hpp
// :409 and :1030, via EXASIM_DRIVER_CALL) call `FintDriver<M>` / `FextDriver<M>`
// with NON-CONST `dstype*` buffers (fhb, fhb_uq, ..., xgb, uhb, ...). A
// non-const `dstype*` lvalue binds the variadic precision-fallback's
// forwarding-ref `First&&` by identity, which OUTRANKS the typed overload's
// qualification conversion (`dstype* -> const dstype*`) on the later args.
// WITHOUT the SFINAE constraint on the fallback, the variadic therefore wins
// overload resolution for the default-precision concrete path and forwards to
// the (No-ABI/null) `multidomain_forward` -> the coupling residual is never
// assembled (SEGV / silent no-op in-repo). WITH the constraint, the typed
// `dstype*` overload owns the path and routes through `fext_kernel<M>`.
//
// This test drives `exasim::FextDriver<M>(...)` / `FintDriver<M>(...)` through
// the SAME argument types the assembly sites pass (non-const `dstype*`
// buffers + the real mesh/master/app/sol/temp/common structs), for a
// text2code-GENERATED coupled model (has_external_coupling=true), and asserts
// the dispatched result is byte-identical to the direct `fext_kernel<M>` /
// `fint_kernel<M>` output. Without the Gap-3 fix the typed overload is never
// selected and the dispatched buffers keep their sentinel -> this test FAILS.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Pulls the FintDriver<M>/FextDriver<M> overload set + the templated kernels +
// the mesh/app/common struct definitions (via <exasim/common.h>).
#include <exasim/drivers.hpp>

// text2code-GENERATED concrete coupled models (regenerated from pdemodel2.txt /
// probe_model.txt with coupling-aware text2code — same headers the
// generated_coupling_equivalence test uses).
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
        printf("FAIL %-30s worst rel err %.3e at slot %d (direct=%.17g dispatch=%.17g)\n",
               name, worst, wi, a[wi], b[wi]);
    } else {
        printf("ok   %-30s n=%d  worst rel err %.3e\n", name, n, worst);
    }
}

// Test state: ncu=1, nd=2, Nq=nc=3, ncw=0, nco=0, ncuext=1.
static constexpr int NG = 3, NCU = 1, ND = 2, NC = 3, NCW = 0, NCO = 0, NCX = 2, NUE = 1;

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
        for (int k = 0; k < NC;  ++k) udg[k*NG+i] = 0.2 + 0.11*k + s;
        uhg[0*NG+i]  = 0.37 + s;
        for (int k = 0; k < ND;  ++k) nlg[k*NG+i] = (k==0 ? 0.6 : 0.8);
        uext[0*NG+i] = 0.91 - s;
    }
}

// Build a minimal-but-real commonstruct/appstruct carrying only the fields the
// typed Fint/Fext drivers read (sizes + tau/physicsparam + time/modelnumber).
static void configure(commonstruct& common, appstruct& app) {
    common.components.nc  = NC;
    common.components.ncu = NCU;
    common.components.ncx = NCX;
    common.components.nco = NCO;
    common.components.ncw = NCW;
    common.grid.nd        = ND;
    common.modelnumber    = 0;
    common.timestate.time = 0.0;
    common.driver_abi     = nullptr;   // No-ABI: the typed path must NOT need this.
    app.tau         = tau.data();
    app.uinf        = nullptr;
    app.physicsparam = param.data();
}

template <class M>
static void run_case(const char* tag) {
    const int ib = 1;
    const int nfi = exasim::nfint_v<M>, nfe = exasim::nfext_v<M>;

    // Unused-by-driver structs (passed by reference, ignored in the body).
    meshstruct   mesh;
    masterstruct master;
    solstruct    sol;
    tempstruct   temp;
    commonstruct common;
    appstruct    app;
    configure(common, app);

    // ---- Fint: direct kernel (reference) vs typed dispatch overload ----
    {
        std::vector<double> f_ref(nfi*NG, -7), uq_ref(nfi*NC*NG, -7), uh_ref(nfi*NCU*NG, -7), w(1, 0);
        exasim::fint_kernel<M>(f_ref.data(), uq_ref.data(), w.data(), uh_ref.data(),
            xdg.data(), udg.data(), odg.data(), wdg.data(), uhg.data(), nlg.data(),
            tau.data(), nullptr, param.data(), 0.0, 0, ib, NG, NC, NCU, ND, NCX, NCO, NCW);

        // Non-const dstype* buffers, exactly like the assembly site (uequation.hpp:409).
        std::vector<double> f(nfi*NG, 1234.5), uq(nfi*NC*NG, 1234.5), uh(nfi*NCU*NG, 1234.5), wd(1, 0);
        std::vector<double> xmut(xdg), uhmut(uhg);   // xg / uhg are non-const in the overload
        exasim::FintDriver<M>(f.data(), uq.data(), wd.data(), uh.data(),
            xmut.data(), udg.data(), odg.data(), wdg.data(), uhmut.data(), nlg.data(),
            mesh, master, app, sol, temp, common, (Int)NG, (Int)ib, (Int)0);
        Kokkos::fence();

        char nm[80];
        snprintf(nm, sizeof nm, "%s Fint dispatch f",     tag); cmp(nm, f_ref.data(),  f.data(),  nfi*NG);
        snprintf(nm, sizeof nm, "%s Fint dispatch f_udg", tag); cmp(nm, uq_ref.data(), uq.data(), nfi*NC*NG);
        snprintf(nm, sizeof nm, "%s Fint dispatch f_uhg", tag); cmp(nm, uh_ref.data(), uh.data(), nfi*NCU*NG);
    }

    // ---- Fext: direct kernel (reference) vs typed dispatch overload ----
    {
        std::vector<double> f_ref(nfe*NG, -7), uq_ref(nfe*NC*NG, -7), uh_ref(nfe*NCU*NG, -7), w(1, 0);
        exasim::fext_kernel<M>(f_ref.data(), uq_ref.data(), w.data(), uh_ref.data(),
            xdg.data(), udg.data(), odg.data(), wdg.data(), uhg.data(), nlg.data(),
            uext.data(), tau.data(), nullptr, param.data(), 0.0, 0, ib, NG, NC, NCU, ND, NCX, NCO, NCW);

        std::vector<double> f(nfe*NG, 1234.5), uq(nfe*NC*NG, 1234.5), uh(nfe*NCU*NG, 1234.5), wd(1, 0);
        std::vector<double> xmut(xdg), uhmut(uhg);
        exasim::FextDriver<M>(f.data(), uq.data(), wd.data(), uh.data(),
            xmut.data(), udg.data(), odg.data(), wdg.data(), uhmut.data(), nlg.data(),
            uext.data(), mesh, master, app, sol, temp, common, (Int)NG, (Int)ib, (Int)0);
        Kokkos::fence();

        char nm[80];
        snprintf(nm, sizeof nm, "%s Fext dispatch f",     tag); cmp(nm, f_ref.data(),  f.data(),  nfe*NG);
        snprintf(nm, sizeof nm, "%s Fext dispatch f_udg", tag); cmp(nm, uq_ref.data(), uq.data(), nfe*NC*NG);
        snprintf(nm, sizeof nm, "%s Fext dispatch f_uhg", tag); cmp(nm, uh_ref.data(), uh.data(), nfe*NCU*NG);
    }
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        fill_inputs();
        static_assert(exasim::has_external_coupling_v<gen_pde2::PdeModel>,
                      "generated pde2 PdeModel must set has_external_coupling=true");
        static_assert(exasim::has_external_coupling_v<gen_probe::PdeModel>,
                      "generated probe PdeModel must set has_external_coupling=true");
        run_case<gen_pde2::PdeModel >("pde2 ");
        run_case<gen_probe::PdeModel>("probe");
    }
    Kokkos::finalize();
    printf(nfail ? "\n%d comparisons FAILED\n" : "\nALL comparisons passed\n", nfail);
    return nfail ? 1 : 0;
}
