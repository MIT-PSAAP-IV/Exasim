// Templated interface-flux SAMPLER equivalence (Gap 2).
//
// CInterfaceSampler::getInterfaceFluxesAt{Nodal,Gauss}Points extracts the HDG
// interface flux the coupled Dirichlet-Neumann driver reads every iteration.
// Before Gap 2 that extraction was ABI-only: it called the loaded-libpdemodel
// FintDriver unconditionally, so a pure-template coupled solve still needed the
// ABI just to read fluxes. Gap 2 adds a templated path
// (getInterfaceFluxesAt*For<M>) that, for a concrete Model M, routes through
// exasim::FintDriver<M> / fint_only_kernel<M> — no loaded ABI — while keeping the
// M == AbiAdapter path byte-identical.
//
// The sampler is thin: everything it does before the driver call — sampling the
// interface field traces (getUDG/ODG/WDG/UHAT + interpolating to Gauss points) —
// is MODEL-FREE, shared verbatim by both the ABI and concrete branches. The ONLY
// model-dependent operation is the final value-only FintDriver call, which
// bottoms out in:
//   - ABI branch      : driver_abi->hdgjac.HdgFintonly  (the checked-in reference)
//   - concrete branch : exasim::FintDriver<M> -> exasim::fint_only_kernel<M>
// So the sampler's concrete-vs-ABI flux equivalence reduces EXACTLY to
// fint_only_kernel<M> == HdgFintonly on the same sampled interface-Gauss buffers.
// This test proves that to 0.0, on the interface-Gauss SoA layout (k*ng+i) the
// sampler produces, for the text2code-GENERATED coupled models (so it also
// covers the Gap 1 codegen end-to-end):
//   pde2   — apps/poisson/poisson2d/pdemodel2.txt (Fint has 2 outputs, ncu==1),
//   probe  — non-trivial Fint/Fext (locks the layout).
//
// (The templated sampler methods themselves are compiled — the AbiAdapter
// instantiation in the backend unity build, the concrete-M instantiation in the
// Phase-3 pure-template consumer; both share this exact driver call, which is
// what is validated numerically here.)

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Pulls Kokkos + the Model contract (exasim::ModelDefaults) + dstype +
// exasim::fint_only_kernel<M>.
#include <exasim/kernels/boundary.hpp>

// ---- ABI reference value-only interface-flux kernels (text2code libpdemodel
// output — the exact HdgFintonly the ABI sampler's FintDriver calls), each in its
// own namespace. Checked in under tests/coupling-models/reference/. ----
namespace abi_pde2  {
#include "reference/pde2_HdgFintonly.cpp"
}
namespace abi_probe {
#include "reference/probe_HdgFintonly.cpp"
}

// ---- text2code-GENERATED concrete coupled models (see generated/README.md). ----
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
        printf("FAIL %-30s worst rel err %.3e at slot %d (abi=%.17g sampler=%.17g)\n",
               name, worst, wi, a[wi], b[wi]);
    } else {
        printf("ok   %-30s n=%d  worst rel err %.3e\n", name, n, worst);
    }
}

// Interface-Gauss-point state (the layout the sampler hands the driver): SoA over
// NG face-quadrature points, ncu=1, nd=2 -> nc=3, ncw=0, nco=0.
static constexpr int NG = 4, NCU = 1, ND = 2, NC = 3, NCW = 0, NCO = 0, NCX = 2;

static std::vector<double> xdggint, udggint, odggint, wdggint, uhgint, nlgint, tau, param;
static void fill_interface_gauss() {
    xdggint.assign(NCX*NG, 0.0);
    udggint.assign(NC*NG, 0.0);
    odggint.assign(1, 0.0);   // nco==0
    wdggint.assign(1, 0.0);   // ncw==0
    uhgint.assign(NCU*NG, 0.0);
    nlgint.assign(ND*NG, 0.0);
    tau.assign(NCU, 3.0);
    param.assign(1, 1.3);
    for (int i = 0; i < NG; ++i) {
        double s = 0.07 * (i + 1);
        for (int k = 0; k < NCX; ++k) xdggint[k*NG+i] = 0.2 + s + 0.04*k;
        for (int k = 0; k < NC;  ++k) udggint[k*NG+i] = 0.15 + 0.13*k + s;   // uq0,uq1,uq2
        uhgint[0*NG+i] = 0.41 + s;
        nlgint[0*NG+i] = 0.6; nlgint[1*NG+i] = 0.8;
    }
}

// ABI HdgFintonly signature (value-only interface flux).
using AbiFintonly = void(*)(dstype*, const dstype*, const dstype*, const dstype*, const dstype*,
                            const dstype*, const dstype*, const dstype*, const dstype*, const dstype*,
                            const dstype, const int, const int, const int, const int, const int,
                            const int, const int, const int, const int);

template <class M>
static void run_case(const char* tag, AbiFintonly abi_fintonly) {
    const int ib = 1, mn = 0; const double t = 0.0;
    const int nf = exasim::nfint_v<M>;

    // ABI sampler branch: driver_abi->hdgjac.HdgFintonly on the sampled buffers.
    std::vector<double> flux_abi(nf*NG, -7);
    abi_fintonly(flux_abi.data(), xdggint.data(), udggint.data(), odggint.data(), wdggint.data(),
                 uhgint.data(), nlgint.data(), tau.data(), nullptr, param.data(), t, mn, ib,
                 NG, NC, NCU, ND, NCX, NCO, NCW);

    // Concrete sampler branch: exasim::FintDriver<M> value-only -> fint_only_kernel<M>
    // on the SAME sampled buffers (identical field sampling, model-free).
    std::vector<double> flux_gen(nf*NG, 9);
    exasim::fint_only_kernel<M>(flux_gen.data(), xdggint.data(), udggint.data(), odggint.data(),
                 wdggint.data(), uhgint.data(), nlgint.data(), tau.data(), nullptr, param.data(),
                 t, mn, ib, NG, NC, NCU, ND, NCX, NCO, NCW);
    Kokkos::fence();

    char nm[80];
    snprintf(nm, sizeof nm, "%s interface flux (value)", tag);
    cmp(nm, flux_abi.data(), flux_gen.data(), nf*NG);
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        fill_interface_gauss();
        static_assert(exasim::has_external_coupling_v<gen_pde2::PdeModel>);
        static_assert(exasim::has_external_coupling_v<gen_probe::PdeModel>);
        run_case<gen_pde2::PdeModel >("pde2 ", abi_pde2::HdgFintonly);
        run_case<gen_probe::PdeModel>("probe", abi_probe::HdgFintonly);
    }
    Kokkos::finalize();
    printf(nfail ? "\n%d comparisons FAILED\n" : "\nALL comparisons passed\n", nfail);
    return nfail ? 1 : 0;
}
