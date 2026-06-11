// Kernel-equivalence test for built-in model 4 (CNS5air: 2D axisymmetric
// 5-species reacting compressible Navier-Stokes).
//
// Model 4 was originally generated from MATLAB symbolic sources; this branch
// replaced that with a text2code source (backend/Model/BuiltIn/pdemodel4.txt).
// This test proves the swap changed nothing numerically: it compiles
//
//   ref/  — the legacy MATLAB-generated kernels checked in at
//           backend/Model/BuiltIn/model4/ (kept as the reference), and
//   gen/  — the kernels text2code regenerated from pdemodel4.txt during THIS
//           build (the exact files compiled into libbuiltinmodel),
//
// into separate namespaces (Kokkos::parallel_for shimmed to a serial loop)
// and evaluates both at a physically meaningful state: the reactingsharb2
// reference scales (rho_inf = 0.00235 kg/m^3, v_inf = 6921 m/s,
// T_inf = 260.6 K, T_wall = 1400 K, Re = 8.99e5) with T ~ 4500 K, mixed
// species densities, and nonzero gradients, so every branch of the
// thermodynamics (NASA9 range switches), transport (Blottner/Gupta/Wilke),
// and Park kinetics is exercised away from trivial points.
//
// 35 comparisons cover every non-empty kernel: Flux/Source/Tdfunc, the EoS
// family (value, d/du, d/dw), Sourcew, Vis*, Ubou/Fbou, all 9 HDG boundary
// conditions, and the full analytic jacobians (HdgFlux 16x24, HdgSource 8x24,
// HdgEoS, HdgSourcew). Expected agreement is machine precision (~1e-13);
// the gate is rtol 2e-9 per slot to absorb CSE reassociation.
//
// The ref/ and gen/ include roots are symlinks created by tests/CMakeLists.txt
// in the build tree; the test target depends on the builtin codegen step.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using dstype = double;
namespace Kokkos {
template <class F> void parallel_for(const char*, size_t n, F f) {
    for (size_t i = 0; i < n; ++i) f(i);
}
}
#define KOKKOS_LAMBDA(...) [=](__VA_ARGS__)

namespace ref {
#include "ref/KokkosFlux.cpp"
#include "ref/KokkosSource.cpp"
#include "ref/KokkosTdfunc.cpp"
#include "ref/KokkosEoS.cpp"
#include "ref/KokkosEoSdu.cpp"
#include "ref/KokkosEoSdw.cpp"
#include "ref/KokkosSourcew.cpp"
#include "ref/KokkosVisScalars.cpp"
#include "ref/KokkosVisVectors.cpp"
#include "ref/KokkosVisTensors.cpp"
#include "ref/KokkosUbou.cpp"
#include "ref/KokkosFbou.cpp"
#include "ref/HdgFbouonly.cpp"
#include "ref/HdgSourcewonly.cpp"
#include "ref/HdgFlux.cpp"
#include "ref/HdgSource.cpp"
#include "ref/HdgSourcew.cpp"
#include "ref/HdgEoS.cpp"
}

namespace gen {
#include "gen/KokkosFlux.cpp"
#include "gen/KokkosSource.cpp"
#include "gen/KokkosTdfunc.cpp"
#include "gen/KokkosEoS.cpp"
#include "gen/KokkosEoSdu.cpp"
#include "gen/KokkosEoSdw.cpp"
#include "gen/KokkosSourcew.cpp"
#include "gen/KokkosVisScalars.cpp"
#include "gen/KokkosVisVectors.cpp"
#include "gen/KokkosVisTensors.cpp"
#include "gen/KokkosUbou.cpp"
#include "gen/KokkosFbou.cpp"
#include "gen/HdgFbouonly.cpp"
#include "gen/HdgSourcewonly.cpp"
#include "gen/HdgFlux.cpp"
#include "gen/HdgSource.cpp"
#include "gen/HdgSourcew.cpp"
#include "gen/HdgEoS.cpp"
}

static int nfail = 0;
static void cmp(const char* name, const double* a, const double* b, int n,
                double rtol = 2e-9) {
    double worst = 0.0; int wi = -1;
    for (int k = 0; k < n; ++k) {
        double scale = std::fmax(1.0, std::fmax(std::fabs(a[k]), std::fabs(b[k])));
        double err = std::fabs(a[k] - b[k]) / scale;
        if (err > worst) { worst = err; wi = k; }
    }
    if (worst > rtol) {
        ++nfail;
        printf("FAIL %-22s worst rel err %.3e at slot %d (ref=%.17g gen=%.17g)\n",
               name, worst, wi, a[wi], b[wi]);
    } else {
        printf("ok   %-22s n=%d  worst rel err %.3e\n", name, n, worst);
    }
}

int main() {
    double param[12] = {0.0023402724, 6921.0, 112099.61, 260.6, 1.801932e-05,
                        0.024602882, 1009.9704, 1.0, 181.99298, 0.7397093,
                        898869.96, 1400.0};
    double uinf[18] = {0.011, 0.13, 0.027, 0.61, 0.14, 0.83, -0.02, 0.95,
                       0.0, 0.05, 0.0, 0.767, 0.183,
                       0.21, 0.13, 0.08, 0.04, 0.06};
    double xdg[2] = {0.31, 0.67};
    double udg[24] = {
        0.021, 0.153, 0.031, 0.552, 0.123,       // rho_i
        0.262, -0.087, 0.514,                    // rho u_z, rho u_r, rhoE
        // q = -grad(u), z block
        0.013, -0.021, 0.008, 0.114, -0.052, 0.064, -0.033, 0.092,
        // q, r block
        -0.017, 0.026, -0.011, -0.083, 0.041, -0.046, 0.057, -0.071};
    double odg[1] = {2.3e-4};
    double wdg[1] = {17.268};                    // T ~ 4500 K
    double uhg[8] = {0.0205, 0.1493, 0.0322, 0.5611, 0.1187, 0.2507, -0.0911, 0.5288};
    double nlg[2] = {0.6, 0.8};
    double tauv[1] = {5.0};
    double time = 0.0;

    const int ng = 1, nc = 24, ncu = 8, nd = 2, ncx = 2, nco = 1, ncw = 1;
    const int npe = 1, ne = 1;

    double fa[1024], fb[1024], ja[4096], jb[4096], wa[64], wb[64];

    // ---- Flux
    std::memset(fa, 0, sizeof fa); std::memset(fb, 0, sizeof fb);
    ref::KokkosFlux(fa, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw);
    gen::KokkosFlux(fb, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw);
    cmp("KokkosFlux", fa, fb, 16);

    // ---- Source
    std::memset(fa, 0, sizeof fa); std::memset(fb, 0, sizeof fb);
    ref::KokkosSource(fa, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw);
    gen::KokkosSource(fb, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw);
    cmp("KokkosSource", fa, fb, 8);

    // ---- Tdfunc
    std::memset(fa, 0, sizeof fa); std::memset(fb, 0, sizeof fb);
    ref::KokkosTdfunc(fa, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw);
    gen::KokkosTdfunc(fb, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw);
    cmp("KokkosTdfunc", fa, fb, 8);

    // ---- EoS family (npe layout; npe=1, ne=1)
    std::memset(fa, 0, sizeof fa); std::memset(fb, 0, sizeof fb);
    ref::KokkosEoS(fa, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw, ncw, npe, ne);
    gen::KokkosEoS(fb, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw, ncw, npe, ne);
    cmp("KokkosEoS", fa, fb, 1);

    std::memset(fa, 0, sizeof fa); std::memset(fb, 0, sizeof fb);
    ref::KokkosEoSdu(fa, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw, ncw*ncu, npe, ne);
    gen::KokkosEoSdu(fb, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw, ncw*ncu, npe, ne);
    cmp("KokkosEoSdu", fa, fb, 8);

    std::memset(fa, 0, sizeof fa); std::memset(fb, 0, sizeof fb);
    ref::KokkosEoSdw(fa, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw, ncw*ncw, npe, ne);
    gen::KokkosEoSdw(fb, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw, ncw*ncw, npe, ne);
    cmp("KokkosEoSdw", fa, fb, 1);

    std::memset(fa, 0, sizeof fa); std::memset(fb, 0, sizeof fb);
    ref::KokkosSourcew(fa, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw, ncw, npe, ne);
    gen::KokkosSourcew(fb, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw, ncw, npe, ne);
    cmp("KokkosSourcew", fa, fb, 1);

    // ---- Vis*
    std::memset(fa, 0, sizeof fa); std::memset(fb, 0, sizeof fb);
    ref::KokkosVisScalars(fa, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw);
    gen::KokkosVisScalars(fb, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw);
    cmp("KokkosVisScalars", fa, fb, 8);

    std::memset(fa, 0, sizeof fa); std::memset(fb, 0, sizeof fb);
    ref::KokkosVisVectors(fa, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw);
    gen::KokkosVisVectors(fb, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw);
    cmp("KokkosVisVectors", fa, fb, 6);

    std::memset(fa, 0, sizeof fa); std::memset(fb, 0, sizeof fb);
    ref::KokkosVisTensors(fa, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw);
    gen::KokkosVisTensors(fb, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw);
    cmp("KokkosVisTensors", fa, fb, 4);

    // ---- Ubou/Fbou (ib = 1)
    std::memset(fa, 0, sizeof fa); std::memset(fb, 0, sizeof fb);
    ref::KokkosUbou(fa, xdg, udg, odg, wdg, uhg, nlg, tauv, uinf, param, time, 0, 1, ng, nc, ncu, nd, ncx, nco, ncw);
    gen::KokkosUbou(fb, xdg, udg, odg, wdg, uhg, nlg, tauv, uinf, param, time, 0, 1, ng, nc, ncu, nd, ncx, nco, ncw);
    cmp("KokkosUbou(ib=1)", fa, fb, 8);

    std::memset(fa, 0, sizeof fa); std::memset(fb, 0, sizeof fb);
    ref::KokkosFbou(fa, xdg, udg, odg, wdg, uhg, nlg, tauv, uinf, param, time, 0, 1, ng, nc, ncu, nd, ncx, nco, ncw);
    gen::KokkosFbou(fb, xdg, udg, odg, wdg, uhg, nlg, tauv, uinf, param, time, 0, 1, ng, nc, ncu, nd, ncx, nco, ncw);
    cmp("KokkosFbou(ib=1)", fa, fb, 8);

    // ---- FbouHdg: all 9 boundary conditions (inflow, outflow, isothermal
    // wall, slip, zero-gradient, noncatalytic, supercatalytic, and the two
    // partial-catalytic wall variants)
    for (int ib = 1; ib <= 9; ++ib) {
        std::memset(fa, 0, sizeof fa); std::memset(fb, 0, sizeof fb);
        ref::HdgFbouonly(fa, xdg, udg, odg, wdg, uhg, nlg, tauv, uinf, param, time, 0, ib, ng, nc, ncu, nd, ncx, nco, ncw);
        gen::HdgFbouonly(fb, xdg, udg, odg, wdg, uhg, nlg, tauv, uinf, param, time, 0, ib, ng, nc, ncu, nd, ncx, nco, ncw);
        char nm[64]; snprintf(nm, sizeof nm, "HdgFbouonly(ib=%d)", ib);
        cmp(nm, fa, fb, 8);
    }

    // ---- HdgSourcewonly (value + d/dw)
    std::memset(fa, 0, sizeof fa); std::memset(fb, 0, sizeof fb);
    std::memset(wa, 0, sizeof wa); std::memset(wb, 0, sizeof wb);
    ref::HdgSourcewonly(fa, wa, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw);
    gen::HdgSourcewonly(fb, wb, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw);
    cmp("HdgSourcewonly f", fa, fb, 1);
    cmp("HdgSourcewonly dw", wa, wb, 1);

    // ---- HdgSourcew (value + d/duq + d/dw)
    std::memset(fa, 0, sizeof fa); std::memset(fb, 0, sizeof fb);
    std::memset(ja, 0, sizeof ja); std::memset(jb, 0, sizeof jb);
    std::memset(wa, 0, sizeof wa); std::memset(wb, 0, sizeof wb);
    ref::HdgSourcew(fa, ja, wa, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw);
    gen::HdgSourcew(fb, jb, wb, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw);
    cmp("HdgSourcew f", fa, fb, 1);
    cmp("HdgSourcew duq", ja, jb, 24);
    cmp("HdgSourcew dw", wa, wb, 1);

    // ---- HdgEoS (value + d/duq + d/dw)
    std::memset(fa, 0, sizeof fa); std::memset(fb, 0, sizeof fb);
    std::memset(ja, 0, sizeof ja); std::memset(jb, 0, sizeof jb);
    std::memset(wa, 0, sizeof wa); std::memset(wb, 0, sizeof wb);
    ref::HdgEoS(fa, ja, wa, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw);
    gen::HdgEoS(fb, jb, wb, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw);
    cmp("HdgEoS f", fa, fb, 1);
    cmp("HdgEoS duq", ja, jb, 24);
    cmp("HdgEoS dw", wa, wb, 1);

    // ---- HdgFlux (value + jacobians)
    {
        static double Ja[16*24], Jb[16*24], Wa[16], Wb[16];
        std::memset(fa, 0, sizeof fa); std::memset(fb, 0, sizeof fb);
        std::memset(Ja, 0, sizeof Ja); std::memset(Jb, 0, sizeof Jb);
        std::memset(Wa, 0, sizeof Wa); std::memset(Wb, 0, sizeof Wb);
        ref::HdgFlux(fa, Ja, Wa, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw);
        gen::HdgFlux(fb, Jb, Wb, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw);
        cmp("HdgFlux f", fa, fb, 16);
        cmp("HdgFlux duq", Ja, Jb, 16*24);
        cmp("HdgFlux dw", Wa, Wb, 16);
    }

    // ---- HdgSource (value + jacobians)
    {
        static double Ja[8*24], Jb[8*24], Wa[8], Wb[8];
        std::memset(fa, 0, sizeof fa); std::memset(fb, 0, sizeof fb);
        std::memset(Ja, 0, sizeof Ja); std::memset(Jb, 0, sizeof Jb);
        std::memset(Wa, 0, sizeof Wa); std::memset(Wb, 0, sizeof Wb);
        ref::HdgSource(fa, Ja, Wa, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw);
        gen::HdgSource(fb, Jb, Wb, xdg, udg, odg, wdg, uinf, param, time, 0, ng, nc, ncu, nd, ncx, nco, ncw);
        cmp("HdgSource f", fa, fb, 8);
        cmp("HdgSource duq", Ja, Jb, 8*24);
        cmp("HdgSource dw", Wa, Wb, 8);
    }

    printf(nfail ? "\n%d comparisons FAILED\n" : "\nALL comparisons passed\n", nfail);
    return nfail ? 1 : 0;
}
