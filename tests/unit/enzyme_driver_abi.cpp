// CPU fixture for ABI dispatch and the real Kokkos interior-flux composition.
// The callbacks below are analytic references, not Enzyme-generated kernels.
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
using std::partial_sum;
#include "../../backend/Common/common.h"
#include "../../backend/Common/cpuimpl.h"
#include "../../backend/Common/kokkosimpl.h"
#include "../../backend/Model/driver_decls.hpp"
#include "../../backend/Model/ModelDispatch/model_drivers_abi.cpp"

static void check(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}

static std::vector<const dstype*> pointers;
static std::vector<int> dimensions;
static dstype callbackTime;
static void record(std::initializer_list<const dstype*> p,
                   std::initializer_list<int> d, dstype time)
{
    pointers = p;
    dimensions = d;
    callbackTime = time;
}

static void recordFlux(dstype* f, dstype* df, const dstype* xg,
                const dstype* udg, const dstype* dudg, const dstype* odg,
                const dstype* dodg, const dstype* wdg, const dstype* dwdg,
    const dstype* uinf, const dstype* param,
    dstype time, int model, int ng, int nc, int ncu,
    int nd, int ncx, int nco, int ncw)
{
    record({f, df, xg, udg, dudg, odg, dodg, wdg, dwdg, uinf, param},
           {model, ng, nc, ncu, nd, ncx, nco, ncw}, time);
    f[0] += 17;
    df[0] += 23;
}

static void recordSource(dstype* f, dstype* df, const dstype* xg,
                  const dstype* udg, const dstype* dudg, const dstype* odg,
                  const dstype* dodg, const dstype* wdg, const dstype* dwdg,
    const dstype* uinf, const dstype* param,
    dstype time, int model, int ng, int nc, int ncu,
    int nd, int ncx, int nco, int ncw)
{
    record({f, df, xg, udg, dudg, odg, dodg, wdg, dwdg, uinf, param},
           {model, ng, nc, ncu, nd, ncx, nco, ncw}, time);
    f[0] += 17;
    df[0] += 23;
}

static void recordAvfield(dstype* f, dstype* df, const dstype* xg,
                   const dstype* udg, const dstype* dudg, const dstype* odg,
                   const dstype* wdg, const dstype* dwdg,
    const dstype* uinf, const dstype* param,
    dstype time, int model, int ng, int nc, int ncu,
    int nd, int ncx, int nco, int ncw, int nce, int npe, int ne)
{
    record({f, df, xg, udg, dudg, odg, wdg, dwdg, uinf, param},
           {model, ng, nc, ncu, nd, ncx, nco, ncw, nce, npe, ne}, time);
    f[0] += 17;
    df[0] += 23;
}

static void recordFbou(dstype* f, dstype* df, const dstype* xg,
                const dstype* udg, const dstype* dudg, const dstype* odg,
                const dstype* dodg, const dstype* wdg, const dstype* dwdg,
                const dstype* uhg, const dstype* duhg, const dstype* nl,
    const dstype* tau, const dstype* uinf, const dstype* param,
    dstype time, int model, int ib, int ng, int nc, int ncu,
    int nd, int ncx, int nco, int ncw)
{
    record({f, df, xg, udg, dudg, odg, dodg, wdg, dwdg,
        uhg, duhg, nl, tau, uinf, param},
           {model, ib, ng, nc, ncu, nd, ncx, nco, ncw}, time);
    f[0] += 17;
    df[0] += 23;
}

static void recordUbou(dstype* f, dstype* df, const dstype* xg,
                const dstype* udg, const dstype* dudg, const dstype* odg,
                const dstype* wdg, const dstype* dwdg, const dstype* uhg,
                const dstype* duhg, const dstype* nl,
    const dstype* tau, const dstype* uinf, const dstype* param,
    dstype time, int model, int ib, int ng, int nc, int ncu,
    int nd, int ncx, int nco, int ncw)
{
    record({f, df, xg, udg, dudg, odg, wdg, dwdg, uhg, duhg, nl, tau, uinf, param},
           {model, ib, ng, nc, ncu, nd, ncx, nco, ncw}, time);
    f[0] += 17;
    df[0] += 23;
}

static void recordFhat(dstype* f, dstype* df, const dstype* xg,
                const dstype* ug1, const dstype* dug1,
                const dstype* ug2, const dstype* dug2,
                const dstype* og1, const dstype* dog1,
                const dstype* og2, const dstype* dog2,
                const dstype* wg1, const dstype* dwg1,
                const dstype* wg2, const dstype* dwg2,
                const dstype* uh, const dstype* duh, const dstype* nl,
    const dstype* tau, const dstype* uinf, const dstype* param,
    dstype time, int model, int ng, int nc, int ncu,
    int nd, int ncx, int nco, int ncw)
{
    record({f, df, xg, ug1, dug1, ug2, dug2, og1, dog1, og2, dog2,
        wg1, dwg1, wg2, dwg2, uh, duh, nl, tau, uinf, param},
           {model, ng, nc, ncu, nd, ncx, nco, ncw}, time);
    f[0] += 17;
    df[0] += 23;
}

// Flux in direction d: (d+1)*u_j^2 + q_{j,d} + o*w.
static void fluxJvp(dstype* f, dstype* df, const dstype*,
    const dstype* u, const dstype* du, const dstype* o, const dstype* dodg,
    const dstype* w, const dstype* dw, const dstype*, const dstype*,
    dstype, int, int ng, int, int ncu, int nd, int, int, int)
{
    Kokkos::parallel_for("analytic_flux_jvp", ng*ncu*nd, KOKKOS_LAMBDA(int k) {
        const int i = k % ng, j = (k/ng) % ncu, d = k/(ng*ncu);
        const int s = i + ng*j, q = i + ng*(ncu + d*ncu + j);
        f[k] = (d+1)*u[s]*u[s] + u[q] + o[i]*w[i];
        df[k] = 2*(d+1)*u[s]*du[s] + du[q] + dodg[i]*w[i] + o[i]*dw[i];
    });
}

// Independent primal callback: used through the unchanged primal FhatDriver
// for a central-difference check of the composed JVP.
static void flux(dstype* f, const dstype*, const dstype* u,
    const dstype* o, const dstype* w, const dstype*, const dstype*, dstype,
    int, int ng, int, int ncu, int nd, int, int, int)
{
    Kokkos::parallel_for("analytic_flux", ng, KOKKOS_LAMBDA(int i) {
        for (int d = 0; d < nd; ++d)
            for (int j = 0; j < ncu; ++j)
                f[i + ng*(j + ncu*d)] = (d+1)*u[i+ng*j]*u[i+ng*j]
                    + u[i+ng*(ncu+j+ncu*d)] + o[i]*w[i];
    });
}

static void run()
{
    ExasimDriverABI abi;
    check(!abi.volume.KokkosFluxJvp && !abi.volume.KokkosSourceJvp && !abi.volume.KokkosTdfuncJvp &&
          !abi.volume.KokkosAvfieldJvp && !abi.boundary.KokkosFbouJvp &&
          !abi.boundary.KokkosUbouJvp && !abi.iface.KokkosFhatJvp &&
          !abi.iface.KokkosStabJvp, "JVPs must default to null");
    commonstruct common;
    meshstruct mesh; masterstruct master; appstruct app; solstruct sol; tempstruct temp;
    common.driver_abi = &abi;
    common.modelnumber = 7;
    common.timestate.time = 0.25;
    common.components.nc = 6; common.components.ncu = 2;
    common.components.nco = 1; common.components.ncw = 1; common.components.ncx = 2;
    common.grid.nd = 2; common.grid.npe = 2; common.meshsizes.ne = 1;
    common.physicsparams.ncAV = 3; // Deliberately different from nco.
    common.couplingparams.extFhat = 1;
    common.couplingparams.extStab = 0;
    dstype f[16] = {}, df[16] = {}, xg[4] = {}, udg[12], dudg[12];
    dstype ug2[12], dug2[12], odg[2] = {2,3}, dodg[2] = {0.2,-0.1};
    dstype og2[2] = {1,4}, dog2[2] = {0.3,0.5};
    dstype wdg[2] = {3,2}, dwdg[2] = {-0.4,0.2};
    dstype wg2[2] = {2,1}, dwg2[2] = {-0.1,0.7};
    dstype uhg[4] = {}, duhg[4] = {}, nl[4] = {0.6,-0.8,0.8,0.6};
    dstype tau[4] = {2,3,-1,4}, uinf[1] = {}, param[1] = {};
    app.tau = tau; app.uinf = uinf; app.physicsparam = param;
    for (int i = 0; i < 12; ++i) {
        udg[i] = 0.5 + 0.1*i; dudg[i] = 0.2 - 0.03*i;
        ug2[i] = 0.8 - 0.02*i; dug2[i] = -0.1 + 0.04*i;
    }
    int end = 5; // Nonzero block start checks ng = nge*(end-start).
    auto callFlux = [&] {
        FluxDriver(f, df, xg, udg, dudg, odg, dodg, wdg, dwdg,
            mesh, master, app, sol, temp, common, 2, 4, end, 0);
    };
    abi.volume.KokkosFluxJvp = recordFlux;
    callFlux();
    check(pointers == std::vector<const dstype*>{f, df, xg, udg, dudg, odg, dodg, wdg, dwdg, uinf, param}
          && dimensions == std::vector<int>{7, 2, 6, 2, 2, 2, 1, 1} && callbackTime == 0.25,
          "Flux argument routing");
    abi.volume.KokkosFluxJvp = nullptr;
    bool missingFlux = false;
    try { callFlux(); }
    catch (const std::runtime_error& e) {
        missingFlux = std::string(e.what()).find("KokkosFluxJvp") != std::string::npos;
    }
    check(missingFlux, "Missing Flux JVP must be diagnosed");
    end = 4; common.meshsizes.ne = 0; pointers.clear();
    callFlux();
    check(pointers.empty(), "Empty Flux block must not dispatch");
    end = 5; common.meshsizes.ne = 1;

    auto callSource = [&] {
        SourceDriver(f, df, xg, udg, dudg, odg, dodg, wdg, dwdg,
            mesh, master, app, sol, temp, common, 2, 4, end, 0);
    };
    abi.volume.KokkosSourceJvp = recordSource;
    callSource();
    check(pointers == std::vector<const dstype*>{f, df, xg, udg, dudg, odg, dodg, wdg, dwdg, uinf, param}
          && dimensions == std::vector<int>{7, 2, 6, 2, 2, 2, 1, 1} && callbackTime == 0.25,
          "Source argument routing");
    abi.volume.KokkosSourceJvp = nullptr;
    bool missingSource = false;
    try { callSource(); }
    catch (const std::runtime_error& e) {
        missingSource = std::string(e.what()).find("KokkosSourceJvp") != std::string::npos;
    }
    check(missingSource, "Missing Source JVP must be diagnosed");
    end = 4; common.meshsizes.ne = 0; pointers.clear();
    callSource();
    check(pointers.empty(), "Empty Source block must not dispatch");
    end = 5; common.meshsizes.ne = 1;

    auto callAvfield = [&] {
        AvfieldDriver(f, df, xg, udg, dudg, odg, wdg, dwdg,
            mesh, master, app, sol, temp, common, 0);
    };
    abi.volume.KokkosAvfieldJvp = recordAvfield;
    callAvfield();
    check(pointers == std::vector<const dstype*>{f, df, xg, udg, dudg, odg, wdg, dwdg, uinf, param}
          && dimensions == std::vector<int>{7, 2, 6, 2, 2, 2, 1, 1, 3, 2, 1} && callbackTime == 0.25,
          "Avfield argument routing");
    abi.volume.KokkosAvfieldJvp = nullptr;
    bool missingAvfield = false;
    try { callAvfield(); }
    catch (const std::runtime_error& e) {
        missingAvfield = std::string(e.what()).find("KokkosAvfieldJvp") != std::string::npos;
    }
    check(missingAvfield, "Missing Avfield JVP must be diagnosed");
    end = 4; common.meshsizes.ne = 0; pointers.clear();
    callAvfield();
    check(pointers.empty(), "Empty Avfield block must not dispatch");
    end = 5; common.meshsizes.ne = 1;

    auto callFbou = [&] {
        FbouDriver(f, df, xg, udg, dudg, odg, dodg, wdg, dwdg,
        uhg, duhg, nl,
            mesh, master, app, sol, temp, common, 2, 4, end, 9, 0);
    };
    abi.boundary.KokkosFbouJvp = recordFbou;
    callFbou();
    check(pointers == std::vector<const dstype*>{f, df, xg, udg, dudg, odg, dodg, wdg, dwdg,
        uhg, duhg, nl, tau, uinf, param}
          && dimensions == std::vector<int>{7, 9, 2, 6, 2, 2, 2, 1, 1} && callbackTime == 0.25,
          "Fbou argument routing");
    abi.boundary.KokkosFbouJvp = nullptr;
    bool missingFbou = false;
    try { callFbou(); }
    catch (const std::runtime_error& e) {
        missingFbou = std::string(e.what()).find("KokkosFbouJvp") != std::string::npos;
    }
    check(missingFbou, "Missing Fbou JVP must be diagnosed");
    end = 4; common.meshsizes.ne = 0; pointers.clear();
    callFbou();
    check(pointers.empty(), "Empty Fbou block must not dispatch");
    end = 5; common.meshsizes.ne = 1;

    auto callUbou = [&] {
        UbouDriver(f, df, xg, udg, dudg, odg, wdg, dwdg, uhg, duhg, nl,
            mesh, master, app, sol, temp, common, 2, 4, end, 9, 0);
    };
    abi.boundary.KokkosUbouJvp = recordUbou;
    callUbou();
    check(pointers == std::vector<const dstype*>{f, df, xg, udg, dudg, odg, wdg, dwdg, uhg, duhg, nl, tau, uinf, param}
          && dimensions == std::vector<int>{7, 9, 2, 6, 2, 2, 2, 1, 1} && callbackTime == 0.25,
          "Ubou argument routing");
    abi.boundary.KokkosUbouJvp = nullptr;
    bool missingUbou = false;
    try { callUbou(); }
    catch (const std::runtime_error& e) {
        missingUbou = std::string(e.what()).find("KokkosUbouJvp") != std::string::npos;
    }
    check(missingUbou, "Missing Ubou JVP must be diagnosed");
    end = 4; common.meshsizes.ne = 0; pointers.clear();
    callUbou();
    check(pointers.empty(), "Empty Ubou block must not dispatch");
    end = 5; common.meshsizes.ne = 1;

    auto callFhat = [&] {
        FhatDriver(f, df, xg, udg, dudg, ug2, dug2, odg, dodg, og2, dog2,
            wdg, dwdg, wg2, dwg2, uhg, duhg, nl,
            mesh, master, app, sol, temp, common, 2, 4, end, 0);
    };
    abi.iface.KokkosFhatJvp = recordFhat;
    callFhat();
    check(pointers == std::vector<const dstype*>{f, df, xg, udg, dudg, ug2, dug2, odg, dodg, og2, dog2,
            wdg, dwdg, wg2, dwg2, uhg, duhg, nl, tau, uinf, param}
          && dimensions == std::vector<int>{7, 2, 6, 2, 2, 2, 1, 1} && callbackTime == 0.25,
          "Fhat argument routing");
    abi.iface.KokkosFhatJvp = nullptr;
    bool missingFhat = false;
    try { callFhat(); }
    catch (const std::runtime_error& e) {
        missingFhat = std::string(e.what()).find("KokkosFhatJvp") != std::string::npos;
    }
    check(missingFhat, "Missing Fhat JVP must be diagnosed");
    end = 4; common.meshsizes.ne = 0; pointers.clear();
    callFhat();
    check(pointers.empty(), "Empty Fhat block must not dispatch");
    end = 5; common.meshsizes.ne = 1;

    abi.volume.KokkosFlux = flux;
    abi.volume.KokkosFluxJvp = fluxJvp;
    common.couplingparams.extFhat = 0;
    const dstype eps = std::is_same_v<dstype, float> ? 1e-2 : 1e-5;
    const dstype tol = std::is_same_v<dstype, float> ? 2e-4 : 1e-9;
    // Central differences of the unchanged primal path, with all active fields
    // perturbed. Non-symmetric dense tau catches a transposed matrix layout.
    for (int ntau : {0, 1, 2, 4}) {
        common.components.ntau = ntau;
        callFhat();
        Kokkos::fence();
        dstype plus[16], minus[16], value[16];
        FhatDriver(value, xg, udg, ug2, odg, og2, wdg, wg2, uhg, nl,
                   mesh, master, app, sol, temp, common, 2, 4, end, 0);
        for (int sign : {-1, 1}) {
            dstype u1p[12], u2p[12], o1p[2], o2p[2], w1p[2], w2p[2];
            for (int i = 0; i < 12; ++i) {
                u1p[i] = udg[i] + sign*eps*dudg[i];
                u2p[i] = ug2[i] + sign*eps*dug2[i];
            }
            for (int i = 0; i < 2; ++i) {
                o1p[i] = odg[i] + sign*eps*dodg[i];
                o2p[i] = og2[i] + sign*eps*dog2[i];
                w1p[i] = wdg[i] + sign*eps*dwdg[i];
                w2p[i] = wg2[i] + sign*eps*dwg2[i];
            }
            FhatDriver(sign == 1 ? plus : minus, xg, u1p, u2p, o1p, o2p,
                       w1p, w2p, uhg, nl, mesh, master, app, sol, temp,
                       common, 2, 4, end, 0);
            Kokkos::fence();
        }
        for (int i = 0; i < 4; ++i) {
            check(std::abs(f[i] - value[i]) < tol, "Fhat primal regression");
            check(std::abs(df[i] - (plus[i]-minus[i])/(2*eps)) < tol,
                  "Fhat JVP disagrees with central differences");
        }
    }
    common.couplingparams.extStab = 1;
    bool missingStab = false;
    try { callFhat(); }
    catch (const std::runtime_error& e) {
        missingStab = std::string(e.what()).find("KokkosStabJvp") != std::string::npos;
    }
    check(missingStab, "Missing custom stabilization JVP must be diagnosed");
    abi.iface.KokkosStabJvp = recordFhat;
    callFhat();
    Kokkos::fence();
    check(pointers == std::vector<const dstype*>{f, df, xg, udg, dudg, ug2, dug2,
              odg, dodg, og2, dog2, wdg, dwdg, wg2, dwg2, uhg, duhg, nl, tau, uinf, param},
          "Custom stabilization argument routing");
    // Optional fields can be null when their component counts are zero.
    common.components.nco = common.components.ncw = 0;
    abi.volume.KokkosFluxJvp = recordFlux;
    FluxDriver(f, df, xg, udg, dudg, nullptr, nullptr, nullptr, nullptr,
               mesh, master, app, sol, temp, common, 2, 4, end, 0);
    check(pointers[5] == nullptr && pointers[6] == nullptr &&
          pointers[7] == nullptr && pointers[8] == nullptr &&
          dimensions[6] == 0 && dimensions[7] == 0, "Zero-sized optional fields");
}

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    int status = 0;
    try { run(); std::puts("Kokkos JVP driver ABI checks passed"); }
    catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); status = 1; }
    Kokkos::finalize();
    return status;
}
