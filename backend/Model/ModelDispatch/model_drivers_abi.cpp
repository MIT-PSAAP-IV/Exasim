/**
 * @file model_drivers_abi.cpp
 * @brief Core-side driver wrappers built on top of ExasimDriverabi.
 *
 * These wrappers preserve the legacy Exasim driver argument mapping while
 * dispatching the low-level kernel calls through a selected provider abi.
 * They intentionally keep the existing LDG/HDG data flow unchanged.
 */

#ifndef __EXASIM_MODEL_DRIVERS_ABI
#define __EXASIM_MODEL_DRIVERS_ABI

#include <cstdio>
#include <cstdlib>
#include "driver_abi.h"

void FluxDriver(dstype* f, const dstype* xg, const dstype* udg,
                const dstype* odg, const dstype* wdg,
                ExasimDriverABI& abi, meshstruct& mesh, masterstruct& master, appstruct& app,
                solstruct& sol, tempstruct& temp, commonstruct& common,
                Int nge, Int e1, Int e2, Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int numPoints = nge * (e2 - e1);
    dstype time = common.timestate.time;

    abi.volume.KokkosFlux(f, xg, udg, odg, wdg, app.uinf, app.physicsparam, time,
                   common.modelnumber, numPoints, nc, ncu, nd, ncx, nco, ncw);
}

void SourceDriver(dstype* f, const dstype* xg, const dstype* udg,
                  const dstype* odg, const dstype* wdg,
                  ExasimDriverABI& abi, meshstruct& mesh, masterstruct& master, appstruct& app,
                  solstruct& sol, tempstruct& temp, commonstruct& common,
                  Int nge, Int e1, Int e2, Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int numPoints = nge * (e2 - e1);
    dstype time = common.timestate.time;

    abi.volume.KokkosSource(f, xg, udg, odg, wdg, app.uinf, app.physicsparam, time,
                     common.modelnumber, numPoints, nc, ncu, nd, ncx, nco,
                     ncw);
}

void SourcewDriver(dstype* f, const dstype* xg, const dstype* udg,
                   const dstype* odg, const dstype* wdg,
                   ExasimDriverABI& abi, meshstruct& mesh, masterstruct& master, appstruct& app,
                   solstruct& sol, tempstruct& temp, commonstruct& common,
                   Int npe, Int e1, Int e2, Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int ne = e2 - e1;
    Int numPoints = npe * ne;
    dstype time = common.timestate.time;

    abi.volume.KokkosSourcew(f, xg, udg, odg, wdg, app.uinf, app.physicsparam, time,
                      common.modelnumber, numPoints, nc, ncu, nd, ncx, nco,
                      ncw, ncw, npe, ne);
}

void OutputDriver(dstype* f, const dstype* xg, const dstype* udg,
                  const dstype* odg, const dstype* wdg,
                  ExasimDriverABI& abi, meshstruct& mesh, masterstruct& master, appstruct& app,
                  solstruct& sol, tempstruct& temp, commonstruct& common,
                  Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nce = common.components.nce;
    Int nd = common.grid.nd;
    Int npe = common.grid.npe;
    Int ne = common.meshsizes.ne;
    Int numPoints = npe * ne;
    dstype time = common.timestate.time;

    abi.output.KokkosOutput(f, xg, udg, odg, wdg, app.uinf, app.physicsparam, time,
                     common.modelnumber, numPoints, nc, ncu, nd, ncx, nco,
                     ncw, nce, npe, ne);
}

void MonitorDriver(dstype* f, Int nc_sol, const dstype* xg,
                   const dstype* udg, const dstype* odg, const dstype* wdg,
                   ExasimDriverABI& abi, meshstruct& mesh,
                   masterstruct& master, appstruct& app, solstruct& sol,
                   tempstruct& temp, commonstruct& common, Int backend)
{
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int ncm = common.components.ncm;
    Int nd = common.grid.nd;
    Int npe = common.grid.npe;
    Int ne = common.meshsizes.ne;
    Int numPoints = npe * ne;
    dstype time = common.timestate.time;

    abi.output.KokkosMonitor(f, xg, udg, odg, wdg, app.uinf, app.physicsparam, time,
                      common.modelnumber, numPoints, nc_sol, ncu, nd, ncx,
                      nco, ncw, ncm, npe, ne);
}

void AvfieldDriver(dstype* f, const dstype* xg, const dstype* udg,
                   const dstype* odg, const dstype* wdg,
                   ExasimDriverABI& abi, meshstruct& mesh, masterstruct& master, appstruct& app,
                   solstruct& sol, tempstruct& temp, commonstruct& common,
                   Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int npe = common.grid.npe;
    Int ne = common.meshsizes.ne;
    Int numPoints = npe * ne;
    dstype time = common.timestate.time;

    abi.volume.KokkosAvfield(f, xg, udg, odg, wdg, app.uinf, app.physicsparam, time,
                      common.modelnumber, numPoints, nc, ncu, nd, ncx, nco,
                      ncw, common.physicsparams.ncAV, npe, ne);
}

void EosDriver(dstype* f, const dstype* xg, const dstype* udg,
               const dstype* odg, const dstype* wdg,
               ExasimDriverABI& abi, meshstruct& mesh, masterstruct& master, appstruct& app,
               solstruct& sol, tempstruct& temp, commonstruct& common,
               Int npe, Int e1, Int e2, Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int ne = e2 - e1;
    Int numPoints = npe * ne;
    dstype time = common.timestate.time;

    abi.eos.KokkosEoS(f, xg, udg, odg, wdg, app.uinf, app.physicsparam, time,
                  common.modelnumber, numPoints, nc, ncu, nd, ncx, nco, ncw,
                  ncw, npe, ne);
}

void EosduDriver(dstype* f, const dstype* xg, const dstype* udg,
                 const dstype* odg, const dstype* wdg,
                 ExasimDriverABI& abi, meshstruct& mesh, masterstruct& master, appstruct& app,
                 solstruct& sol, tempstruct& temp, commonstruct& common,
                 Int npe, Int e1, Int e2, Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int ne = e2 - e1;
    Int numPoints = npe * ne;
    dstype time = common.timestate.time;

    abi.eos.KokkosEoSdu(f, xg, udg, odg, wdg, app.uinf, app.physicsparam, time,
                    common.modelnumber, numPoints, nc, ncu, nd, ncx, nco, ncw,
                    ncw * ncu, npe, ne);
}

void EosdwDriver(dstype* f, const dstype* xg, const dstype* udg,
                 const dstype* odg, const dstype* wdg,
                 ExasimDriverABI& abi, meshstruct& mesh, masterstruct& master, appstruct& app,
                 solstruct& sol, tempstruct& temp, commonstruct& common,
                 Int npe, Int e1, Int e2, Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int ne = e2 - e1;
    Int numPoints = npe * ne;
    dstype time = common.timestate.time;

    abi.eos.KokkosEoSdw(f, xg, udg, odg, wdg, app.uinf, app.physicsparam, time,
                    common.modelnumber, numPoints, nc, ncu, nd, ncx, nco, ncw,
                    ncw * ncw, npe, ne);
}

void TdfuncDriver(dstype* f, const dstype* xg, const dstype* udg,
                  const dstype* odg, const dstype* wdg,
                  ExasimDriverABI& abi, meshstruct& mesh, masterstruct& master, appstruct& app,
                  solstruct& sol, tempstruct& temp, commonstruct& common,
                  Int nge, Int e1, Int e2, Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int numPoints = nge * (e2 - e1);
    dstype time = common.timestate.time;

    abi.volume.KokkosTdfunc(f, xg, udg, odg, wdg, app.uinf, app.physicsparam, time,
                     common.modelnumber, numPoints, nc, ncu, nd, ncx, nco,
                     ncw);
}

void FhatDriver(dstype* fg, const dstype* xg, const dstype* ug1,
                const dstype* ug2, const dstype* og1, const dstype* og2,
                const dstype* wg1, const dstype* wg2, const dstype* uh,
                const dstype* nl, ExasimDriverABI& abi, meshstruct& mesh,
                masterstruct& master, appstruct& app, solstruct& sol,
                tempstruct& tmp, commonstruct& common, Int ngf, Int f1, Int f2,
                Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int numPoints = ngf * (f2 - f1);
    Int M = numPoints * ncu;
    Int N = numPoints * ncu * nd;
    Int ntau = common.components.ntau;
    dstype time = common.timestate.time;

    if (common.couplingparams.extFhat == 1) {
        abi.iface.KokkosFhat(fg, xg, ug1, ug2, og1, og2, wg1, wg2, uh, nl, app.tau,
                       app.uinf, app.physicsparam, time, common.modelnumber,
                       numPoints, nc, ncu, nd, ncx, nco, ncw);
    }
    else {
        FluxDriver(fg, xg, ug1, og1, wg1, abi, mesh, master, app, sol, tmp,
                   common, ngf, f1, f2, backend);
        dstype* fg2 = &fg[N];
        FluxDriver(fg2, xg, ug2, og2, wg2, abi, mesh, master, app, sol, tmp,
                   common, ngf, f1, f2, backend);

        AverageFlux(fg, N);
        AverageFluxDotNormal(fg, nl, N, M, numPoints, nd);

        if (common.couplingparams.extStab >= 1) {
            abi.iface.KokkosStab(fg, xg, ug1, ug2, og1, og2, wg1, wg2, uh, nl,
                           app.tau, app.uinf, app.physicsparam, time,
                           common.modelnumber, numPoints, nc, ncu, nd, ncx,
                           nco, ncw);
        }
        else if (ntau == 0) {
            AddStabilization1(fg, ug1, ug2, app.tau, M);
        }
        else if (ntau == 1) {
            AddStabilization1(fg, ug1, ug2, app.tau, M);
        }
        else if (ntau == ncu) {
            AddStabilization2(fg, ug1, ug2, app.tau, M, numPoints);
        }
        else if (ntau == ncu * ncu) {
            AddStabilization3(fg, ug1, ug2, app.tau, M, numPoints, ncu);
        }
        else {
            printf("Stabilization option is not implemented");
            exit(-1);
        }
    }
}

void FbouDriver(dstype* fb, const dstype* xg, const dstype* udg,
                const dstype* odg, const dstype* wdg, const dstype* uhg,
                const dstype* nl, ExasimDriverABI& abi, meshstruct& mesh,
                masterstruct& master, appstruct& app, solstruct& sol,
                tempstruct& temp, commonstruct& common, Int ngf, Int f1,
                Int f2, Int ib, Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int numPoints = ngf * (f2 - f1);
    dstype time = common.timestate.time;

    abi.boundary.KokkosFbou(fb, xg, udg, odg, wdg, uhg, nl, app.tau, app.uinf,
                   app.physicsparam, time, common.modelnumber, ib, numPoints,
                   nc, ncu, nd, ncx, nco, ncw);
}

void FbouJacDriver(dstype* fb, dstype* fb_udg, dstype* fb_wdg,
                   dstype* fb_uhg, const dstype* xg, const dstype* udg,
                   const dstype* odg, const dstype* wdg, const dstype* uhg,
                   const dstype* nl, ExasimDriverABI& abi, meshstruct& mesh,
                   masterstruct& master, appstruct& app, solstruct& sol,
                   tempstruct& temp, commonstruct& common, Int nga, Int ib,
                   Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int numPoints = nga;
    dstype time = common.timestate.time;

    abi.boundary.KokkosFbouJac(fb, fb_udg, fb_wdg, fb_uhg, xg, udg, odg, wdg, uhg,
                      nl, app.tau, app.uinf, app.physicsparam, time,
                      common.modelnumber, ib, numPoints, nc, ncu, nd, ncx,
                      nco, ncw);
}

void FbouJacDriver(dstype* fb, dstype* fb_udg, dstype* fb_wdg,
                   dstype* fb_uhg, const dstype* xg, const dstype* udg,
                   const dstype* odg, const dstype* wdg, const dstype* uhg,
                   const dstype* nl, ExasimDriverABI& abi, meshstruct& mesh,
                   masterstruct& master, appstruct& app, solstruct& sol,
                   tempstruct& temp, commonstruct& common, Int ngf, Int f1,
                   Int f2, Int ib, Int backend)
{
    Int numPoints = ngf * (f2 - f1);
    FbouJacDriver(fb, fb_udg, fb_wdg, fb_uhg, xg, udg, odg, wdg, uhg, nl,
                  abi, mesh, master, app, sol, temp, common, numPoints, ib,
                  backend);
}

void UhatDriver(dstype* fg, dstype* xg, dstype* ug1, dstype* ug2,
                const dstype* og1, const dstype* og2, const dstype* wg1,
                const dstype* wg2, const dstype* uh, const dstype* nl,
                ExasimDriverABI& abi, meshstruct& mesh, masterstruct& master,
                appstruct& app, solstruct& sol, tempstruct& tmp,
                commonstruct& common, Int ngf, Int f1, Int f2, Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int numPoints = ngf * (f2 - f1);
    dstype time = common.timestate.time;

    if (common.couplingparams.extUhat == 1) {
        abi.iface.KokkosUhat(fg, xg, ug1, ug2, og1, og2, wg1, wg2, uh, nl, app.tau,
                       app.uinf, app.physicsparam, time, common.modelnumber,
                       numPoints, nc, ncu, nd, ncx, nco, ncw);
    }
    else {
        ArrayAXPBY(fg, ug1, ug2, (dstype)0.5, (dstype)0.5,
                   ngf * common.components.ncu * (f2 - f1));
    }
}

void UbouDriver(dstype* ub, const dstype* xg, const dstype* udg,
                const dstype* odg, const dstype* wdg, const dstype* uhg,
                const dstype* nl, ExasimDriverABI& abi, meshstruct& mesh,
                masterstruct& master, appstruct& app, solstruct& sol,
                tempstruct& temp, commonstruct& common, Int ngf, Int f1,
                Int f2, Int ib, Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int numPoints = ngf * (f2 - f1);
    dstype time = common.timestate.time;

    abi.boundary.KokkosUbou(ub, xg, udg, odg, wdg, uhg, nl, app.tau, app.uinf,
                   app.physicsparam, time, common.modelnumber, ib, numPoints,
                   nc, ncu, nd, ncx, nco, ncw);
}

void UbouJacDriver(dstype* ub, dstype* ub_udg, dstype* ub_wdg,
                   dstype* ub_uhg, const dstype* xg, const dstype* udg,
                   const dstype* odg, const dstype* wdg, const dstype* uhg,
                   const dstype* nl, ExasimDriverABI& abi, meshstruct& mesh,
                   masterstruct& master, appstruct& app, solstruct& sol,
                   tempstruct& temp, commonstruct& common, Int nga, Int ib,
                   Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int numPoints = nga;
    dstype time = common.timestate.time;

    abi.boundary.KokkosUbouJac(ub, ub_udg, ub_wdg, ub_uhg, xg, udg, odg, wdg, uhg,
                      nl, app.tau, app.uinf, app.physicsparam, time,
                      common.modelnumber, ib, numPoints, nc, ncu, nd, ncx,
                      nco, ncw);
}

void UbouJacDriver(dstype* ub, dstype* ub_udg, dstype* ub_wdg,
                   dstype* ub_uhg, const dstype* xg, const dstype* udg,
                   const dstype* odg, const dstype* wdg, const dstype* uhg,
                   const dstype* nl, ExasimDriverABI& abi, meshstruct& mesh,
                   masterstruct& master, appstruct& app, solstruct& sol,
                   tempstruct& temp, commonstruct& common, Int ngf, Int f1,
                   Int f2, Int ib, Int backend)
{
    Int numPoints = ngf * (f2 - f1);
    UbouJacDriver(ub, ub_udg, ub_wdg, ub_uhg, xg, udg, odg, wdg, uhg, nl,
                  abi, mesh, master, app, sol, temp, common, numPoints, ib,
                  backend);
}

void InitodgDriver(dstype* f, const dstype* xg,
                   ExasimDriverABI& abi, appstruct& app, Int ncx, Int nco, Int npe, Int ne,
                   Int backend)
{
    Int numPoints = npe * ne;
    Int modelnumber = app.modelnumber;

    abi.init.KokkosInitodg(f, xg, app.uinf, app.physicsparam, modelnumber,
                      numPoints, ncx, nco, npe, ne);
}

void InitqDriver(dstype* f, const dstype* xg,
                 ExasimDriverABI& abi, appstruct& app, Int ncx, Int nc, Int npe, Int ne,
                 Int backend)
{
    Int numPoints = npe * ne;
    Int modelnumber = app.modelnumber;

    abi.init.KokkosInitq(f, xg, app.uinf, app.physicsparam, modelnumber, numPoints,
                    ncx, nc, npe, ne);
}

void InitudgDriver(dstype* f, const dstype* xg,
                   ExasimDriverABI& abi, appstruct& app, Int ncx, Int nc, Int npe, Int ne,
                   Int backend)
{
    Int numPoints = npe * ne;
    Int modelnumber = app.modelnumber;

    abi.init.KokkosInitudg(f, xg, app.uinf, app.physicsparam, modelnumber,
                      numPoints, ncx, nc, npe, ne);
}

void InituDriver(dstype* f, const dstype* xg,
                 ExasimDriverABI& abi, appstruct& app, Int ncx, Int nc, Int npe, Int ne,
                 Int backend)
{
    Int numPoints = npe * ne;
    Int modelnumber = app.modelnumber;

    abi.init.KokkosInitu(f, xg, app.uinf, app.physicsparam, modelnumber, numPoints,
                    ncx, nc, npe, ne);
}

void InitwdgDriver(dstype* f, const dstype* xg,
                   ExasimDriverABI& abi, appstruct& app, Int ncx, Int ncw, Int npe, Int ne,
                   Int backend)
{
    Int numPoints = npe * ne;
    Int modelnumber = app.modelnumber;

    abi.init.KokkosInitwdg(f, xg, app.uinf, app.physicsparam, modelnumber,
                      numPoints, ncx, ncw, npe, ne);
}

void cpuInitodgDriver(dstype* f, const dstype* xg,
                      ExasimDriverABI& abi, appstruct& app, Int ncx, Int nco, Int npe, Int ne,
                      Int backend)
{
    Int numPoints = npe * ne;
    Int modelnumber = app.modelnumber;

    abi.init.cpuInitodg(f, xg, app.uinf, app.physicsparam, modelnumber, numPoints,
                   ncx, nco, npe, ne);
}

void cpuInitqDriver(dstype* f, const dstype* xg,
                    ExasimDriverABI& abi, appstruct& app, Int ncx, Int nc, Int npe, Int ne,
                    Int backend)
{
    Int numPoints = npe * ne;
    Int modelnumber = app.modelnumber;

    abi.init.cpuInitq(f, xg, app.uinf, app.physicsparam, modelnumber, numPoints,
                 ncx, nc, npe, ne);
}

void cpuInitudgDriver(dstype* f, const dstype* xg,
                      ExasimDriverABI& abi, appstruct& app, Int ncx, Int nc, Int npe, Int ne,
                      Int backend)
{
    Int numPoints = npe * ne;
    Int modelnumber = app.modelnumber;

    abi.init.cpuInitudg(f, xg, app.uinf, app.physicsparam, modelnumber, numPoints,
                   ncx, nc, npe, ne);
}

void cpuInituDriver(dstype* f, const dstype* xg,
                    ExasimDriverABI& abi, appstruct& app, Int ncx, Int nc, Int npe, Int ne,
                    Int backend)
{
    Int numPoints = npe * ne;
    Int modelnumber = app.modelnumber;

    abi.init.cpuInitu(f, xg, app.uinf, app.physicsparam, modelnumber, numPoints,
                 ncx, nc, npe, ne);
}

void cpuInitwdgDriver(dstype* f, const dstype* xg,
                      ExasimDriverABI& abi, appstruct& app, Int ncx, Int ncw, Int npe, Int ne,
                      Int backend)
{
    Int numPoints = npe * ne;
    Int modelnumber = app.modelnumber;

    abi.init.cpuInitwdg(f, xg, app.uinf, app.physicsparam, modelnumber, numPoints,
                   ncx, ncw, npe, ne);
}

void FluxDriver(dstype* f, dstype* f_udg, dstype* f_wdg, const dstype* xg,
                dstype* udg, const dstype* odg, const dstype* wdg,
                ExasimDriverABI& abi, meshstruct& mesh, masterstruct& master,
                appstruct& app, solstruct& sol, tempstruct& temp,
                commonstruct& common, Int nge, Int e1, Int e2, Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int numPoints = nge * (e2 - e1);
    dstype time = common.timestate.time;

    abi.hdgjac.HdgFlux(f, f_udg, f_wdg, xg, udg, odg, wdg, app.uinf,
                app.physicsparam, time, common.modelnumber, numPoints, nc,
                ncu, nd, ncx, nco, ncw);
}

void SourceDriver(dstype* f, dstype* f_udg, dstype* f_wdg,
                  const dstype* xg, const dstype* udg, const dstype* odg,
                  const dstype* wdg, ExasimDriverABI& abi, meshstruct& mesh,
                  masterstruct& master, appstruct& app, solstruct& sol,
                  tempstruct& temp, commonstruct& common, Int nge, Int e1,
                  Int e2, Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int numPoints = nge * (e2 - e1);
    dstype time = common.timestate.time;

    abi.hdgjac.HdgSource(f, f_udg, f_wdg, xg, udg, odg, wdg, app.uinf,
                  app.physicsparam, time, common.modelnumber, numPoints, nc,
                  ncu, nd, ncx, nco, ncw);
}

void SourcewDriver(dstype* f, dstype* f_udg, dstype* f_wdg,
                   const dstype* xg, const dstype* udg, const dstype* odg,
                   const dstype* wdg, ExasimDriverABI& abi, meshstruct& mesh,
                   masterstruct& master, appstruct& app, solstruct& sol,
                   tempstruct& temp, commonstruct& common, Int nge, Int e1,
                   Int e2, Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int numPoints = nge * (e2 - e1);
    dstype time = common.timestate.time;

    abi.hdgjac.HdgSourcew(f, f_udg, f_wdg, xg, udg, odg, wdg, app.uinf,
                   app.physicsparam, time, common.modelnumber, numPoints, nc,
                   ncu, nd, ncx, nco, ncw);
}

void SourcewDriver(dstype* f, dstype* f_wdg, const dstype* xg,
                   const dstype* udg, const dstype* odg, const dstype* wdg,
                   ExasimDriverABI& abi, meshstruct& mesh,
                   masterstruct& master, appstruct& app, solstruct& sol,
                   tempstruct& temp, commonstruct& common, Int nge, Int e1,
                   Int e2, Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int numPoints = nge * (e2 - e1);
    dstype time = common.timestate.time;

    abi.hdgjac.HdgSourcewonly(f, f_wdg, xg, udg, odg, wdg, app.uinf,
                       app.physicsparam, time, common.modelnumber, numPoints,
                       nc, ncu, nd, ncx, nco, ncw);
}

void EosDriver(dstype* f, dstype* f_udg, dstype* f_wdg, const dstype* xg,
               const dstype* udg, const dstype* odg, const dstype* wdg,
               ExasimDriverABI& abi, meshstruct& mesh, masterstruct& master,
               appstruct& app, solstruct& sol, tempstruct& temp,
               commonstruct& common, Int nge, Int e1, Int e2, Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int numPoints = nge * (e2 - e1);
    dstype time = common.timestate.time;

    abi.hdgjac.HdgEoS(f, f_udg, f_wdg, xg, udg, odg, wdg, app.uinf,
               app.physicsparam, time, common.modelnumber, numPoints, nc, ncu,
               nd, ncx, nco, ncw);
}

void FbouDriver(dstype* f, dstype* f_udg, dstype* f_wdg, dstype* f_uhg,
                dstype* xg, const dstype* udg, const dstype* odg,
                const dstype* wdg, dstype* uhg, const dstype* nl,
                ExasimDriverABI& abi, meshstruct& mesh, masterstruct& master,
                appstruct& app, solstruct& sol, tempstruct& temp,
                commonstruct& common, Int nga, Int ib, Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int numPoints = nga;
    dstype time = common.timestate.time;

    abi.hdgjac.HdgFbou(f, f_udg, f_wdg, f_uhg, xg, udg, odg, wdg, uhg, nl, app.tau,
                app.uinf, app.physicsparam, time, common.modelnumber, ib,
                numPoints, nc, ncu, nd, ncx, nco, ncw);
}

void FbouDriver(dstype* f, dstype* xg, const dstype* udg,
                const dstype* odg, const dstype* wdg, dstype* uhg,
                const dstype* nl, ExasimDriverABI& abi, meshstruct& mesh,
                masterstruct& master, appstruct& app, solstruct& sol,
                tempstruct& temp, commonstruct& common, Int nga, Int ib,
                Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int numPoints = nga;
    dstype time = common.timestate.time;

    abi.hdgjac.HdgFbouonly(f, xg, udg, odg, wdg, uhg, nl, app.tau, app.uinf,
                    app.physicsparam, time, common.modelnumber, ib, numPoints,
                    nc, ncu, nd, ncx, nco, ncw);
}

void FintDriver(dstype* f, dstype* f_udg, dstype* f_wdg, dstype* f_uhg,
                dstype* xg, const dstype* udg, const dstype* odg,
                const dstype* wdg, dstype* uhg, const dstype* nl,
                ExasimDriverABI& abi, meshstruct& mesh, masterstruct& master,
                appstruct& app, solstruct& sol, tempstruct& temp,
                commonstruct& common, Int nga, Int ib, Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int numPoints = nga;
    dstype time = common.timestate.time;

    abi.hdgjac.HdgFint(f, f_udg, f_wdg, f_uhg, xg, udg, odg, wdg, uhg, nl, app.tau,
                app.uinf, app.physicsparam, time, common.modelnumber, ib,
                numPoints, nc, ncu, nd, ncx, nco, ncw);
}

void FintDriver(dstype* f, dstype* xg, const dstype* udg,
                const dstype* odg, const dstype* wdg, dstype* uhg,
                const dstype* nl, ExasimDriverABI& abi, meshstruct& mesh,
                masterstruct& master, appstruct& app, solstruct& sol,
                tempstruct& temp, commonstruct& common, Int nga, Int ib,
                Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int numPoints = nga;
    dstype time = common.timestate.time;

    abi.hdgjac.HdgFintonly(f, xg, udg, odg, wdg, uhg, nl, app.tau, app.uinf,
                    app.physicsparam, time, common.modelnumber, ib, numPoints,
                    nc, ncu, nd, ncx, nco, ncw);
}

void FextDriver(dstype* f, dstype* f_udg, dstype* f_wdg, dstype* f_uhg,
                dstype* xg, const dstype* udg, const dstype* odg,
                const dstype* wdg, dstype* uhg, const dstype* nl,
                const dstype* uext, ExasimDriverABI& abi, meshstruct& mesh,
                masterstruct& master, appstruct& app, solstruct& sol,
                tempstruct& temp, commonstruct& common, Int nga, Int ib,
                Int backend)
{
    abi.hdgjac.HdgFext(f, f_udg, f_wdg, f_uhg, xg, udg, odg, wdg, uhg, nl, uext,
                app.tau, app.uinf, app.physicsparam, common.timestate.time,
                common.modelnumber, ib, nga, common.components.nc, common.components.ncu, common.grid.nd,
                common.components.ncx, common.components.nco, common.components.ncw);
}

void FextDriver(dstype* f, dstype* xg, const dstype* udg,
                const dstype* odg, const dstype* wdg, dstype* uhg,
                const dstype* nl, const dstype* uext,
                ExasimDriverABI& abi, meshstruct& mesh, masterstruct& master, appstruct& app,
                solstruct& sol, tempstruct& temp, commonstruct& common,
                Int nga, Int ib, Int backend)
{
    abi.hdgjac.HdgFextonly(f, xg, udg, odg, wdg, uhg, nl, uext, app.tau, app.uinf,
                    app.physicsparam, common.timestate.time, common.modelnumber, ib, nga,
                    common.components.nc, common.components.ncu, common.grid.nd, common.components.ncx, common.components.nco,
                    common.components.ncw);
}

void FhatDriver(dstype* f, dstype* f_udg, dstype* f_wdg, dstype* f_uhg,
                const dstype* xg, dstype* udg, const dstype* odg,
                const dstype* wdg, const dstype* uhg, dstype* nl,
                ExasimDriverABI& abi, meshstruct& mesh, masterstruct& master,
                appstruct& app, solstruct& sol, tempstruct& temp,
                commonstruct& common, Int nga, Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int numPoints = nga;
    Int M = numPoints * ncu;
    Int N = numPoints * ncu * nd;
    dstype time = common.timestate.time;

    ArrayCopy(f_uhg, udg, numPoints * ncu);
    ArrayCopy(udg, uhg, numPoints * ncu);

    abi.hdgjac.HdgFlux(f, f_udg, f_wdg, xg, udg, odg, wdg, app.uinf,
                app.physicsparam, time, common.modelnumber, numPoints, nc,
                ncu, nd, ncx, nco, ncw);

    ArrayCopy(udg, f_uhg, numPoints * ncu);

    FluxDotNormal(f, f, nl, M, numPoints, nd);

    for (int n = 0; n < nc; n++) {
        FluxDotNormal(&f_udg[M * n], &f_udg[N * n], nl, M, numPoints, nd);
    }

    if ((ncw > 0) & (common.timeparams.wave == 0)) {
        for (int n = 0; n < ncw; n++) {
            FluxDotNormal(&f_wdg[M * n], &f_wdg[N * n], nl, M, numPoints, nd);
        }
    }

    ArrayCopy(f_uhg, f_udg, numPoints * ncu * ncu);
    ArraySetValue(f_udg, zero, numPoints * ncu * ncu);
    AddStabilization1(f, f_udg, f_uhg, udg, uhg, app.tau, M, numPoints);
}

void FhatDriver(dstype* f, dstype* u, const dstype* xg, dstype* udg,
                const dstype* odg, const dstype* wdg, const dstype* uhg,
                dstype* nl, ExasimDriverABI& abi, meshstruct& mesh,
                masterstruct& master, appstruct& app, solstruct& sol,
                tempstruct& temp, commonstruct& common, Int nga, Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int numPoints = nga;
    Int M = numPoints * ncu;
    dstype time = common.timestate.time;

    ArrayCopy(u, udg, numPoints * ncu);
    ArrayCopy(udg, uhg, numPoints * ncu);

    abi.volume.KokkosFlux(f, xg, udg, odg, wdg, app.uinf, app.physicsparam, time,
                   common.modelnumber, numPoints, nc, ncu, nd, ncx, nco, ncw);

    ArrayCopy(udg, u, numPoints * ncu);

    FluxDotNormal(f, f, nl, M, numPoints, nd);
    AddStabilization1(f, udg, uhg, app.tau, M);
}

void VisScalarsDriver(dstype* f, const dstype* xg, const dstype* udg,
                      const dstype* odg, const dstype* wdg,
                      ExasimDriverABI& abi, meshstruct& mesh,
                      masterstruct& master, appstruct& app, solstruct& sol,
                      tempstruct& temp, commonstruct& common, Int nge, Int e1,
                      Int e2, Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int numPoints = nge * (e2 - e1);
    dstype time = common.timestate.time;

    abi.output.KokkosVisScalars(f, xg, udg, odg, wdg, app.uinf, app.physicsparam,
                         time, common.modelnumber, numPoints, nc, ncu, nd, ncx,
                         nco, ncw);
}

void VisVectorsDriver(dstype* f, const dstype* xg, const dstype* udg,
                      const dstype* odg, const dstype* wdg,
                      ExasimDriverABI& abi, meshstruct& mesh,
                      masterstruct& master, appstruct& app, solstruct& sol,
                      tempstruct& temp, commonstruct& common, Int nge, Int e1,
                      Int e2, Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int numPoints = nge * (e2 - e1);
    dstype time = common.timestate.time;

    abi.output.KokkosVisVectors(f, xg, udg, odg, wdg, app.uinf, app.physicsparam,
                         time, common.modelnumber, numPoints, nc, ncu, nd, ncx,
                         nco, ncw);
}

void VisTensorsDriver(dstype* f, const dstype* xg, const dstype* udg,
                      const dstype* odg, const dstype* wdg,
                      ExasimDriverABI& abi, meshstruct& mesh,
                      masterstruct& master, appstruct& app, solstruct& sol,
                      tempstruct& temp, commonstruct& common, Int nge, Int e1,
                      Int e2, Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int numPoints = nge * (e2 - e1);
    dstype time = common.timestate.time;

    abi.output.KokkosVisTensors(f, xg, udg, odg, wdg, app.uinf, app.physicsparam,
                         time, common.modelnumber, numPoints, nc, ncu, nd, ncx,
                         nco, ncw);
}

void QoIvolumeDriver(dstype* f, const dstype* xg, const dstype* udg,
                     const dstype* odg, const dstype* wdg,
                     ExasimDriverABI& abi, meshstruct& mesh,
                     masterstruct& master, appstruct& app, solstruct& sol,
                     tempstruct& temp, commonstruct& common, Int nge, Int e1,
                     Int e2, Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int numPoints = nge * (e2 - e1);
    dstype time = common.timestate.time;

    abi.qoi.KokkosQoIvolume(f, xg, udg, odg, wdg, app.uinf, app.physicsparam,
                        time, common.modelnumber, numPoints, nc, ncu, nd, ncx,
                        nco, ncw);
}

void QoIboundaryDriver(dstype* fb, const dstype* xg, const dstype* udg,
                       const dstype* odg, const dstype* wdg,
                       const dstype* uhg, const dstype* nl,
                       ExasimDriverABI& abi, meshstruct& mesh,
                       masterstruct& master, appstruct& app, solstruct& sol,
                       tempstruct& temp, commonstruct& common, Int ngf, Int f1,
                       Int f2, Int ib, Int backend)
{
    Int nc = common.components.nc;
    Int ncu = common.components.ncu;
    Int ncw = common.components.ncw;
    Int nco = common.components.nco;
    Int ncx = common.components.ncx;
    Int nd = common.grid.nd;
    Int numPoints = ngf * (f2 - f1);
    dstype time = common.timestate.time;

    abi.qoi.KokkosQoIboundary(fb, xg, udg, odg, wdg, uhg, nl, app.tau, app.uinf,
                          app.physicsparam, time, common.modelnumber, ib,
                          numPoints, nc, ncu, nd, ncx, nco, ncw);
}

// Recover the runtime ABI from common.driver_abi for the no-driver (AbiAdapter) overloads
// below. Fails loudly instead of null-dereferencing deep in kernel dispatch if an AbiAdapter
// path reached here without initializing common.driver_abi.
static inline ExasimDriverABI& require_driver_abi(const commonstruct& common) {
    if (!common.driver_abi) {
        std::fprintf(stderr, "[exasim] FATAL: unified driver invoked but common.driver_abi is "
                             "null (AbiAdapter path did not initialize the driver ABI)\n");
        std::abort();
    }
    return *common.driver_abi;
}

// --- No-driver_abi overloads (auto-generated) --------------------------------------------
// The unified templated FEM code invokes kernel drivers via EXASIM_DRIVER_CALL without
// threading driver_abi. For the AbiAdapter path these overloads recover the ABI from
// common.driver_abi and forward to the explicit-ABI versions above. Only drivers that take
// a commonstruct (the solve-loop kernels) are covered; init-path drivers (no common) are
// handled when their file is unified.

inline void FluxDriver(dstype* f, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int nge, Int e1, Int e2, Int backend)
{
    FluxDriver(f, xg, udg, odg, wdg, require_driver_abi(common), mesh, master, app, sol, temp, common, nge, e1, e2, backend);
}

inline void SourceDriver(dstype* f, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int nge, Int e1, Int e2, Int backend)
{
    SourceDriver(f, xg, udg, odg, wdg, require_driver_abi(common), mesh, master, app, sol, temp, common, nge, e1, e2, backend);
}

inline void SourcewDriver(dstype* f, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int npe, Int e1, Int e2, Int backend)
{
    SourcewDriver(f, xg, udg, odg, wdg, require_driver_abi(common), mesh, master, app, sol, temp, common, npe, e1, e2, backend);
}

inline void OutputDriver(dstype* f, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int backend)
{
    OutputDriver(f, xg, udg, odg, wdg, require_driver_abi(common), mesh, master, app, sol, temp, common, backend);
}

inline void MonitorDriver(dstype* f, Int nc_sol, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int backend)
{
    MonitorDriver(f, nc_sol, xg, udg, odg, wdg, require_driver_abi(common), mesh, master, app, sol, temp, common, backend);
}

inline void AvfieldDriver(dstype* f, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int backend)
{
    AvfieldDriver(f, xg, udg, odg, wdg, require_driver_abi(common), mesh, master, app, sol, temp, common, backend);
}

inline void EosDriver(dstype* f, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int npe, Int e1, Int e2, Int backend)
{
    EosDriver(f, xg, udg, odg, wdg, require_driver_abi(common), mesh, master, app, sol, temp, common, npe, e1, e2, backend);
}

inline void EosduDriver(dstype* f, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int npe, Int e1, Int e2, Int backend)
{
    EosduDriver(f, xg, udg, odg, wdg, require_driver_abi(common), mesh, master, app, sol, temp, common, npe, e1, e2, backend);
}

inline void EosdwDriver(dstype* f, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int npe, Int e1, Int e2, Int backend)
{
    EosdwDriver(f, xg, udg, odg, wdg, require_driver_abi(common), mesh, master, app, sol, temp, common, npe, e1, e2, backend);
}

inline void TdfuncDriver(dstype* f, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int nge, Int e1, Int e2, Int backend)
{
    TdfuncDriver(f, xg, udg, odg, wdg, require_driver_abi(common), mesh, master, app, sol, temp, common, nge, e1, e2, backend);
}

inline void FhatDriver(dstype* fg, const dstype* xg, const dstype* ug1, const dstype* ug2, const dstype* og1, const dstype* og2, const dstype* wg1, const dstype* wg2, const dstype* uh, const dstype* nl, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& tmp, commonstruct& common, Int ngf, Int f1, Int f2, Int backend)
{
    FhatDriver(fg, xg, ug1, ug2, og1, og2, wg1, wg2, uh, nl, require_driver_abi(common), mesh, master, app, sol, tmp, common, ngf, f1, f2, backend);
}

inline void FbouDriver(dstype* fb, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uhg, const dstype* nl, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int ngf, Int f1, Int f2, Int ib, Int backend)
{
    FbouDriver(fb, xg, udg, odg, wdg, uhg, nl, require_driver_abi(common), mesh, master, app, sol, temp, common, ngf, f1, f2, ib, backend);
}

inline void FbouJacDriver(dstype* fb, dstype* fb_udg, dstype* fb_wdg, dstype* fb_uhg, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uhg, const dstype* nl, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int nga, Int ib, Int backend)
{
    FbouJacDriver(fb, fb_udg, fb_wdg, fb_uhg, xg, udg, odg, wdg, uhg, nl, require_driver_abi(common), mesh, master, app, sol, temp, common, nga, ib, backend);
}

inline void FbouJacDriver(dstype* fb, dstype* fb_udg, dstype* fb_wdg, dstype* fb_uhg, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uhg, const dstype* nl, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int ngf, Int f1, Int f2, Int ib, Int backend)
{
    FbouJacDriver(fb, fb_udg, fb_wdg, fb_uhg, xg, udg, odg, wdg, uhg, nl, require_driver_abi(common), mesh, master, app, sol, temp, common, ngf, f1, f2, ib, backend);
}

inline void UhatDriver(dstype* fg, dstype* xg, dstype* ug1, dstype* ug2, const dstype* og1, const dstype* og2, const dstype* wg1, const dstype* wg2, const dstype* uh, const dstype* nl, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& tmp, commonstruct& common, Int ngf, Int f1, Int f2, Int backend)
{
    UhatDriver(fg, xg, ug1, ug2, og1, og2, wg1, wg2, uh, nl, require_driver_abi(common), mesh, master, app, sol, tmp, common, ngf, f1, f2, backend);
}

inline void UbouDriver(dstype* ub, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uhg, const dstype* nl, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int ngf, Int f1, Int f2, Int ib, Int backend)
{
    UbouDriver(ub, xg, udg, odg, wdg, uhg, nl, require_driver_abi(common), mesh, master, app, sol, temp, common, ngf, f1, f2, ib, backend);
}

inline void UbouJacDriver(dstype* ub, dstype* ub_udg, dstype* ub_wdg, dstype* ub_uhg, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uhg, const dstype* nl, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int nga, Int ib, Int backend)
{
    UbouJacDriver(ub, ub_udg, ub_wdg, ub_uhg, xg, udg, odg, wdg, uhg, nl, require_driver_abi(common), mesh, master, app, sol, temp, common, nga, ib, backend);
}

inline void UbouJacDriver(dstype* ub, dstype* ub_udg, dstype* ub_wdg, dstype* ub_uhg, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uhg, const dstype* nl, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int ngf, Int f1, Int f2, Int ib, Int backend)
{
    UbouJacDriver(ub, ub_udg, ub_wdg, ub_uhg, xg, udg, odg, wdg, uhg, nl, require_driver_abi(common), mesh, master, app, sol, temp, common, ngf, f1, f2, ib, backend);
}

inline void FluxDriver(dstype* f, dstype* f_udg, dstype* f_wdg, const dstype* xg, dstype* udg, const dstype* odg, const dstype* wdg, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int nge, Int e1, Int e2, Int backend)
{
    FluxDriver(f, f_udg, f_wdg, xg, udg, odg, wdg, require_driver_abi(common), mesh, master, app, sol, temp, common, nge, e1, e2, backend);
}

inline void SourceDriver(dstype* f, dstype* f_udg, dstype* f_wdg, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int nge, Int e1, Int e2, Int backend)
{
    SourceDriver(f, f_udg, f_wdg, xg, udg, odg, wdg, require_driver_abi(common), mesh, master, app, sol, temp, common, nge, e1, e2, backend);
}

inline void SourcewDriver(dstype* f, dstype* f_udg, dstype* f_wdg, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int nge, Int e1, Int e2, Int backend)
{
    SourcewDriver(f, f_udg, f_wdg, xg, udg, odg, wdg, require_driver_abi(common), mesh, master, app, sol, temp, common, nge, e1, e2, backend);
}

inline void SourcewDriver(dstype* f, dstype* f_wdg, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int nge, Int e1, Int e2, Int backend)
{
    SourcewDriver(f, f_wdg, xg, udg, odg, wdg, require_driver_abi(common), mesh, master, app, sol, temp, common, nge, e1, e2, backend);
}

inline void EosDriver(dstype* f, dstype* f_udg, dstype* f_wdg, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int nge, Int e1, Int e2, Int backend)
{
    EosDriver(f, f_udg, f_wdg, xg, udg, odg, wdg, require_driver_abi(common), mesh, master, app, sol, temp, common, nge, e1, e2, backend);
}

inline void FbouDriver(dstype* f, dstype* f_udg, dstype* f_wdg, dstype* f_uhg, dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, dstype* uhg, const dstype* nl, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int nga, Int ib, Int backend)
{
    FbouDriver(f, f_udg, f_wdg, f_uhg, xg, udg, odg, wdg, uhg, nl, require_driver_abi(common), mesh, master, app, sol, temp, common, nga, ib, backend);
}

inline void FbouDriver(dstype* f, dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, dstype* uhg, const dstype* nl, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int nga, Int ib, Int backend)
{
    FbouDriver(f, xg, udg, odg, wdg, uhg, nl, require_driver_abi(common), mesh, master, app, sol, temp, common, nga, ib, backend);
}

inline void FintDriver(dstype* f, dstype* f_udg, dstype* f_wdg, dstype* f_uhg, dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, dstype* uhg, const dstype* nl, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int nga, Int ib, Int backend)
{
    FintDriver(f, f_udg, f_wdg, f_uhg, xg, udg, odg, wdg, uhg, nl, require_driver_abi(common), mesh, master, app, sol, temp, common, nga, ib, backend);
}

inline void FintDriver(dstype* f, dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, dstype* uhg, const dstype* nl, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int nga, Int ib, Int backend)
{
    FintDriver(f, xg, udg, odg, wdg, uhg, nl, require_driver_abi(common), mesh, master, app, sol, temp, common, nga, ib, backend);
}

inline void FextDriver(dstype* f, dstype* f_udg, dstype* f_wdg, dstype* f_uhg, dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, dstype* uhg, const dstype* nl, const dstype* uext, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int nga, Int ib, Int backend)
{
    FextDriver(f, f_udg, f_wdg, f_uhg, xg, udg, odg, wdg, uhg, nl, uext, require_driver_abi(common), mesh, master, app, sol, temp, common, nga, ib, backend);
}

inline void FextDriver(dstype* f, dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, dstype* uhg, const dstype* nl, const dstype* uext, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int nga, Int ib, Int backend)
{
    FextDriver(f, xg, udg, odg, wdg, uhg, nl, uext, require_driver_abi(common), mesh, master, app, sol, temp, common, nga, ib, backend);
}

inline void FhatDriver(dstype* f, dstype* f_udg, dstype* f_wdg, dstype* f_uhg, const dstype* xg, dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uhg, dstype* nl, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int nga, Int backend)
{
    FhatDriver(f, f_udg, f_wdg, f_uhg, xg, udg, odg, wdg, uhg, nl, require_driver_abi(common), mesh, master, app, sol, temp, common, nga, backend);
}

inline void FhatDriver(dstype* f, dstype* u, const dstype* xg, dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uhg, dstype* nl, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int nga, Int backend)
{
    FhatDriver(f, u, xg, udg, odg, wdg, uhg, nl, require_driver_abi(common), mesh, master, app, sol, temp, common, nga, backend);
}

inline void VisScalarsDriver(dstype* f, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int nge, Int e1, Int e2, Int backend)
{
    VisScalarsDriver(f, xg, udg, odg, wdg, require_driver_abi(common), mesh, master, app, sol, temp, common, nge, e1, e2, backend);
}

inline void VisVectorsDriver(dstype* f, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int nge, Int e1, Int e2, Int backend)
{
    VisVectorsDriver(f, xg, udg, odg, wdg, require_driver_abi(common), mesh, master, app, sol, temp, common, nge, e1, e2, backend);
}

inline void VisTensorsDriver(dstype* f, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int nge, Int e1, Int e2, Int backend)
{
    VisTensorsDriver(f, xg, udg, odg, wdg, require_driver_abi(common), mesh, master, app, sol, temp, common, nge, e1, e2, backend);
}

inline void QoIvolumeDriver(dstype* f, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int nge, Int e1, Int e2, Int backend)
{
    QoIvolumeDriver(f, xg, udg, odg, wdg, require_driver_abi(common), mesh, master, app, sol, temp, common, nge, e1, e2, backend);
}

inline void QoIboundaryDriver(dstype* fb, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uhg, const dstype* nl, meshstruct& mesh, masterstruct& master, appstruct& app, solstruct& sol, tempstruct& temp, commonstruct& common, Int ngf, Int f1, Int f2, Int ib, Int backend)
{
    QoIboundaryDriver(fb, xg, udg, odg, wdg, uhg, nl, require_driver_abi(common), mesh, master, app, sol, temp, common, ngf, f1, f2, ib, backend);
}

#endif
