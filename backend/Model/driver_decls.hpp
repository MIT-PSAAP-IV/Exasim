// SPDX-License-Identifier: see LICENSE
//
// <backend/Model/driver_decls.hpp>
//
// Declarations of the legacy global *Driver(...) functions, used by the
// header-only / templated build (HAVE_KOKKOSKERNEL). EXASIM_DRIVER_CALL's
// AbiAdapter branch names these unqualified; for a real Model M that branch is
// discarded by `if constexpr`, but the names must still be declared for the
// template to parse. The definitions (KokkosDrivers.cpp / ModelDrivers.cpp /
// BuiltinModelDrivers.cpp) are NOT included on this path -- the templated
// branch (exasim::*Driver<M>) provides the physics from M's pointwise math.
//
// Auto-extractable from backend/Model/KokkosDrivers.cpp signatures.
#pragma once

void FluxDriver(dstype* f, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct &mesh, 
        masterstruct &master, appstruct &app, solstruct &sol, tempstruct &temp, 
        commonstruct &common, Int nge, Int e1, Int e2, Int backend)
;
void SourceDriver(dstype* f, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct &mesh, 
        masterstruct &master, appstruct &app, solstruct &sol, tempstruct &temp, 
        commonstruct &common, Int nge, Int e1, Int e2, Int backend)
;
void QoIvolumeDriver(dstype* f, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct &mesh, 
        masterstruct &master, appstruct &app, solstruct &sol, tempstruct &temp, 
        commonstruct &common, Int nge, Int e1, Int e2, Int backend)
;
void VisScalarsDriver(dstype* f, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct &mesh, 
        masterstruct &master, appstruct &app, solstruct &sol, tempstruct &temp, 
        commonstruct &common, Int nge, Int e1, Int e2, Int backend)
;
void VisVectorsDriver(dstype* f, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct &mesh, 
        masterstruct &master, appstruct &app, solstruct &sol, tempstruct &temp, 
        commonstruct &common, Int nge, Int e1, Int e2, Int backend)
;
void VisTensorsDriver(dstype* f, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct &mesh, 
        masterstruct &master, appstruct &app, solstruct &sol, tempstruct &temp, 
        commonstruct &common, Int nge, Int e1, Int e2, Int backend)
;
void SourcewDriver(dstype* f, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct &mesh, 
        masterstruct &master, appstruct &app, solstruct &sol, tempstruct &temp, 
        commonstruct &common, Int npe, Int e1, Int e2, Int backend)
;
void OutputDriver(dstype* f, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct &mesh, 
        masterstruct &master, appstruct &app, solstruct &sol, tempstruct &temp, 
        commonstruct &common, Int backend)
;
void MonitorDriver(dstype* f, Int nc_sol, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct &mesh, 
  masterstruct &master, appstruct &app, solstruct &sol, tempstruct &temp, 
  commonstruct &common, Int backend)
;
void AvfieldDriver(dstype* f, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct &mesh, 
        masterstruct &master, appstruct &app, solstruct &sol, tempstruct &temp, 
        commonstruct &common, Int backend)
;
void EosDriver(dstype* f, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct &mesh, 
        masterstruct &master, appstruct &app, solstruct &sol, tempstruct &temp, 
        commonstruct &common, Int npe, Int e1, Int e2, Int backend)
;
void EosduDriver(dstype* f, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct &mesh, 
        masterstruct &master, appstruct &app, solstruct &sol, tempstruct &temp, 
        commonstruct &common, Int npe, Int e1, Int e2, Int backend)
;
void EosdwDriver(dstype* f, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct &mesh, 
        masterstruct &master, appstruct &app, solstruct &sol, tempstruct &temp, 
        commonstruct &common, Int npe, Int e1, Int e2, Int backend)
;
void TdfuncDriver(dstype* f, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct &mesh, 
        masterstruct &master, appstruct &app, solstruct &sol, tempstruct &temp, 
        commonstruct &common, Int nge, Int e1, Int e2, Int backend)
;
void FhatDriver(dstype* fg, const dstype* xg, const dstype* ug1, const dstype* ug2, const dstype*  og1, const dstype*  og2, 
    const dstype* wg1, const dstype* wg2, const dstype* uh, const dstype* nl, meshstruct &mesh, masterstruct &master, appstruct &app, 
    solstruct &sol, tempstruct &tmp, commonstruct &common, Int ngf, Int f1, Int f2, Int backend)
;
void FbouDriver(dstype* fb, const dstype* xg, const dstype* udg, const dstype*  odg, const dstype*  wdg, const dstype* uhg, const dstype* nl, 
        meshstruct &mesh, masterstruct &master, appstruct &app, solstruct &sol, tempstruct &temp, 
        commonstruct &common, Int ngf, Int f1, Int f2, Int ib, Int backend)
;
void QoIboundaryDriver(dstype* fb, const dstype* xg, const dstype* udg, const dstype*  odg, const dstype*  wdg, const dstype* uhg, const dstype* nl, 
        meshstruct &mesh, masterstruct &master, appstruct &app, solstruct &sol, tempstruct &temp, 
        commonstruct &common, Int ngf, Int f1, Int f2, Int ib, Int backend)
;
void UhatDriver(dstype* fg, dstype* xg, dstype* ug1, dstype* ug2, const dstype*  og1, 
     const dstype*  og2, const dstype* wg1, const dstype* wg2, const dstype* uh, const dstype* nl, meshstruct &mesh, masterstruct &master, appstruct &app, solstruct &sol, 
     tempstruct &tmp, commonstruct &common, Int ngf, Int f1, Int f2, Int backend)
;
void UbouDriver(dstype* ub, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uhg, const dstype* nl, 
        meshstruct &mesh, masterstruct &master, appstruct &app, solstruct &sol, tempstruct &temp, 
        commonstruct &common, Int ngf, Int f1, Int f2, Int ib, Int backend)
;
void InitodgDriver(dstype* f, const dstype* xg, appstruct &app, Int ncx, Int nco, Int npe, Int ne, Int backend)
;
void InitqDriver(dstype* f, const dstype* xg, appstruct &app, Int ncx, Int nc, Int npe, Int ne, Int backend)
;
void InitudgDriver(dstype* f, const dstype* xg, appstruct &app, Int ncx, Int nc, Int npe, Int ne, Int backend)
;
void InituDriver(dstype* f, const dstype* xg, appstruct &app, Int ncx, Int nc, Int npe, Int ne, Int backend)
;
void InitwdgDriver(dstype* f, const dstype* xg, appstruct &app, Int ncx, Int ncw, Int npe, Int ne, Int backend)
;
void cpuInitodgDriver(dstype* f, const dstype* xg, appstruct &app, Int ncx, Int nco, Int npe, Int ne, Int backend)
;
void cpuInitqDriver(dstype* f, const dstype* xg, appstruct &app, Int ncx, Int nc, Int npe, Int ne, Int backend)
;
void cpuInitudgDriver(dstype* f, const dstype* xg, appstruct &app, Int ncx, Int nc, Int npe, Int ne, Int backend)
;
void cpuInituDriver(dstype* f, const dstype* xg, appstruct &app, Int ncx, Int nc, Int npe, Int ne, Int backend)
;
void cpuInitwdgDriver(dstype* f, const dstype* xg, appstruct &app, Int ncx, Int ncw, Int npe, Int ne, Int backend)
;
void FluxDriver(dstype* f, dstype* f_udg, dstype* f_wdg, const dstype* xg,  dstype* udg, const dstype* odg, const dstype* wdg, meshstruct &mesh, 
        masterstruct &master, appstruct &app, solstruct &sol, tempstruct &temp, 
        commonstruct &common, Int nge, Int e1, Int e2, Int backend)
;
void SourceDriver(dstype* f, dstype* f_udg, dstype* f_wdg, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct &mesh, 
        masterstruct &master, appstruct &app, solstruct &sol, tempstruct &temp, 
        commonstruct &common, Int nge, Int e1, Int e2, Int backend)
;
void SourcewDriver(dstype* f, dstype* f_udg, dstype* f_wdg, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct &mesh, 
        masterstruct &master, appstruct &app, solstruct &sol, tempstruct &temp, 
        commonstruct &common, Int nge, Int e1, Int e2, Int backend)
;
void SourcewDriver(dstype* f, dstype* f_wdg, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct &mesh, 
        masterstruct &master, appstruct &app, solstruct &sol, tempstruct &temp, 
        commonstruct &common, Int nge, Int e1, Int e2, Int backend)
;
void EosDriver(dstype* f, dstype* f_udg, dstype* f_wdg, const dstype* xg, const dstype* udg, const dstype* odg, const dstype* wdg, meshstruct &mesh, 
        masterstruct &master, appstruct &app, solstruct &sol, tempstruct &temp, 
        commonstruct &common, Int nge, Int e1, Int e2, Int backend)
;
void FbouDriver(dstype* f,  dstype* f_udg, dstype* f_wdg, dstype* f_uhg, dstype* xg, const dstype* udg, 
        const dstype*  odg, const dstype*  wdg, dstype* uhg, const dstype* nl, meshstruct &mesh, 
        masterstruct &master, appstruct &app, solstruct &sol, tempstruct &temp, commonstruct &common, Int nga, Int ib, Int backend)
;
void FbouDriver(dstype* f, dstype* xg, const dstype* udg, const dstype*  odg, const dstype*  wdg, dstype* uhg, 
        const dstype* nl, meshstruct &mesh, masterstruct &master, appstruct &app, solstruct &sol, tempstruct &temp, 
        commonstruct &common, Int nga, Int ib, Int backend)
;
void FintDriver(dstype* f,  dstype* f_udg, dstype* f_wdg, dstype* f_uhg, dstype* xg, const dstype* udg, 
        const dstype*  odg, const dstype*  wdg, dstype* uhg, const dstype* nl, meshstruct &mesh, 
        masterstruct &master, appstruct &app, solstruct &sol, tempstruct &temp, commonstruct &common, Int nga, Int ib, Int backend)
;
void FintDriver(dstype* f, dstype* xg, const dstype* udg, const dstype*  odg, const dstype*  wdg, dstype* uhg, 
        const dstype* nl, meshstruct &mesh, masterstruct &master, appstruct &app, solstruct &sol, tempstruct &temp, 
        commonstruct &common, Int nga, Int ib, Int backend)
;
void FextDriver(dstype* f,  dstype* f_udg, dstype* f_wdg, dstype* f_uhg, dstype* xg, const dstype* udg, 
        const dstype*  odg, const dstype*  wdg, dstype* uhg, const dstype* nl, const dstype* uext, meshstruct &mesh, 
        masterstruct &master, appstruct &app, solstruct &sol, tempstruct &temp, commonstruct &common, Int nga, Int ib, Int backend)
;
void FextDriver(dstype* f, dstype* xg, const dstype* udg, const dstype*  odg, const dstype*  wdg, dstype* uhg, 
        const dstype* nl, const dstype* uext, meshstruct &mesh, masterstruct &master, appstruct &app, 
        solstruct &sol, tempstruct &temp, commonstruct &common, Int nga, Int ib, Int backend)
;
void FhatDriver(dstype* f,  dstype* f_udg, dstype* f_wdg, dstype* f_uhg, const dstype* xg, dstype* udg, 
        const dstype*  odg, const dstype*  wdg, const dstype* uhg,  dstype* nl, meshstruct &mesh, 
        masterstruct &master, appstruct &app, solstruct &sol, tempstruct &temp, commonstruct &common, Int nga, Int backend)
;
void FhatDriver(dstype* f, dstype* u, const dstype* xg, dstype* udg, const dstype*  odg, const dstype*  wdg, const dstype* uhg,  
        dstype* nl, meshstruct &mesh, masterstruct &master, appstruct &app, solstruct &sol, tempstruct &temp, 
        commonstruct &common, Int nga, Int backend)
;

// HDG w-equation source kernels, called directly by wequation.hpp.
void HdgSourcew(dstype* f, dstype* f_udg, dstype* f_wdg, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg,
                 const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc,
                 const int ncu, const int nd, const int ncx, const int nco, const int ncw);
void HdgSourcewonly(dstype* f, dstype* f_wdg, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg,
                 const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc,
                 const int ncu, const int nd, const int ncx, const int nco, const int ncw);
