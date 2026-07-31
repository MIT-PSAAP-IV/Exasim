/*
    CInterfaceSampler -- extract field data (coordinates, normals, solution traces, fluxes,
    boundary averages) on a set of interface/boundary faces. This is coupling/sampling I/O on
    the discretization, not a function-space operation: callers are the CHEFSI coupling
    interface exchange (ExasimSolver), the time-loop boundary averaging (run.hpp), and the
    wall-model sampling. Extracted from CDiscretization; behaviour re-home (holds a
    CDiscretization& and operates on its data; method bodies unchanged via disc.* aliases).
*/
#ifndef __INTERFACESAMPLER
#define __INTERFACESAMPLER

#include "interfacesampler.h"

// Concrete-model (templated) interface-flux extraction needs the templated
// exasim::FintDriver<M> forwarder and the AbiAdapter tag. This file is
// #included late (end of discretization.cpp) so the driver + struct chain are
// already visible; the includes are idempotent (pragma once).
#include <type_traits>
#include <exasim/detail/abi_adapter.hpp>
#include <exasim/drivers.hpp>

Int CInterfaceSampler::getFacesOnInterface(Int **faces, const Int boundarycondition)
{
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common; [[maybe_unused]] auto& driver_abi = disc.driver_abi;
    int nintfaces = getinterfacefaces(mesh.bf, common.meshsizes.nfe, common.meshsizes.ne1, boundarycondition);
    int *intfaces = nullptr; 
    TemplateMalloc(&intfaces, nintfaces, 0);

    getinterfacefaces(intfaces, mesh.bf, common.meshsizes.nfe, common.meshsizes.ne1, boundarycondition, nintfaces);

    // Free any previous allocation first: this function is called again on every
    // re-initialise of the interface coupling (which is documented as safe to repeat),
    // and without this each call leaked the last nextfaces buffer (review #44).
    if (common.nextfaces != nullptr) { TemplateFree(common.nextfaces, 0); common.nextfaces = nullptr; }
    TemplateMalloc(&common.nextfaces, common.meshsizes.nbe1+1, 0);    
    Int nfacestotal = 0;
    common.nextfaces[0] = nfacestotal;
    for (Int j=0; j<common.meshsizes.nbe1; j++) {    
        Int e1 = common.eblks[3*j]-1;
        Int e2 = common.eblks[3*j+1];                    
        Int nfe = common.meshsizes.nfe;      
        int nfaces = getinterfacefaces(&mesh.bf[nfe*e1], nfe, e2-e1, boundarycondition);  
        nfacestotal += nfaces;
        common.nextfaces[j+1] = nfacestotal;
    }                                         
    if (nfacestotal != nintfaces) error("getFacesOnInterface is wrong because the numbers of faces do not match.");
  
    TemplateMalloc(faces, nintfaces, common.backend);
    TemplateCopytoDevice(*faces, intfaces, nintfaces, common.backend);                           

    CPUFREE(intfaces);

    return nintfaces;
}

void CInterfaceSampler::getDGNodesOnInterface(dstype* xdgint, const Int* faces, const Int nfaces)
{
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common; [[maybe_unused]] auto& driver_abi = disc.driver_abi;
    // npf * nfaces * ncx
    GetBoudaryNodes(xdgint, sol.xdg, faces, mesh.perm, common.meshsizes.nfe, 
                  common.grid.npf, common.grid.npe, common.components.ncx, common.components.ncx, nfaces);
}

void CInterfaceSampler::getUDGOnInterface(dstype* udgint, const Int* faces, const Int nfaces)
{
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common; [[maybe_unused]] auto& driver_abi = disc.driver_abi;
    GetBoudaryNodes(udgint, sol.udg, faces, mesh.perm, common.meshsizes.nfe, 
                  common.grid.npf, common.grid.npe, common.components.nc, common.components.nc, nfaces);
}

void CInterfaceSampler::getWDGOnInterface(dstype* wdgint, const Int* faces, const Int nfaces)
{
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common; [[maybe_unused]] auto& driver_abi = disc.driver_abi;
    GetBoudaryNodes(wdgint, sol.wdg, faces, mesh.perm, common.meshsizes.nfe, 
                  common.grid.npf, common.grid.npe, common.components.ncw, common.components.ncw, nfaces);
}

void CInterfaceSampler::getODGOnInterface(dstype* odgint, const Int* faces, const Int nfaces)
{
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common; [[maybe_unused]] auto& driver_abi = disc.driver_abi;
    GetBoudaryNodes(odgint, sol.odg, faces, mesh.perm, common.meshsizes.nfe, 
                  common.grid.npf, common.grid.npe, common.components.nco, common.components.nco, nfaces);
}

void CInterfaceSampler::getUHATOnInterface(dstype* uhint, const Int* faces, const Int nfaces)
{
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common; [[maybe_unused]] auto& driver_abi = disc.driver_abi;
    GetBoudaryNodes(uhint, sol.uh, faces, mesh.elemcon, common.meshsizes.nfe, 
                  common.grid.npf, common.components.ncu, nfaces);
}

void CInterfaceSampler::getNormalVectorOnInterface(dstype* nlint, dstype* xdgint, const Int nfaces)
{  
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common; [[maybe_unused]] auto& driver_abi = disc.driver_abi;
    Int nd = common.grid.nd; 
    Int npf = common.grid.npf; 
    Int nn = npf*nfaces; 
    Int ncx = common.components.ncx;    
    Int n2 = 0;    // jac
    Int n3 = nn;   // Jg
  
    if (nd==1) {
        FaceGeom1D(&tmp.tempn[n2], nlint, xdgint, nn);    
    }
    else if (nd==2){
        Node2Gauss(common.cublasHandle, &tmp.tempn[n3], xdgint, &master.shapfnt[npf*npf], npf, npf, nfaces*nd, common.backend);                
        FaceGeom2D(&tmp.tempn[n2], nlint, &tmp.tempn[n3], nn);
    }
    else if (nd==3) {
        Node2Gauss(common.cublasHandle, &tmp.tempn[n3], xdgint, &master.shapfnt[npf*npf], npf, npf, nfaces*nd, common.backend);                     
        Node2Gauss(common.cublasHandle, &tmp.tempn[n3+nn*nd], xdgint, &master.shapfnt[2*npf*npf], npf, npf, nfaces*nd, common.backend);                
        FaceGeom3D(&tmp.tempn[n2], nlint, &tmp.tempn[n3], nn);
    }
}

void CInterfaceSampler::getFieldsAtGaussPointsOnInterface(dstype* xdggint, dstype* xdgint, const Int nfaces, const Int ncx)
{
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common; [[maybe_unused]] auto& driver_abi = disc.driver_abi;
    Node2Gauss(common.cublasHandle, xdggint, xdgint, master.shapfgt, common.grid.ngf, common.grid.npf, nfaces*ncx, common.backend);    
}

// ABI-only entry points (used by the runtime solver, ExasimSolver / CSolution<>=AbiAdapter).
// They delegate to the templated implementation instantiated for the AbiAdapter tag, whose
// `if constexpr` AbiAdapter branch is the exact former body -> byte-identical.
void CInterfaceSampler::getInterfaceFluxesAtNodalPoints(dstype *flux, dstype* xdgint, dstype* nlint, const Int* faces, const Int nfaces)
{
    this->getInterfaceFluxesAtNodalPointsFor<exasim::detail::AbiAdapter>(flux, xdgint, nlint, faces, nfaces);
}

void CInterfaceSampler::getInterfaceFluxesAtGaussPoints(dstype *flux, dstype* xdggint, dstype* nlgint, const Int* faces, const Int nfaces)
{
    this->getInterfaceFluxesAtGaussPointsFor<exasim::detail::AbiAdapter>(flux, xdggint, nlgint, faces, nfaces);
}

// Templated interface-flux extraction. The field sampling (getUDG/ODG/WDG/UHAT +
// getFieldsAtGaussPointsOnInterface) is model-free and identical to the former ABI body;
// only the final FintDriver call is dispatched on M: AbiAdapter -> loaded ABI (byte-identical
// to before), concrete M -> exasim::FintDriver<M> value-only (fint_only_kernel<M>, no ABI).
template <class M>
void CInterfaceSampler::getInterfaceFluxesAtNodalPointsFor(dstype *flux, dstype* xdgint, dstype* nlint, const Int* faces, const Int nfaces)
{
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common; [[maybe_unused]] auto& driver_abi = disc.driver_abi;
    Int npf = common.grid.npf;
    dstype *udgint = &tmp.tempn[0]; // reuse tempg for udgint
    dstype *odgint = &tmp.tempn[npf * nfaces * common.components.nc];
    dstype *wdgint = &tmp.tempn[npf * nfaces * common.components.nc + npf * nfaces * common.components.nco];
    dstype *uhint = &tmp.tempn[npf * nfaces * common.components.nc + npf * nfaces * common.components.nco + npf * nfaces * common.components.ncw];

    this->getUDGOnInterface(udgint, faces, nfaces);
    this->getODGOnInterface(odgint, faces, nfaces);
    this->getWDGOnInterface(wdgint, faces, nfaces);
    this->getUHATOnInterface(uhint, faces, nfaces);

    if constexpr (std::is_same_v<M, exasim::detail::AbiAdapter>) {
        FintDriver(flux, xdgint, udgint, odgint, wdgint, uhint, nlint, driver_abi, mesh,
            master, app, sol, tmp, common, nfaces*npf, 1, common.backend);
    } else {
        exasim::FintDriver<M>(flux, xdgint, udgint, odgint, wdgint, uhint, nlint, mesh,
            master, app, sol, tmp, common, nfaces*npf, 1, common.backend);
    }
}

template <class M>
void CInterfaceSampler::getInterfaceFluxesAtGaussPointsFor(dstype *flux, dstype* xdggint, dstype* nlgint, const Int* faces, const Int nfaces)
{
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common; [[maybe_unused]] auto& driver_abi = disc.driver_abi;
    Int npf = common.grid.npf;
    Int ngf = common.grid.ngf;

    dstype *udgint = &tmp.tempn[0]; // reuse tempg for udgint
    dstype *odgint = &tmp.tempn[npf * nfaces * common.components.nc];
    dstype *wdgint = &tmp.tempn[npf * nfaces * common.components.nc + npf * nfaces * common.components.nco];
    dstype *uhint = &tmp.tempn[npf * nfaces * common.components.nc + npf * nfaces * common.components.nco + npf * nfaces * common.components.ncw];

    this->getUDGOnInterface(udgint, faces, nfaces);
    this->getODGOnInterface(odgint, faces, nfaces);
    this->getWDGOnInterface(wdgint, faces, nfaces);
    this->getUHATOnInterface(uhint, faces, nfaces);

    dstype *udggint = &tmp.tempg[0]; // reuse tempg2 for udggint
    dstype *odggint = &tmp.tempg[ngf * nfaces * common.components.nc];
    dstype *wdggint = &tmp.tempg[ngf * nfaces * (common.components.nc + common.components.nco)];
    dstype *uhgint = &tmp.tempg[ngf * nfaces * (common.components.nc + common.components.nco + common.components.ncw)];

    this->getFieldsAtGaussPointsOnInterface(udggint, udgint, nfaces, common.components.nc);
    this->getFieldsAtGaussPointsOnInterface(odggint, odgint, nfaces, common.components.nco);
    this->getFieldsAtGaussPointsOnInterface(wdggint, wdgint, nfaces, common.components.ncw);
    this->getFieldsAtGaussPointsOnInterface(uhgint, uhint, nfaces, common.components.ncu);

    if constexpr (std::is_same_v<M, exasim::detail::AbiAdapter>) {
        FintDriver(flux, xdggint, udggint, odggint, wdggint, uhgint, nlgint, driver_abi, mesh,
            master, app, sol, tmp, common, nfaces*ngf, 1, common.backend);
    } else {
        exasim::FintDriver<M>(flux, xdggint, udggint, odggint, wdggint, uhgint, nlgint, mesh,
            master, app, sol, tmp, common, nfaces*ngf, 1, common.backend);
    }
}

void CInterfaceSampler::computeAverageSolutionsOnBoundary() 
{   
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common; [[maybe_unused]] auto& driver_abi = disc.driver_abi;
    if ( common.outputparams.saveSolBouFreq>0 ) {
        for (Int j=0; j<common.meshsizes.nbf; j++) {
            Int ib = common.fblks[3*j+2];            
            if (ib == common.qoiparams.ibs) {     
                Int f1 = common.fblks[3*j]-1;
                Int f2 = common.fblks[3*j+1];                      
                Int npf = common.grid.npf; // number of nodes on master face      
                Int npe = common.grid.npe; // number of nodes on master face      
                Int nf = f2-f1;
                Int nn = npf*nf; 
                Int nc = common.components.nc; // number of compoments of (u, q, p)            
                Int ncu = common.components.ncu;
                Int ncw = common.components.ncw;
                GetArrayAtIndex(tmp.tempn, sol.udg, &mesh.findudg1[npf*nc*f1], nn*nc);                
                ArrayAXPBY(sol.bouudgavg, sol.bouudgavg, tmp.tempn, one, one, nn*nc);            
                ArrayAddScalar(&sol.bouudgavg[nn*nc], one, 1);
              
                if (common.spatialScheme==1)
                  GetFaceNodesHDG(tmp.tempn, sol.uh, npf, ncu, 0, ncu, f1, f2);
                else
                  GetElemNodes(tmp.tempn, sol.uh, npf, ncu, 0, ncu, f1, f2);
                ArrayAXPBY(sol.bouuhavg, sol.bouuhavg, tmp.tempn, one, one, nn*ncu);            
                ArrayAddScalar(&sol.bouuhavg[nn*ncu], one, 1);                              

                if (ncw>0) {
                    GetFaceNodes(tmp.tempn, sol.wdg, mesh.facecon, npf, ncw, npe, ncw, f1, f2, 1);      
                    ArrayAXPBY(sol.bouwdgavg, sol.bouwdgavg, tmp.tempn, one, one, nn*ncw);            
                    ArrayAddScalar(&sol.bouwdgavg[nn*ncw], one, 1);                                  
                }
            }
        }                                        
    }
}

#endif
