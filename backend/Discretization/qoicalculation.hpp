#include <exasim/drivers.hpp>
#include <exasim/detail/driver_dispatch.hpp>

#ifndef __QOICALCULATION
#define __QOICALCULATION

template <class M, class T=dstype, class I=Int>
inline void qoiElemBlock(solstructT<T,I> &sol, resstructT<T,I> &res, appstructT<T,I> &app, masterstructT<T,I> &master, 
        meshstructT<T,I> &mesh, tempstructT<T,I> &tmp, commonstructT<T,I> &common, cublasHandle_t handle, Int jth, Int backend)
{
    using dstype=T;        
    Int nc = common.components.nc; // number of compoments of (u, q, p)
    Int ncu = common.components.ncu;// number of compoments of (u)
    Int ncq = common.components.ncq;// number of compoments of (q)
    Int nco = common.components.nco;// number of compoments of (o)
    Int ncx = common.components.ncx;// number of compoments of (xdg) 
    Int ncs = common.components.ncs;// number of compoments of (sdg) 
    Int ncw = common.components.ncw;// number of compoments of (wdg) 
    Int nd = common.grid.nd;     // spatial dimension    
    Int npe = common.grid.npe; // number of nodes on master element
    Int nge = common.grid.nge; // number of gauss points on master element        

    Int e1 = common.eblks[3*jth]-1;
    Int e2 = common.eblks[3*jth+1];            
    Int ne = e2-e1;
    Int nga = nge*ne;   
    Int n1 = nga*ncx;                  // Xx
    Int n2 = nga*(ncx+nd*nd);          // jac        
    Int nm = nge*e1*(ncx+nd*nd+1);

    dstype *xg = &sol.elemg[nm];
    dstype *Xx = &sol.elemg[nm+n1];
    dstype *jac = &sol.elemg[nm+n2];

    dstype *og = &sol.odgg[nge*nco*e1];
    dstype *uqg = &tmp.tempg[0];
    dstype *wg = &tmp.tempg[nga*nc];    
    dstype *sg = &tmp.tempg[nga*(nc+ncw)];    
            
    GetElemNodes(tmp.tempn, sol.udg, npe, nc, 0, nc, e1, e2);   
    Node2Gauss(handle, uqg, tmp.tempn, master.shapegt, nge, npe, ne*nc, backend);        
    if ((ncw>0) & (common.timeparams.wave==0)) {
        GetElemNodes(tmp.tempn, sol.wdg, npe, ncw, 0, ncw, e1, e2);    
        Node2Gauss(handle, wg, tmp.tempn, master.shapegt, nge, npe, ne*ncw, backend);        
    }
    
    int nvqoi = common.qoiparams.nvqoi;
    ArraySetValue(sg, 0.0, nga*nvqoi);
    EXASIM_DRIVER_CALL(QoIvolumeDriver, sg, xg, uqg, og, wg, mesh, master, app, sol, tmp, common, nge, e1, e2, backend);

    // Weight each Gauss point by ITS OWN Jacobian, not by one scalar per component.
    //
    // This used to read ApplyJac(sg, jac, nge*ne, nge*ne*nvqoi). ApplyJac is
    // (kokkosimpl.h:2692) `R[n] *= jac[n/M]` -- a broadcast whose jac is indexed by the
    // SLOW dimension, documented there as one entry per ELEMENT. Here jac is
    // &sol.elemg[...], one entry per GAUSS POINT (nga = nge*ne of them), and M was passed
    // as nga, so n/M ranged over [0, nvqoi): the whole of QoI component k was scaled by
    // the single number jac[k] -- the Jacobian at Gauss point k of the block's FIRST
    // element -- instead of each point being scaled by its own.
    //
    // What that looked like: on a uniform affine mesh every jac is the same number, so the
    // wrong expression is arithmetically the right one and the QoI is correct and
    // partition-invariant -- which is why poisson2d-new-architecture reproduces
    // 1.806679e-01 / 3.871878e-01 bit-for-bit at np = 2..48. On a graded or curved mesh it
    // is not: the reported QoI becomes (an arbitrary element's Jacobian) x (an unweighted
    // sum), and WHICH element is first in each block is decided by the partition and the
    // element-block split. isoq3d-poisson (curved 3D cone mesh) therefore reported
    // int u = 3.932279e-04 on one rank and 5.723467e-04 on two, from a bit-identical
    // solution field -- same residual (1.78972e-03), same GMRES history, same interface
    // flux to six digits. The two components scaled by DIFFERENT factors (jac[0] and
    // jac[1]), so even the QoI2/QoI1 ratio moved, which is what rules out a
    // "different set of elements" explanation.
    //
    // columnwiseMultiply(C, A, b, N, M) is `C[i] = A[i]*b[i%N]` and is the idiom used for
    // exactly this everywhere else in the backend (uequation.hpp:314,
    // ldgblockjacobian.cpp:976). N = nga selects the Gauss-point index.
    columnwiseMultiply(sg, sg, jac, nga, nvqoi);
    Gauss2Node(handle, tmp.tempn, sg, master.gwe, nge, 1, nvqoi*ne, backend);
    
    ArraySetValue(tmp.tempg, 1.0, ne);
    for (int i = 0; i<nvqoi; i++) {
        dstype dotprod = 0;
        LDOT(handle, ne, tmp.tempg, inc1, &tmp.tempn[i*ne], inc1, &dotprod, backend);
        common.qoiparams.qoivolume[i] += dotprod;
    }
}

template <class M, class T=dstype, class I=Int>
inline void qoiElement(solstructT<T,I> &sol, resstructT<T,I> &res, appstructT<T,I> &app, masterstructT<T,I> &master, 
        meshstructT<T,I> &mesh, tempstructT<T,I> &tmp, commonstructT<T,I> &common)
{
    using dstype=T;    
    for (int i = 0; i<common.qoiparams.nvqoi; i++) common.qoiparams.qoivolume[i] = 0.0;
    for (Int j=0; j<common.meshsizes.nbe; j++) {              
        Int e2 = common.eblks[3*j+1];            
        if (e2 <= common.meshsizes.ne1) qoiElemBlock<M>(sol, res, app, master, mesh, tmp, common, common.cublasHandle, j, common.backend);        
    }                     

    // ONE global reduction for the whole QoI vector, AFTER the block loop.
    //
    // PDOT is COLLECTIVE (pblas.h: MPI_Allreduce over EXASIM_COMM_WORLD). It used to be
    // called from inside qoiElemBlock, i.e. once per block per QoI component. The
    // number of blocks passing the guard above is PARTITION-DEPENDENT, so different
    // ranks issued different numbers of MPI_Allreduce calls -- a collective count
    // mismatch. Ranks with fewer blocks left this function early and ran on into the
    // next communication, while ranks with more blocks waited forever for peers that
    // were never coming. Reproduced as a hard deadlock at np=17 on poisson2d (17 is
    // prime, so the mesh does not divide evenly and the block counts diverge).
    //
    // Accumulating locally with LDOT and reducing once here makes the collective count
    // exactly one per call, independent of the partition -- and removes a latency-bound
    // global barrier from an inner loop.
#ifdef HAVE_MPI
    if (common.qoiparams.nvqoi > 0)
        MPI_Allreduce(MPI_IN_PLACE, common.qoiparams.qoivolume, common.qoiparams.nvqoi,
                      mpi_type<dstype>(), MPI_SUM, EXASIM_COMM_WORLD);
#endif
}

template <class M, class T=dstype, class I=Int>
inline void qoiFaceBlock(solstructT<T,I> &sol, resstructT<T,I> &res, appstructT<T,I> &app, masterstructT<T,I> &master, 
        meshstructT<T,I> &mesh, tempstructT<T,I> &tmp, commonstructT<T,I> &common, 
        cublasHandle_t handle, Int f1, Int f2, Int ib, Int backend)
{
    using dstype=T;            
    Int nc = common.components.nc; // number of compoments of (u, q, p)
    Int ncu = common.components.ncu;// number of compoments of (u)
    Int nco = common.components.nco;// number of compoments of (o)
    Int ncx = common.components.ncx;// number of compoments of (xdg)        
    Int ncw = common.components.ncw;
    Int nd = common.grid.nd;     // spatial dimension    
    Int npe = common.grid.npe; // number of nodes on master element
    Int npf = common.grid.npf; // number of nodes on master face           
    Int ngf = common.grid.ngf; // number of gauss poInts on master face              

    Int nf = f2-f1;
    Int nn =  npf*nf; 
    Int nga = ngf*nf;   
    Int nm = ngf*f1*(ncx+nd+1);
    Int n1 = nga*ncx;                           // nlg
    Int n2 = nga*(ncx+nd);                      // jac

    GetElemNodes(tmp.tempn, sol.uh, npf, ncu, 0, ncu, f1, f2);        
    //GetFaceNodes(tmp.tempn, sol.udg, mesh.facecon, npf, nc, npe, nc, f1, f2, 1, backend);      
    GetArrayAtIndex(&tmp.tempn[nn*ncu], sol.udg, &mesh.findudg1[npf*nc*f1], nn*nc);        
    if (ncw>0)
        GetFaceNodes(&tmp.tempn[nn*(ncu+nc)], sol.wdg, mesh.facecon, npf, ncw, npe, ncw, f1, f2, 1);             
    Node2Gauss(handle, tmp.tempg, tmp.tempn, master.shapfgt, ngf, npf, nf*(ncu+nc+ncw), backend);
        
    int nsurf = common.qoiparams.nsurf;     
    ArraySetValue(tmp.tempn, 0.0, nga*nsurf);     
    EXASIM_DRIVER_CALL(QoIboundaryDriver, tmp.tempn, &sol.faceg[nm], &tmp.tempg[nga*ncu], &sol.og1[ngf*nco*f1], 
            &tmp.tempg[nga*(ncu+nc)], &tmp.tempg[0], &sol.faceg[nm+n1], mesh, master, app, 
            sol, tmp, common, ngf, f1, f2, ib, backend);        

    // Same defect as in qoiElemBlock above, on the surface QoI: &sol.faceg[nm+n2] is the
    // face Jacobian at each of the nga = ngf*nf face Gauss points, but ApplyJac indexed it
    // by the component, so component k of the boundary QoI was scaled by jac[k] alone.
    // Invisible with nsurf == 1 on a mesh of identical faces; wrong on any curved or
    // graded boundary, and partition-dependent because the block's first face changes.
    columnwiseMultiply(tmp.tempn, tmp.tempn, &sol.faceg[nm+n2], nga, nsurf);
    Gauss2Node(handle, tmp.tempg, tmp.tempn, master.gwf, ngf, 1, nsurf*nf, backend);

    ArraySetValue(tmp.tempn, 1.0, nf);
    for (int i = 0; i<nsurf; i++) {
        dstype dotprod = 0;
        LDOT(handle, nf, tmp.tempn, inc1, &tmp.tempg[i*nf], inc1, &dotprod, backend);
        common.qoiparams.qoisurface[i] += dotprod;
    }        
}

template <class M, class T=dstype, class I=Int>
inline void qoiFace(solstructT<T,I> &sol, resstructT<T,I> &res, appstructT<T,I> &app, masterstructT<T,I> &master, 
        meshstructT<T,I> &mesh, tempstructT<T,I> &tmp, commonstructT<T,I> &common)
{
    using dstype=T;    
    for (int i = 0; i<common.qoiparams.nsurf; i++) common.qoiparams.qoisurface[i] = 0.0;
    for (Int j=0; j<common.meshsizes.nbf; j++) {
        Int f1 = common.fblks[3*j]-1;
        Int f2 = common.fblks[3*j+1];    
        Int ib = common.fblks[3*j+2];    
        if ((common.qoiparams.ibs > 0) && (ib == common.qoiparams.ibs))
            qoiFaceBlock<M>(sol, res, app, master, mesh, tmp, common, common.cublasHandle, f1, f2, 1, common.backend);
    }                          

    // ONE global reduction for the whole QoI vector, AFTER the block loop.
    //
    // PDOT is COLLECTIVE (pblas.h: MPI_Allreduce over EXASIM_COMM_WORLD). It used to be
    // called from inside qoiFaceBlock, i.e. once per block per QoI component. The
    // number of blocks passing the guard above is PARTITION-DEPENDENT, so different
    // ranks issued different numbers of MPI_Allreduce calls -- a collective count
    // mismatch. Ranks with fewer blocks left this function early and ran on into the
    // next communication, while ranks with more blocks waited forever for peers that
    // were never coming. Reproduced as a hard deadlock at np=17 on poisson2d (17 is
    // prime, so the mesh does not divide evenly and the block counts diverge).
    //
    // Accumulating locally with LDOT and reducing once here makes the collective count
    // exactly one per call, independent of the partition -- and removes a latency-bound
    // global barrier from an inner loop.
#ifdef HAVE_MPI
    if (common.qoiparams.nsurf > 0)
        MPI_Allreduce(MPI_IN_PLACE, common.qoiparams.qoisurface, common.qoiparams.nsurf,
                      mpi_type<dstype>(), MPI_SUM, EXASIM_COMM_WORLD);
#endif
}

#endif

