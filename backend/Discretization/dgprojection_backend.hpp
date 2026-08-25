/*
    dgprojection_backend.hpp

    Batched, backend-portable (CPU / CUDA / HIP) L2 projection of a DG field
    from a source nodal basis onto a target nodal basis, element by element.
    This is the performant counterpart of the portable scalar reference in
    dgprojection.{hpp,cpp} (the port of frontends/Matlab/Utilities/dgprojection.m)
    and is built the same way ComputeMinv (massinv.hpp) builds and applies the
    element mass matrix:

        M(e) = shapegw_t * ShapJac(shapegt_t, jac(e))     [npe_t x npe_t]   (mass, target basis)
        C(e) = shapegw_t * ShapJac(shapegt_s, jac(e))     [npe_t x npe_s]   (cross-mass, target<-source)
        U1(:,:,e) = M(e)^{-1} * ( C(e) * U(:,:,e) )                          (per element)

    where shapegt_t / shapegw_t are the target master's Gauss-point shape values
    (and weighted values), and shapegt_s is the SOURCE basis evaluated at the
    TARGET master's Gauss points (master.gpe): shapegs[nge*npe_s], laid out like
    master.shapegt, i.e. the values block of
    mkshape(porder_source, masternodes(porder_source,...), master.gpe, elemtype).

    Two things make this the backend-style implementation the solver wants:

    * GPU-capable: all heavy work is the existing device primitives
      (ShapJac / Gauss2Node / Inverse / ArrayGemmBatch1), dispatched by `backend`.
    * MPI-capable: the projection is element-local, so operating on this rank's
      element blocks (common.eblks) is all that is needed — no halo exchange.

    Straight vs curved mesh. For a straight (affine) element the Jacobian is a
    per-element constant and cancels in M^{-1} C: M = jac*M0, C = jac*C0, so
    M^{-1} C = M0^{-1} C0 = P0, the SAME reference operator for every element.
    We therefore build P0 = M0^{-1} C0 once on the master element and apply it to
    every element with a single shared-matrix gemm per block. (This mirrors the
    straight-mesh fast path in ComputeMinv, which likewise treats jac as one
    value per element.) Only curved meshes take the per-element path.
*/
#ifndef __DGPROJECTION_BACKEND
#define __DGPROJECTION_BACKEND

// NB: U and shapegs are non-const because the backend BLAS wrappers
// (Gauss2Node/Node2Gauss) take mutable T* for all matrix operands, matching the
// rest of the backend (ComputeMinv et al.). They are not modified here.
template <class T=dstype, class I=Int>
inline void DGProjection(T* U1, T* U, T* shapegs, Int npe_s, Int nc,
        solstructT<T,I> &sol, resstructT<T,I> &res, appstructT<T,I> &app, masterstructT<T,I> &master,
        meshstructT<T,I> &mesh, tempstructT<T,I> &tmp, commonstructT<T,I> &common, cublasHandle_t handle, Int backend)
{
    using dstype=T;

    Int ncx = common.components.ncx;   // number of components of (xdg)
    Int nd  = common.grid.nd;          // spatial dimension
    Int npe = common.grid.npe;         // nodes on the master (target) element
    Int nge = common.grid.nge;         // Gauss points on the master element
    Int nbe = common.meshsizes.nbe;    // number of element blocks
    Int neb = common.meshsizes.neb;    // max elements per block

    Int npemax = (npe_s > npe) ? npe_s : npe;

    // scratch shared by both paths (sized for the largest block)
    dstype *Minv=NULL;   // element mass inverse (curved) / master mass inverse (straight)
    dstype *Cmat=NULL;   // cross-mass C (curved) / master C0 (straight)
    dstype *work=NULL;   // ShapJac output / gemm scratch
    dstype *invw=NULL;   // Inverse() workspace
    dstype *L=NULL;      // C * U per element (curved)
    Int    *ipiv=NULL;

    TemplateMalloc(&Cmat, npe*npemax*neb, backend);
    TemplateMalloc(&work, nge*npemax*neb, backend);
    TemplateMalloc(&invw, npe*npe*neb, backend);
    TemplateMalloc(&ipiv, npe+1, backend);

    if (common.grid.curvedMesh==0) {
        // ---- straight mesh: one shared operator P0 = M0^{-1} C0 ----
        TemplateMalloc(&Minv, npe*npe, backend);       // = M0^{-1}
        dstype *P0 = Cmat;                              // reuse Cmat as C0 then P0 (npe x npe_s)

        // M0 = shapegw_t * shapegt_t   (reference mass, jac = 1)
        Gauss2Node(handle, Minv, master.shapegt, master.shapegw, nge, npe, npe, backend);
        Inverse(handle, Minv, invw, ipiv, npe, 1, backend);
        // C0 = shapegw_t * shapegs
        Gauss2Node(handle, work, shapegs, master.shapegw, nge, npe, npe_s, backend);  // work = C0 [npe x npe_s]
        // P0 = M0^{-1} * C0                                   [npe x npe_s]
        // (ArrayMatrixMultiplication1 accumulates into P0, so zero it first)
        ArraySetValue(P0, zero, npe*npe_s);
        ArrayMatrixMultiplication1(P0, Minv, work, npe, npe_s, npe);

        for (Int j=0; j<nbe; j++) {
            Int e1 = common.eblks[3*j]-1;
            Int e2 = common.eblks[3*j+1];
            Int ns = e2-e1;
            // U1(:,:,e) = P0 * U(:,:,e), shared operator across the block:
            //   U1blk[npe x nc*ns] = P0[npe x npe_s] * Ublk[npe_s x nc*ns]
            Gauss2Node(handle, &U1[npe*nc*e1], &U[npe_s*nc*e1], P0, npe_s, npe, nc*ns, backend);
        }

        TemplateFree(Minv, backend);
    }
    else {
        // ---- curved mesh: per-element M(e), C(e), then M^{-1}(C U) ----
        TemplateMalloc(&Minv, npe*npe*neb, backend);
        TemplateMalloc(&L,    npe*nc*neb, backend);

        for (Int j=0; j<nbe; j++) {
            Int e1 = common.eblks[3*j]-1;
            Int e2 = common.eblks[3*j+1];
            Int ns = e2-e1;
            Int nga = nge*ns;
            Int n0 = 0;                     // xg
            Int n1 = nga*ncx;               // Xx
            Int n2 = nga*(ncx+nd*nd);       // jac
            Int n3 = nga*(ncx+nd*nd+1);     // Jg

            // --- geometry: jac at Gauss points (identical to ComputeMinv) ---
            GetElemNodes(tmp.tempn, sol.xdg, npe, ncx, 0, ncx, e1, e2);
            Node2Gauss(handle, &tmp.tempg[n0], tmp.tempn, master.shapegt, nge, npe, ns*ncx, backend);
            if (nd==1) {
                Node2Gauss(handle, &tmp.tempg[n3], tmp.tempn, &master.shapegt[nge*npe], nge, npe, ns*nd, backend);
                ElemGeom1D(&tmp.tempg[n2], &tmp.tempg[n1], &tmp.tempg[n3], nga);
            }
            else if (nd==2) {
                Node2Gauss(handle, &tmp.tempg[n3], tmp.tempn, &master.shapegt[nge*npe], nge, npe, ns*nd, backend);
                Node2Gauss(handle, &tmp.tempg[n3+nga*nd], tmp.tempn, &master.shapegt[2*nge*npe], nge, npe, ns*nd, backend);
                ElemGeom2D(&tmp.tempg[n2], &tmp.tempg[n1], &tmp.tempg[n1+2*nga], &tmp.tempg[n1+nga], &tmp.tempg[n1+3*nga],
                    &tmp.tempg[n3], &tmp.tempg[n3+nga], &tmp.tempg[n3+2*nga], &tmp.tempg[n3+3*nga], nga);
            }
            else if (nd==3) {
                Node2Gauss(handle, &tmp.tempg[n3], tmp.tempn, &master.shapegt[nge*npe], nge, npe, ns*nd, backend);
                Node2Gauss(handle, &tmp.tempg[n3+nga*nd], tmp.tempn, &master.shapegt[2*nge*npe], nge, npe, ns*nd, backend);
                Node2Gauss(handle, &tmp.tempg[n3+2*nga*nd], tmp.tempn, &master.shapegt[3*nge*npe], nge, npe, ns*nd, backend);
                ElemGeom3D(&tmp.tempg[n2], &tmp.tempg[n1], &tmp.tempg[n1+3*nga], &tmp.tempg[n1+6*nga],
                    &tmp.tempg[n1+nga], &tmp.tempg[n1+4*nga], &tmp.tempg[n1+7*nga],
                    &tmp.tempg[n1+2*nga], &tmp.tempg[n1+5*nga], &tmp.tempg[n1+8*nga],
                    &tmp.tempg[n3], &tmp.tempg[n3+nga], &tmp.tempg[n3+2*nga],
                    &tmp.tempg[n3+3*nga], &tmp.tempg[n3+4*nga], &tmp.tempg[n3+5*nga],
                    &tmp.tempg[n3+6*nga], &tmp.tempg[n3+7*nga], &tmp.tempg[n3+8*nga], nga);
            }

            // --- M(e) = shapegw_t * ShapJac(shapegt_t, jac); invert in place ---
            ShapJac(work, master.shapegt, &tmp.tempg[n2], nge, npe, ns);
            Gauss2Node(handle, Minv, work, master.shapegw, nge, npe, npe*ns, backend);
            Inverse(handle, Minv, invw, ipiv, npe, ns, backend);

            // --- C(e) = shapegw_t * ShapJac(shapegt_s, jac) ---
            ShapJac(work, shapegs, &tmp.tempg[n2], nge, npe_s, ns);
            Gauss2Node(handle, Cmat, work, master.shapegw, nge, npe, npe_s*ns, backend);

            // --- L = C(e) * U(:,:,e) ; U1(:,:,e) = M(e)^{-1} * L ---
            ArraySetValue(L, zero, npe*nc*ns);
            ArrayGemmBatch1(L, Cmat, &U[npe_s*nc*e1], npe, nc, npe_s, ns);
            ArraySetValue(&U1[npe*nc*e1], zero, npe*nc*ns);
            ArrayGemmBatch1(&U1[npe*nc*e1], Minv, L, npe, nc, npe, ns);
        }

        TemplateFree(Minv, backend);
        TemplateFree(L, backend);
    }

    TemplateFree(Cmat, backend);
    TemplateFree(work, backend);
    TemplateFree(invw, backend);
    TemplateFree(ipiv, backend);
}

#endif
