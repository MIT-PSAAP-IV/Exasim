/*
    l2eprojection_backend.hpp

    Backend-portable (CPU / CUDA / HIP), MPI-ready Galerkin L2 projection of a
    Gauss-sampled load onto the DG space -- the performant counterpart of the
    scalar reference in l2eprojection.{hpp,cpp} (port of
    frontends/Matlab/Utilities/l2eprojection.m). Built exactly like DGProjection
    (dgprojection_backend.hpp) / ComputeMinv (massinv.hpp):

        M(e) = shapegw * ShapJac(shapegt, jac(e))          [npe x npe]  (mass)
        F(e) = shapegw * ( jac(e) .* fg(:,:,e) )           [npe x nc]   (load)
        U1(:,:,e) = M(e)^{-1} F(e)

    where fg[nge, nc, ne] is f sampled at the target element Gauss points (the
    caller evaluates f at the physical Gauss points; keeping that out of the
    kernel makes the device path pure linear algebra). Element-local: a rank
    projects its own elements with no communication.

    Straight (affine) mesh: jac is a per-element constant and cancels in M^{-1}F
    (M = jac*M0, F = jac*(shapegw*fg)), so build one shared reference inverse
    M0^{-1} and apply it to every element's load -- the same fast path as
    DGProjection / ComputeMinv. Curved meshes take the per-element path.
*/
#ifndef __L2EPROJECTION_BACKEND
#define __L2EPROJECTION_BACKEND

template <class T=dstype, class I=Int>
inline void L2eProjection(T* U1, T* fg, Int nc,
        solstructT<T,I> &sol, resstructT<T,I> &res, appstructT<T,I> &app, masterstructT<T,I> &master,
        meshstructT<T,I> &mesh, tempstructT<T,I> &tmp, commonstructT<T,I> &common, cublasHandle_t handle, Int backend)
{
    using dstype=T;

    Int ncx = common.components.ncx;
    Int nd  = common.grid.nd;
    Int npe = common.grid.npe;
    Int nge = common.grid.nge;
    Int nbe = common.meshsizes.nbe;
    Int neb = common.meshsizes.neb;

    dstype *Minv=NULL, *work=NULL, *invw=NULL, *Farr=NULL, *wg=NULL;
    Int    *ipiv=NULL;
    TemplateMalloc(&work, nge*npe*neb, backend);
    TemplateMalloc(&invw, npe*npe*neb, backend);
    TemplateMalloc(&Farr, npe*nc*neb, backend);
    TemplateMalloc(&ipiv, npe+1, backend);

    if (common.grid.curvedMesh==0) {
        // ---- straight mesh: shared reference inverse M0^{-1}, applied per block ----
        TemplateMalloc(&Minv, npe*npe, backend);
        Gauss2Node(handle, Minv, master.shapegt, master.shapegw, nge, npe, npe, backend); // M0
        Inverse(handle, Minv, invw, ipiv, npe, 1, backend);                               // M0^{-1}
        for (Int j=0; j<nbe; j++) {
            Int e1 = common.eblks[3*j]-1;
            Int e2 = common.eblks[3*j+1];
            Int ns = e2-e1;
            // F0 = shapegw * fg ; U1 = M0^{-1} F0   (jac cancels for straight elements)
            Gauss2Node(handle, Farr, &fg[nge*nc*e1], master.shapegw, nge, npe, nc*ns, backend);
            Gauss2Node(handle, &U1[npe*nc*e1], Farr, Minv, npe, npe, nc*ns, backend);
        }
        TemplateFree(Minv, backend);
    }
    else {
        // ---- curved mesh: per-element M(e), F(e), then M^{-1}F ----
        TemplateMalloc(&Minv, npe*npe*neb, backend);
        TemplateMalloc(&wg,   nge*nc*neb, backend);

        for (Int j=0; j<nbe; j++) {
            Int e1 = common.eblks[3*j]-1;
            Int e2 = common.eblks[3*j+1];
            Int ns = e2-e1;
            Int nga = nge*ns;
            Int n0 = 0;                     // xg
            Int n1 = nga*ncx;               // Xx
            Int n2 = nga*(ncx+nd*nd);       // jac
            Int n3 = nga*(ncx+nd*nd+1);     // Jg

            // --- geometry: jac at Gauss points (identical to ComputeMinv/DGProjection) ---
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

            // --- M(e) = shapegw * ShapJac(shapegt, jac); invert in place ---
            ShapJac(work, master.shapegt, &tmp.tempg[n2], nge, npe, ns);
            Gauss2Node(handle, Minv, work, master.shapegw, nge, npe, npe*ns, backend);
            Inverse(handle, Minv, invw, ipiv, npe, ns, backend);

            // --- F(e) = shapegw * ( jac .* fg ) ---
            {
                dstype* fgblk = &fg[nge*nc*e1];
                dstype* jacptr = &tmp.tempg[n2];   // [nge x ns]
                Int Nw = nge*nc*ns;
                Kokkos::parallel_for("L2eScaleFgByJac", Nw, KOKKOS_LAMBDA(const size_t idx) {
                    Int g = idx % nge;
                    Int s = (idx / nge) / nc;
                    wg[idx] = fgblk[idx] * jacptr[g + nge*s];
                });
            }
            Gauss2Node(handle, Farr, wg, master.shapegw, nge, npe, nc*ns, backend);

            // --- U1(:,:,e) = M(e)^{-1} F(e) ---
            ArraySetValue(&U1[npe*nc*e1], zero, npe*nc*ns);
            ArrayGemmBatch1(&U1[npe*nc*e1], Minv, Farr, npe, nc, npe, ns);
        }
        TemplateFree(Minv, backend);
        TemplateFree(wg, backend);
    }

    TemplateFree(work, backend);
    TemplateFree(invw, backend);
    TemplateFree(Farr, backend);
    TemplateFree(ipiv, backend);
}

#endif
