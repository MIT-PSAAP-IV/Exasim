/*
    refinemesh_backend.hpp

    Backend-portable (CPU / CUDA / HIP), MPI-ready high-order uniform mesh
    refinement -- the performant counterpart of the scalar reference in
    refinemesh.{hpp,cpp}. Each child's geometry is the parent's isoparametric map
    evaluated at the child node positions, so curvature is preserved exactly.

    Because every parent shares the child node positions, refinement is a set of
    shared-matrix products (one per child): refined_c = P_c * dgnodes, computed by
    Gauss2Node exactly like the straight-mesh DGProjection/L2eProjection apply.
    Element-local: a rank refines its own elements with no communication.

    P_c[npe x npe x nchild] with P_c[i + npe*(a + npe*c)] = phi_a(xi_child_c[i])
    is built once on the host via mkshape (parent nodal basis at the child node
    positions from refine_child_refnodes) and passed in. Output is CHILD-MAJOR:
    the refined element index is c*ne + e, so each child block is contiguous.
*/
#ifndef __REFINEMESH_BACKEND
#define __REFINEMESH_BACKEND

// refined[npe*ncx*(ne*nchild)] = for each child c: refined(:,:,c*ne+e) = P_c * dgnodes(:,:,e)
template <class T=dstype, class I=Int>
inline void RefineMeshHighOrder(T* refined, T* dgnodes, T* Pc,
        Int npe, Int ncx, Int ne, Int nchild, cublasHandle_t handle, Int backend)
{
    for (Int c = 0; c < nchild; c++) {
        // refined_c[npe x (ncx*ne)] = P_c[npe x npe] * dgnodes[npe x (ncx*ne)]
        Gauss2Node(handle, &refined[npe*ncx*(c*ne)], dgnodes, &Pc[npe*npe*c],
                   npe, npe, ncx*ne, backend);
    }
}

#endif
