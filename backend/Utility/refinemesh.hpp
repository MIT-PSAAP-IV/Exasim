#pragma once

// refinemesh.hpp
//
// Uniform mesh refinement that is compatible with HIGH-ORDER (curved) element
// geometry. Each tensor-product element (quad / hex, elemtype = 1) is split into
// nref^nd children, and each child's npe geometry nodes are set by evaluating the
// PARENT's isoparametric map at the child node positions:
//
//     x_child(xi) = sum_a phi_a(xi) * parent_dgnodes[a]
//
// so curvature is preserved EXACTLY (the children follow the parent's degree-
// porder map, not a straight-sided chord). This is the mesh analog of the
// solution "maneuvering" ports (dgprojection / l2eprojection): the same
// operator that refines the mesh also prolongs a DG field onto the children.
//
// Because every parent of a given (porder, elemtype) shares the child node
// positions, the interpolation is a SHARED operator P_c[npe x npe] per child
// (P_c[i,a] = phi_a(xi_child_c[i]), built once via mkshape), and refinement is a
// shared-matrix batched apply -- the same fast path as the straight-mesh
// projection. The performant backend-portable (CPU/CUDA/HIP), MPI-ready kernel
// is RefineMeshHighOrder in refinemesh_backend.hpp; this is the scalar reference.
//
// Element ordering of the output is CHILD-MAJOR: the refined element index is
//   e3 = c * ne + e      (child c in [0,nchild), parent e in [0,ne))
// so each child block is contiguous (nchild shared-matrix products). All arrays
// column-major.

// Child reference-node positions in the PARENT reference element, for a tensor
// element subdivided nref times per direction. plocal[npe*nd] are the parent
// reference nodes (master.xpe); xic[npe*nd*nchild] out, nchild = nref^nd,
// child c has integer subcell offset o (lexicographic): xi = (o + plocal)/nref.
void refine_child_refnodes(double* xic, const double* plocal, int npe, int nd, int nref);

// nref^nd for a tensor element in nd dimensions.
int refine_nchild(int nd, int nref);

// Batched high-order refinement apply: refined(:,:,c*ne+e) = P_c * dgnodes(:,:,e).
//   dgnodes [npe * ncx * ne]           parent node coords
//   Pc      [npe * npe * nchild]       Pc[i + npe*(a + npe*c)] = phi_a(xi_child_c[i])
//   refined [npe * ncx * (ne*nchild)]  out (child-major; caller-allocated)
void refinemesh(double* refined, const double* dgnodes, const double* Pc,
                int npe, int ncx, int ne, int nchild);
