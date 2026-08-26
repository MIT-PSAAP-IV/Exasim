#pragma once

// dgprojection.hpp
//
// Host-side C++ port of frontends/Matlab/Utilities/dgprojection.m
//
// L2-projects a per-element discontinuous-Galerkin field from a SOURCE nodal
// basis onto a TARGET nodal basis, one element at a time. This is the standard
// operation for "maneuvering" a solution between polynomial orders (p-refine /
// p-coarsen) on a fixed mesh: given a field sampled at the source element's
// nodes, produce the field sampled at the target element's nodes such that the
// two represent the same function in the L2(element) sense.
//
// For each element e the MATLAB routine computes, using the TARGET element's
// quadrature (weights gw at nge Gauss points) and the physical geometry:
//
//     Jg(g)   = sum_i dshape_t(i,g,:) * dgnodes(i,1:nd,e)   // dx/dxi at Gauss pt g
//     jac(g)  = det(Jg(g))                                  // volgeom()
//     M       = shape_t * diag(gw .* jac) * shape_t^T       // [npe_t x npe_t]
//     C       = shape_t * diag(gw .* jac) * shape_s^T       // [npe_t x npe_s]
//     U1(:,:,e) = M \ ( C * U(:,:,e) )                      // [npe_t x nc]
//
// where shape_t / shape_s are the target / source nodal shape-function values
// evaluated at the TARGET element's Gauss points (source shapes come from
// mkshape(porder, masternodes(porder,...), master.gpe, elemtype) in MATLAB).
//
// All arrays are column-major (MATLAB / Fortran storage), matching how the
// Exasim master/mesh structs already hold this data, so a caller can pass the
// struct fields directly:
//
//   shape_t   [npe_t * nge]        <- master.shapen(:,:,1)
//   dshape_t  [npe_t * nge * nd]   <- master.shapent(:,:,2:nd+1)
//   shape_s   [npe_s * nge]        <- mkshape(porder,...)(:,:,1)
//   gw        [nge]                <- master.gwe
//   dgnodes   [npe_t * ncx * ne]   <- mesh.dgnodes         (only the first nd of ncx cols are read)
//   U         [npe_s * nc * ne]    <- source field
//   U1        [npe_t * nc * ne]    <- projected field (output; caller-allocated)
//
// The routine is deliberately self-contained (no Kokkos / solver structs, no
// BLAS dependency), mirroring the eulereval.* / reynolds_averages_3d.* utility
// ports: it can be compiled and unit-tested on its own.
//
// This host version is the portable SCALAR REFERENCE / oracle. The performant,
// backend-portable (CPU / CUDA / HIP) and MPI-ready implementation is
// DGProjection() in dgprojection_backend.hpp, which expresses the same math as
// batched calls to the existing backend primitives (ShapJac / Gauss2Node /
// Inverse / ArrayGemmBatch1) over this rank's element blocks. The two are kept
// in lockstep by dgprojection_backend_test.cpp.

// Determinant of the nd x nd Jacobian J with layout J[a*nd + b] = dx_b/dxi_a,
// matching frontends/Matlab/Utilities/volgeom.m (nd = 1, 2, 3). Throws
// std::runtime_error for other nd.
double volgeom_det(const double* J, int nd);

// L2 projection between nodal bases, element by element. See file header for
// the array layouts. Throws std::runtime_error on an unsupported nd or a
// singular element mass matrix.
void dgprojection(double* U1,
                  const double* U,
                  const double* dgnodes,
                  const double* shape_t,
                  const double* dshape_t,
                  const double* shape_s,
                  const double* gw,
                  int npe_t,
                  int npe_s,
                  int nge,
                  int nd,
                  int ncx,
                  int nc,
                  int ne);
