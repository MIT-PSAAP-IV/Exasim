#pragma once

// l2eprojection.hpp
//
// Host-side scalar reference for frontends/Matlab/Utilities/l2eprojection.m:
// the Galerkin L2 projection of a function onto the DG space, element by
// element. Given f sampled at the element Gauss points (fg), it solves
//
//     M(e) U1(:,:,e) = F(e),   M_ab = INT phi_a phi_b jac,   F_ac = INT phi_a f_c jac
//
// per element -- the standard way to build a DG field from an analytic initial
// condition / manufactured solution, or to set up an L2 error.
//
// The func -> fg step (evaluate f at the physical Gauss points pg = shapv * dg)
// is left to the caller so this core is pure linear algebra and GPU-portable;
// the standalone test supplies fg for a known analytic f and checks exactness.
//
// All arrays column-major (MATLAB / Fortran), matching the Exasim master struct:
//   shapv   [nge * npe]        gauss-major shape values      (master.shapegt values block)
//   dshapv  [nge * npe * nd]   gauss-major shape derivatives (master.shapegt deriv blocks)
//   gw      [nge]              Gauss weights                 (master.gwe)
//   dgnodes [npe * ncx * ne]   element node coords (first nd used)
//   fg      [nge * nc * ne]    f sampled at the element Gauss points
//   UDG     [npe * nc * ne]    projected field (out; caller-allocated)
//
// The performant backend-portable (CPU/CUDA/HIP), MPI-ready kernel is
// L2eProjection in l2eprojection_backend.hpp; this is the scalar reference/oracle.

void l2eprojection(double* UDG, const double* fg, const double* dgnodes,
                   const double* shapv, const double* dshapv, const double* gw,
                   int npe, int nge, int nd, int ncx, int nc, int ne);
