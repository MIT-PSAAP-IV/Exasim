#pragma once

// extrudesol.hpp
//
// Host-side scalar reference for the 2D->3D extrusion utilities
//   frontends/Matlab/Utilities/{extrudesol,extrudecoord,extrudevelocity}.m
//
// These "maneuver" a 2D (or axisymmetric) DG solution into a 3D field by
// replicating it across np1d = porder+1 high-order layers and nz extrusion
// slabs. This is the axisymmetric/2D -> 3D initial-field construction used for
// warm-starting 3D runs.
//
// All arrays are column-major (MATLAB / Fortran storage). The extruded layout
// matches extrudesol.m exactly:
//
//   3D node  n3 = a + np2d*d      (a in [0,np2d) 2D node, d in [0,np1d) layer)
//   3D elem  e3 = c + ne2d*e      (c in [0,ne2d) 2D elem, e in [0,nz)   slab)
//   UDG3D[np2d*np1d, nc, ne2d*nz]
//
// and the value at (n3, b, e3) is UDG2D(a, b, c) -- constant along the
// extrusion direction (the field does not vary with the layer d or slab e).
//
// The performant, backend-portable (CPU / CUDA / HIP) and MPI-ready kernels are
// ExtrudeSolution / ExtrudeCoord / ExtrudeVelocity in extrudesol_backend.hpp;
// this header is the portable scalar reference / oracle (no Kokkos), mirroring
// the eulereval.* / dgprojection.* utility ports so it can be unit-tested on its
// own. The two are kept in lockstep by the extrusion self-test.

// Replicate a 2D field into the 3D extruded layout (extrudesol.m).
//   UDG2D [np2d * nc * ne2d]          in
//   UDG3D [np2d*(porder+1) * nc * ne2d*nz]  out (caller-allocated)
void extrudesol(double* UDG3D, const double* UDG2D,
                int np2d, int nc, int ne2d, int porder, int nz);

// The extrusion coordinate field in the extrudesol layout (extrudecoord.m).
//   zz    [nz+1]        extrusion interval endpoints
//   plc1d [porder+1]    1D reference nodes on [0,1] (e.g. masternodes(porder,1,1))
//   zdg   [np2d*(porder+1) * nc * ne2d*nz]  out; value at (n3,b,e3) =
//         zz[e] + (zz[e+1]-zz[e]) * plc1d[d]   (e = e3/ne2d, d = n3/np2d)
void extrudecoord(double* zdg, const double* zz, const double* plc1d,
                  int np2d, int nc, int ne2d, int porder, int nz);

// Extrude a radial velocity through the angles tt and rotate to Cartesian
// (extrudevelocity.m): vx = vr*cos(theta), vy = vr*sin(theta), with
// theta = extrudecoord(tt, ...). vr2d is [np2d*nc*ne2d]; vx3d/vy3d are the
// extruded size [np2d*(porder+1) * nc * ne2d*nz] (caller-allocated).
void extrudevelocity(double* vx3d, double* vy3d, const double* vr2d,
                     const double* tt, const double* plc1d,
                     int np2d, int nc, int ne2d, int porder, int nz);
