/*
    extrudesol_backend.hpp

    Backend-portable (CPU / CUDA / HIP) 2D->3D extrusion kernels: the performant
    counterpart of the scalar reference in extrudesol.{hpp,cpp} (the port of
    frontends/Matlab/Utilities/{extrudesol,extrudecoord,extrudevelocity}.m).

    Each kernel is a single Kokkos::parallel_for over the extruded output, so it
    runs on whatever Kokkos execution space the build targets (Serial/OpenMP on
    CPU, CUDA on NVIDIA, HIP on AMD) -- exactly like the ArrayExtract/ArrayInsert
    index-shuffle kernels in Common/kokkosimpl.h. Every output entry reads its own
    input entry with no cross-element coupling and no reduction, so the operation
    is also MPI-ready by construction: a rank extrudes its own 2D elements into
    their 3D columns with no communication. Pointers must live in the execution
    space's memory (device pointers in a GPU build), as elsewhere in the backend.

    Layout (column-major), matching extrudesol.m:
      3D node n3 = a + np2d*d  (a: 2D node, d in [0,np1d) layer, np1d = porder+1)
      3D elem e3 = c + ne2d*e  (c: 2D elem, e in [0,nz) slab)
      out[np2d*np1d, nc, ne2d*nz]
*/
#ifndef __EXTRUDESOL_BACKEND
#define __EXTRUDESOL_BACKEND

// out[n3,b,e3] = in2d(a=n3%np2d, b, c=e3%ne2d)  (extrudesol.m)
template <class Ty = dstype>
inline void ExtrudeSolution(Ty* out, const Ty* in2d,
        const int np2d, const int nc, const int ne2d, const int np1d, const int nz)
{
    using dstype = Ty;
    const int N3  = np2d * np1d;   // 3D nodes per element
    const int NE3 = ne2d * nz;     // 3D elements
    const int N   = N3 * nc * NE3;
    Kokkos::parallel_for("ExtrudeSolution", N, KOKKOS_LAMBDA(const size_t idx) {
        const int n3 = idx % N3;
        const int r  = idx / N3;
        const int b  = r % nc;
        const int e3 = r / nc;
        const int a  = n3 % np2d;
        const int c  = e3 % ne2d;
        out[idx] = in2d[a + np2d * (b + nc * c)];
    });
}

// zdg[n3,b,e3] = zz[e] + (zz[e+1]-zz[e]) * plc1d[d]   (extrudecoord.m)
template <class Ty = dstype>
inline void ExtrudeCoord(Ty* zdg, const Ty* zz, const Ty* plc1d,
        const int np2d, const int nc, const int ne2d, const int np1d, const int nz)
{
    using dstype = Ty;
    const int N3  = np2d * np1d;
    const int NE3 = ne2d * nz;
    const int N   = N3 * nc * NE3;
    Kokkos::parallel_for("ExtrudeCoord", N, KOKKOS_LAMBDA(const size_t idx) {
        const int n3 = idx % N3;
        const int r  = idx / N3;
        const int e3 = r / nc;
        const int d  = n3 / np2d;
        const int e  = e3 / ne2d;
        zdg[idx] = zz[e] + (zz[e + 1] - zz[e]) * plc1d[d];
    });
}

// vx = vr*cos(theta), vy = vr*sin(theta), theta = extrudecoord(tt,...) (extrudevelocity.m)
template <class Ty = dstype>
inline void ExtrudeVelocity(Ty* vx3d, Ty* vy3d, const Ty* vr2d, const Ty* tt, const Ty* plc1d,
        const int np2d, const int nc, const int ne2d, const int np1d, const int nz)
{
    using dstype = Ty;
    const int N3  = np2d * np1d;
    const int NE3 = ne2d * nz;
    const int N   = N3 * nc * NE3;
    Kokkos::parallel_for("ExtrudeVelocity", N, KOKKOS_LAMBDA(const size_t idx) {
        const int n3 = idx % N3;
        const int r  = idx / N3;
        const int b  = r % nc;
        const int e3 = r / nc;
        const int a  = n3 % np2d;
        const int c  = e3 % ne2d;
        const int d  = n3 / np2d;
        const int e  = e3 / ne2d;
        const Ty vr    = vr2d[a + np2d * (b + nc * c)];
        const Ty theta = tt[e] + (tt[e + 1] - tt[e]) * plc1d[d];
        vx3d[idx] = vr * Kokkos::cos(theta);
        vy3d[idx] = vr * Kokkos::sin(theta);
    });
}

#endif
