#include "extrudesol.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {
void checkdims(int np2d, int nc, int ne2d, int porder, int nz)
{
    if (np2d < 1 || nc < 1 || ne2d < 1 || porder < 0 || nz < 1)
        throw std::runtime_error("extrude: np2d, nc, ne2d, nz must be >= 1 and porder >= 0");
}
} // namespace

void extrudesol(double* UDG3D, const double* UDG2D,
                int np2d, int nc, int ne2d, int porder, int nz)
{
    checkdims(np2d, nc, ne2d, porder, nz);
    const int np1d = porder + 1;
    const int N3   = np2d * np1d;   // 3D nodes per element
    const int NE3  = ne2d * nz;     // 3D elements
    for (int e3 = 0; e3 < NE3; ++e3) {
        const int c = e3 % ne2d;    // 2D element (slab e = e3/ne2d is unused: value is z-invariant)
        for (int b = 0; b < nc; ++b) {
            for (int n3 = 0; n3 < N3; ++n3) {
                const int a = n3 % np2d;   // 2D node (layer d = n3/np2d is unused)
                UDG3D[n3 + static_cast<std::size_t>(N3) * (b + nc * e3)] =
                    UDG2D[a + static_cast<std::size_t>(np2d) * (b + nc * c)];
            }
        }
    }
}

void extrudecoord(double* zdg, const double* zz, const double* plc1d,
                  int np2d, int nc, int ne2d, int porder, int nz)
{
    checkdims(np2d, nc, ne2d, porder, nz);
    const int np1d = porder + 1;
    const int N3   = np2d * np1d;
    const int NE3  = ne2d * nz;
    for (int e3 = 0; e3 < NE3; ++e3) {
        const int e = e3 / ne2d;             // slab
        const double z0 = zz[e], dz = zz[e + 1] - zz[e];
        for (int b = 0; b < nc; ++b) {
            for (int n3 = 0; n3 < N3; ++n3) {
                const int d = n3 / np2d;     // 1D layer
                zdg[n3 + static_cast<std::size_t>(N3) * (b + nc * e3)] = z0 + dz * plc1d[d];
            }
        }
    }
}

void extrudevelocity(double* vx3d, double* vy3d, const double* vr2d,
                     const double* tt, const double* plc1d,
                     int np2d, int nc, int ne2d, int porder, int nz)
{
    checkdims(np2d, nc, ne2d, porder, nz);
    const int np1d = porder + 1;
    const std::size_t N = static_cast<std::size_t>(np2d) * np1d * nc * ne2d * nz;

    std::vector<double> vr3d(N), theta(N);
    extrudesol(vr3d.data(), vr2d, np2d, nc, ne2d, porder, nz);
    extrudecoord(theta.data(), tt, plc1d, np2d, nc, ne2d, porder, nz);
    for (std::size_t i = 0; i < N; ++i) {
        vx3d[i] = vr3d[i] * std::cos(theta[i]);
        vy3d[i] = vr3d[i] * std::sin(theta[i]);
    }
}
