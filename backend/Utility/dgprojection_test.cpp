#include "dgprojection.hpp"

#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool nearlyEqual(double a, double b, double tol = 1.0e-11)
{
    return std::abs(a - b) <= tol * (1.0 + std::abs(b));
}

void expectClose(double actual, double expected, const std::string& label)
{
    if (!nearlyEqual(actual, expected)) {
        throw std::runtime_error(label + ": expected " + std::to_string(expected) +
                                 ", got " + std::to_string(actual));
    }
}

void expectThrows(const std::string& label, const std::function<void()>& fn)
{
    bool threw = false;
    try {
        fn();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    if (!threw) {
        throw std::runtime_error(label + ": expected runtime_error");
    }
}

// 3-point Gauss-Legendre on the reference interval [0,1] (exact to degree 5).
struct Gauss1D {
    std::array<double, 3> xi;
    std::array<double, 3> w;
    Gauss1D()
    {
        const double a = std::sqrt(0.6);
        xi = {0.5 * (1.0 - a), 0.5, 0.5 * (1.0 + a)};
        w  = {5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0};
    }
};

// 1D nodal Lagrange bases on [0,1].
std::array<double, 2> p1(double xi) { return {1.0 - xi, xi}; }
std::array<double, 2> dp1(double)   { return {-1.0, 1.0}; }
std::array<double, 3> p2(double xi)
{
    return {2.0 * xi * xi - 3.0 * xi + 1.0,   // node 0.0
            -4.0 * xi * xi + 4.0 * xi,        // node 0.5
            2.0 * xi * xi - xi};              // node 1.0
}
std::array<double, 3> dp2(double xi)
{
    return {4.0 * xi - 3.0, -8.0 * xi + 4.0, 4.0 * xi - 1.0};
}

} // namespace

int main()
{
    // ---------------------------------------------------------------
    // Test A: volgeom_det against hand-computed determinants (1D/2D/3D).
    // ---------------------------------------------------------------
    {
        const double J1[1] = {3.5};
        expectClose(volgeom_det(J1, 1), 3.5, "volgeom 1D");

        // J[a*nd+b] = dx_b/dxi_a. det = J0*J3 - J1*J2.
        const double J2[4] = {2.0, 1.0, 0.0, 3.0};
        expectClose(volgeom_det(J2, 2), 6.0, "volgeom 2D");

        // 3x3 with a known determinant (this matrix has det 1).
        const double J3[9] = {2.0, -3.0, 1.0, 2.0, 0.0, -1.0, 1.0, 4.0, 5.0};
        // det = 2*(0*5 - (-1)*4) - (-3)*(2*5 - (-1)*1) + 1*(2*4 - 0*1)
        //     = 2*4 + 3*11 + 8 = 8 + 33 + 8 = 49
        expectClose(volgeom_det(J3, 3), 49.0, "volgeom 3D");

        expectThrows("volgeom bad nd", [&]() { volgeom_det(J3, 4); });
    }

    Gauss1D gq;
    const int nge = 3;
    std::vector<double> gw(gq.w.begin(), gq.w.end());

    // Build a shape-value matrix [np x nge] (column-major) from a basis fn.
    auto buildShape = [&](int np, const std::function<std::vector<double>(double)>& basis) {
        std::vector<double> S(np * nge);
        for (int g = 0; g < nge; ++g) {
            std::vector<double> phi = basis(gq.xi[g]);
            for (int i = 0; i < np; ++i) S[i + np * g] = phi[i];
        }
        return S;
    };
    // Build a 1D derivative-shape array [np x nge x 1] (nd=1) from a basis-deriv fn.
    auto buildDShape1D = [&](int np, const std::function<std::vector<double>(double)>& dbasis) {
        std::vector<double> D(np * nge * 1);
        for (int g = 0; g < nge; ++g) {
            std::vector<double> d = dbasis(gq.xi[g]);
            for (int i = 0; i < np; ++i) D[i + np * (g + nge * 0)] = d[i];
        }
        return D;
    };
    auto asVec2 = [](const std::function<std::array<double, 2>(double)>& f) {
        return [f](double xi) { auto a = f(xi); return std::vector<double>(a.begin(), a.end()); };
    };
    auto asVec3 = [](const std::function<std::array<double, 3>(double)>& f) {
        return [f](double xi) { auto a = f(xi); return std::vector<double>(a.begin(), a.end()); };
    };

    // Precompute the shapes we reuse.
    std::vector<double> S_p1  = buildShape(2, asVec2(p1));
    std::vector<double> S_p2  = buildShape(3, asVec3(p2));
    std::vector<double> D_p1  = buildDShape1D(2, asVec2(dp1));
    std::vector<double> D_p2  = buildDShape1D(3, asVec3(dp2));

    // ---------------------------------------------------------------
    // Test B: identity projection (source basis == target basis) on
    // two elements with NON-affine node spacing (varying jac). The
    // cross-mass C equals the mass M, so U1 must reproduce U exactly.
    // This exercises the M/C build, C*U, dense solve and element striding.
    // ---------------------------------------------------------------
    {
        const int npe = 3, nd = 1, ncx = 1, nc = 2, ne = 2;
        // curved / unevenly spaced target nodes per element (node 0.5 pulled off-centre)
        std::vector<double> dgnodes = {
            2.0, 3.1, 5.0,   // element 0
            0.0, 0.7, 1.0    // element 1
        };
        std::vector<double> U = {
            // element 0: two components (col-major [npe x nc])
            7.0, -3.0, 2.0,   1.0, 1.0, 1.0,
            // element 1
            -4.0, 6.0, 0.5,   9.0, -2.0, 3.0
        };
        std::vector<double> U1(npe * nc * ne, 0.0);
        dgprojection(U1.data(), U.data(), dgnodes.data(),
                     S_p2.data(), D_p2.data(), S_p2.data(), gw.data(),
                     npe, npe, nge, nd, ncx, nc, ne);
        for (std::size_t i = 0; i < U.size(); ++i) {
            expectClose(U1[i], U[i], "identity idx " + std::to_string(i));
        }
    }

    // ---------------------------------------------------------------
    // Test C: cross-basis exactness. Project a LINEAR field p1 -> p2 on an
    // affine element. Because p2 contains linears, the L2 projection is
    // exact: U1 = f at the three p2 node coordinates.
    //   element [2,5], p2 nodes at {2, 3.5, 5}; f(x) = 1 + 2x
    //   source p1 values at {2,5} = {5, 11}; expected p2 = {5, 8, 11}
    // ---------------------------------------------------------------
    {
        const int nd = 1, ncx = 1, nc = 1, ne = 1;
        std::vector<double> dgnodes = {2.0, 3.5, 5.0};   // target = p2 (3 nodes)
        std::vector<double> U = {5.0, 11.0};             // source = p1 (2 nodes)
        std::vector<double> U1(3, 0.0);
        dgprojection(U1.data(), U.data(), dgnodes.data(),
                     S_p2.data(), D_p2.data(), S_p1.data(), gw.data(),
                     /*npe_t*/3, /*npe_s*/2, nge, nd, ncx, nc, ne);
        expectClose(U1[0], 5.0,  "p1->p2 node 0");
        expectClose(U1[1], 8.0,  "p1->p2 node 1");
        expectClose(U1[2], 11.0, "p1->p2 node 2");

        // Round trip p2 -> p1 recovers the linear field at the endpoints.
        std::vector<double> dgnodes_p1 = {2.0, 5.0};     // target = p1 (2 nodes)
        std::vector<double> back(2, 0.0);
        dgprojection(back.data(), U1.data(), dgnodes_p1.data(),
                     S_p1.data(), D_p1.data(), S_p2.data(), gw.data(),
                     /*npe_t*/2, /*npe_s*/3, nge, nd, ncx, nc, ne);
        expectClose(back[0], 5.0,  "p2->p1 node 0 (round trip)");
        expectClose(back[1], 11.0, "p2->p1 node 1 (round trip)");
    }

    // ---------------------------------------------------------------
    // Test D: unsupported dimension is rejected.
    // ---------------------------------------------------------------
    {
        std::vector<double> dummy(4, 0.0);
        expectThrows("dgprojection bad nd", [&]() {
            dgprojection(dummy.data(), dummy.data(), dummy.data(),
                         dummy.data(), dummy.data(), dummy.data(), dummy.data(),
                         1, 1, 1, /*nd*/4, 1, 1, 1);
        });
    }

    std::cout << "dgprojection tests passed\n";
    return 0;
}
