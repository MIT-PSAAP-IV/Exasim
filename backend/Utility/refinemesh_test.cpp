#include "refinemesh.hpp"

#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
bool nearlyEqual(double a, double b, double tol = 1.0e-12) { return std::abs(a - b) <= tol * (1.0 + std::abs(b)); }
void expectClose(double a, double b, const std::string& l)
{
    if (!nearlyEqual(a, b)) throw std::runtime_error(l + ": expected " + std::to_string(b) + ", got " + std::to_string(a));
}
// p2 nodal Lagrange through {0, 0.5, 1}
std::array<double, 3> lag2(double x) { return {2*x*x - 3*x + 1, -4*x*x + 4*x, 2*x*x - x}; }
} // namespace

int main()
{
    // ---------------------------------------------------------------
    // child reference-node positions, 1D nref=2 (p2 nodes {0,0.5,1})
    // ---------------------------------------------------------------
    {
        const int npe = 3, nd = 1, nref = 2;
        std::vector<double> plocal = {0.0, 0.5, 1.0};
        std::vector<double> xic(npe * nd * refine_nchild(nd, nref));
        refine_child_refnodes(xic.data(), plocal.data(), npe, nd, nref);
        // child 0: {0, 0.25, 0.5}; child 1: {0.5, 0.75, 1}
        const std::vector<double> exp = {0, 0.25, 0.5,  0.5, 0.75, 1.0};
        for (std::size_t i = 0; i < exp.size(); ++i) expectClose(xic[i], exp[i], "xic1d " + std::to_string(i));
    }
    // child positions, 2D nref=2 (p1 quad corners) -> 4 children tile the parent
    {
        const int npe = 4, nd = 2, nref = 2;
        std::vector<double> plocal = {0,1,0,1,  0,0,1,1};   // [4 x 2] col-major
        std::vector<double> xic(npe * nd * refine_nchild(nd, nref));
        refine_child_refnodes(xic.data(), plocal.data(), npe, nd, nref);
        // child 0 (offset (0,0)) = plocal/2
        expectClose(xic[0 + npe * (0 + nd * 0)], 0.0, "xic2d c0 n0 x");
        expectClose(xic[1 + npe * (0 + nd * 0)], 0.5, "xic2d c0 n1 x");
        expectClose(xic[2 + npe * (1 + nd * 0)], 0.5, "xic2d c0 n2 y");
        // child 3 (offset (1,1)) node 3 -> (1,1)
        expectClose(xic[3 + npe * (0 + nd * 3)], 1.0, "xic2d c3 n3 x");
        expectClose(xic[3 + npe * (1 + nd * 3)], 1.0, "xic2d c3 n3 y");
    }

    // ---------------------------------------------------------------
    // high-order refinement apply: a CURVED p2 1D element, nref=2.
    // parent nodes x = {0, 0.4, 1} -> quadratic map x(xi). Children must follow
    // the curve (NOT the chord), share the interior node, and keep the corners.
    // ---------------------------------------------------------------
    {
        const int npe = 3, nd = 1, ncx = 1, ne = 1, nref = 2;
        const int nchild = refine_nchild(nd, nref);              // 2
        std::vector<double> plocal = {0.0, 0.5, 1.0};
        std::vector<double> xic(npe * nd * nchild);
        refine_child_refnodes(xic.data(), plocal.data(), npe, nd, nref);
        // Pc[i + npe*(a + npe*c)] = L_a(xic_c[i])
        std::vector<double> Pc(npe * npe * nchild);
        for (int c = 0; c < nchild; ++c)
            for (int i = 0; i < npe; ++i) {
                auto L = lag2(xic[i + npe * (0 + nd * c)]);
                for (int a = 0; a < npe; ++a) Pc[i + npe * (a + npe * c)] = L[a];
            }
        std::vector<double> dg = {0.0, 0.4, 1.0};                 // curved parent
        std::vector<double> refined(npe * ncx * (ne * nchild), 0.0);
        refinemesh(refined.data(), dg.data(), Pc.data(), npe, ncx, ne, nchild);

        // analytic parent map x(xi) = L1(xi)*0.4 + L2(xi)*1
        auto xmap = [](double xi) { auto L = lag2(xi); return L[1] * 0.4 + L[2] * 1.0; };
        // child-major: child0 at 0..2, child1 at 3..5
        const std::vector<double> xi_all = {0, 0.25, 0.5,  0.5, 0.75, 1.0};
        for (std::size_t k = 0; k < xi_all.size(); ++k)
            expectClose(refined[k], xmap(xi_all[k]), "refine node " + std::to_string(k));
        // corners exact, interior shared, and NOT the chord midpoint
        expectClose(refined[0], 0.0, "corner0"); expectClose(refined[5], 1.0, "corner1");
        expectClose(refined[2], refined[3], "conformity (shared interior node)");
        if (nearlyEqual(refined[1], 0.2)) throw std::runtime_error("refine is linearizing curvature (chord, not curve)");
    }

    std::cout << "refinemesh tests passed\n";
    return 0;
}
