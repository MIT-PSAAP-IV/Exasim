#include "l2eprojection.hpp"

#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
bool nearlyEqual(double a, double b, double tol = 1.0e-10) { return std::abs(a - b) <= tol * (1.0 + std::abs(b)); }
void expectClose(double a, double b, const std::string& l)
{
    if (!nearlyEqual(a, b)) throw std::runtime_error(l + ": expected " + std::to_string(b) + ", got " + std::to_string(a));
}
void expectThrows(const std::string& l, const std::function<void()>& fn)
{
    bool t = false; try { fn(); } catch (const std::runtime_error&) { t = true; }
    if (!t) throw std::runtime_error(l + ": expected runtime_error");
}

struct Gauss1D {
    std::array<double, 3> xi, w;
    Gauss1D() { double a = std::sqrt(0.6); xi = {0.5 * (1 - a), 0.5, 0.5 * (1 + a)}; w = {5.0/18, 4.0/9, 5.0/18}; }
};
std::array<double, 2> p1(double x)  { return {1 - x, x}; }
std::array<double, 2> dp1(double)   { return {-1.0, 1.0}; }
std::array<double, 3> p2(double x)  { return {2*x*x - 3*x + 1, -4*x*x + 4*x, 2*x*x - x}; }
std::array<double, 3> dp2(double x) { return {4*x - 3, -8*x + 4, 4*x - 1}; }

// L2-project a linear/quadratic f onto an affine 1D element and check the result
// reproduces f at the element nodes (exact when f is in the polynomial space).
template <int NP>
void checkExact(std::function<std::array<double, NP>(double)> basis,
                std::function<std::array<double, NP>(double)> dbasis,
                const std::array<double, NP>& xnodes,       // reference node coords on [0,1]
                double x0, double x1,                       // physical endpoints
                std::function<double(double)> f, const std::string& tag)
{
    Gauss1D g; const int nge = 3, npe = NP, nd = 1, ncx = 1, nc = 1, ne = 1;
    std::vector<double> shapv(nge * npe), dshapv(nge * npe), gw(g.w.begin(), g.w.end());
    for (int q = 0; q < nge; ++q) {
        auto ph = basis(g.xi[q]); auto dp = dbasis(g.xi[q]);
        for (int a = 0; a < npe; ++a) { shapv[q + nge * a] = ph[a]; dshapv[q + nge * a] = dp[a]; }
    }
    // physical node coords (affine map of the reference nodes)
    std::vector<double> dgn(npe);
    for (int a = 0; a < npe; ++a) dgn[a] = x0 + (x1 - x0) * xnodes[a];
    // f at the Gauss points: pg = shapv * dgnodes
    std::vector<double> fg(nge);
    for (int q = 0; q < nge; ++q) {
        double pg = 0; for (int a = 0; a < npe; ++a) pg += shapv[q + nge * a] * dgn[a];
        fg[q] = f(pg);
    }
    std::vector<double> UDG(npe, 0.0);
    l2eprojection(UDG.data(), fg.data(), dgn.data(), shapv.data(), dshapv.data(), gw.data(),
                  npe, nge, nd, ncx, nc, ne);
    for (int a = 0; a < npe; ++a) expectClose(UDG[a], f(dgn[a]), tag + " node " + std::to_string(a));
}
} // namespace

int main()
{
    // p1: project a linear f -> exact at the two nodes
    checkExact<2>(p1, dp1, {0.0, 1.0}, 2.0, 5.0, [](double x) { return 1.0 + 2.0 * x; }, "p1 linear");

    // p2: project a quadratic f -> exact at the three nodes
    checkExact<3>(p2, dp2, {0.0, 0.5, 1.0}, 2.0, 5.0,
                  [](double x) { return 1.0 + 2.0 * x + 0.5 * x * x; }, "p2 quadratic");

    // two components at once: F assembles each column independently
    {
        Gauss1D g; const int nge = 3, npe = 2, nd = 1, ncx = 1, nc = 2, ne = 1;
        std::vector<double> shapv(nge * npe), dshapv(nge * npe), gw(g.w.begin(), g.w.end());
        for (int q = 0; q < nge; ++q) { auto ph = p1(g.xi[q]); auto dp = dp1(g.xi[q]);
            for (int a = 0; a < npe; ++a) { shapv[q + nge * a] = ph[a]; dshapv[q + nge * a] = dp[a]; } }
        std::vector<double> dgn = {2.0, 5.0};
        auto f0 = [](double x) { return 1.0 + 2.0 * x; };
        auto f1 = [](double x) { return -3.0 + 0.5 * x; };
        std::vector<double> fg(nge * nc);
        for (int q = 0; q < nge; ++q) { double pg = shapv[q] * dgn[0] + shapv[q + nge] * dgn[1];
            fg[q + nge * 0] = f0(pg); fg[q + nge * 1] = f1(pg); }
        std::vector<double> UDG(npe * nc, 0.0);
        l2eprojection(UDG.data(), fg.data(), dgn.data(), shapv.data(), dshapv.data(), gw.data(), npe, nge, nd, ncx, nc, ne);
        expectClose(UDG[0 + npe * 0], f0(2.0), "2c f0 n0"); expectClose(UDG[1 + npe * 0], f0(5.0), "2c f0 n1");
        expectClose(UDG[0 + npe * 1], f1(2.0), "2c f1 n0"); expectClose(UDG[1 + npe * 1], f1(5.0), "2c f1 n1");
    }

    expectThrows("bad nd", []() { double a = 0; l2eprojection(&a, &a, &a, &a, &a, &a, 1, 1, 4, 1, 1, 1); });

    std::cout << "l2eprojection tests passed\n";
    return 0;
}
