#include "extrudesol.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
bool nearlyEqual(double a, double b, double tol = 1.0e-12) { return std::abs(a - b) <= tol; }
void expectClose(double a, double b, const std::string& l)
{
    if (!nearlyEqual(a, b))
        throw std::runtime_error(l + ": expected " + std::to_string(b) + ", got " + std::to_string(a));
}
void expectThrows(const std::string& l, const std::function<void()>& fn)
{
    bool t = false;
    try { fn(); } catch (const std::runtime_error&) { t = true; }
    if (!t) throw std::runtime_error(l + ": expected runtime_error");
}
} // namespace

int main()
{
    const double PI = 3.14159265358979323846;

    // ---------------------------------------------------------------
    // extrudesol: np2d=2, nc=1, ne2d=1, porder=1 (np1d=2), nz=2.
    // Hand-derived from extrudesol.m (5D build -> permute[1 4 2 3 5] -> reshape):
    // each of the ne2d*nz=2 3D elements repeats the 2D field down the np1d layers.
    // ---------------------------------------------------------------
    {
        const int np2d = 2, nc = 1, ne2d = 1, porder = 1, nz = 2;
        const std::vector<double> U2 = {10.0, 20.0};          // [np2d x nc x ne2d]
        std::vector<double> U3(2 * 2 * 1 * 2, 0.0);           // [np2d*np1d, nc, ne2d*nz] = [4,1,2]
        extrudesol(U3.data(), U2.data(), np2d, nc, ne2d, porder, nz);
        const std::vector<double> exp = {10, 20, 10, 20,  10, 20, 10, 20};
        for (std::size_t i = 0; i < exp.size(); ++i) expectClose(U3[i], exp[i], "extrudesol idx " + std::to_string(i));
    }

    // Two components + two 2D elements: check a component/element does not leak.
    {
        const int np2d = 2, nc = 2, ne2d = 2, porder = 0, nz = 2;  // np1d=1
        // U2[np2d=2, nc=2, ne2d=2], flat a + 2*(b + 2*c)
        std::vector<double> U2(2 * 2 * 2);
        for (int c = 0; c < 2; ++c) for (int b = 0; b < 2; ++b) for (int a = 0; a < 2; ++a)
            U2[a + 2 * (b + 2 * c)] = 100 * c + 10 * b + a;   // distinct per (a,b,c)
        std::vector<double> U3(2 * 1 * 2 * 2 * 2, 0.0);       // [np2d*np1d=2, nc=2, ne2d*nz=4]
        extrudesol(U3.data(), U2.data(), np2d, nc, ne2d, porder, nz);
        // 3D elem e3 = c + ne2d*e ; value at (n3=a, b, e3) = U2(a,b,c=e3%ne2d)
        const int N3 = 2, NE3 = 4;
        for (int e3 = 0; e3 < NE3; ++e3) for (int b = 0; b < 2; ++b) for (int a = 0; a < 2; ++a) {
            double got = U3[a + N3 * (b + 2 * e3)];
            double want = U2[a + 2 * (b + 2 * (e3 % ne2d))];
            expectClose(got, want, "extrudesol2 e3=" + std::to_string(e3));
        }
    }

    // ---------------------------------------------------------------
    // extrudecoord: zz=[0,1,3], porder=1, plc1d=[0,1] -> z grows with layer & slab.
    // ---------------------------------------------------------------
    {
        const int np2d = 2, nc = 1, ne2d = 1, porder = 1, nz = 2;
        const std::vector<double> zz = {0.0, 1.0, 3.0};
        const std::vector<double> plc1d = {0.0, 1.0};
        std::vector<double> z(4 * 1 * 2, 0.0);
        extrudecoord(z.data(), zz.data(), plc1d.data(), np2d, nc, ne2d, porder, nz);
        const std::vector<double> exp = {0, 0, 1, 1,  1, 1, 3, 3};
        for (std::size_t i = 0; i < exp.size(); ++i) expectClose(z[i], exp[i], "extrudecoord idx " + std::to_string(i));
    }

    // ---------------------------------------------------------------
    // extrudevelocity: vr=1, tt=[0,pi], porder=1, plc1d=[0,1].
    // theta layers: 0 and pi -> vx=[1,1,-1,-1], vy=[0,0,0,0]; vx^2+vy^2=1.
    // ---------------------------------------------------------------
    {
        const int np2d = 2, nc = 1, ne2d = 1, porder = 1, nz = 1;
        const std::vector<double> vr = {1.0, 1.0};
        const std::vector<double> tt = {0.0, PI};
        const std::vector<double> plc1d = {0.0, 1.0};
        std::vector<double> vx(4), vy(4);
        extrudevelocity(vx.data(), vy.data(), vr.data(), tt.data(), plc1d.data(), np2d, nc, ne2d, porder, nz);
        expectClose(vx[0], 1.0,  "vx[0]");
        expectClose(vx[2], -1.0, "vx[2]");   // layer d=1 -> theta=pi
        for (int i = 0; i < 4; ++i) expectClose(vx[i] * vx[i] + vy[i] * vy[i], 1.0, "vx^2+vy^2 " + std::to_string(i));
    }

    expectThrows("bad nz", []() { double a; extrudesol(&a, &a, 1, 1, 1, 1, 0); });

    std::cout << "extrudesol tests passed\n";
    return 0;
}
