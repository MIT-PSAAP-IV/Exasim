#include "eulereval.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool nearlyEqual(double a, double b, double tol = 1.0e-12)
{
    return std::abs(a - b) <= tol;
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

} // namespace

int main()
{
    const double gamma = 1.4;

    {
        const int npe = 2;
        const int nc = 3;
        const int ne = 1;
        const std::vector<double> u = {
            2.0, 3.0,
            4.0, 9.0,
            10.0, 20.0
        };
        std::vector<double> sca(static_cast<std::size_t>(npe * ne));

        eulereval1d(sca.data(), u.data(), "p", gamma, npe, nc, ne);
        expectClose(sca[0], 2.4, "1D pressure point 0");
        expectClose(sca[1], 2.6, "1D pressure point 1");

        eulereval(sca.data(), u.data(), "M", gamma, npe, nc, ne, 1);
        expectClose(sca[0], (4.0 / 2.0) / std::sqrt(1.4 * 2.4 / 2.0), "1D Mach point 0");
        expectClose(sca[1], (9.0 / 3.0) / std::sqrt(1.4 * 2.6 / 3.0), "1D Mach point 1");

        expectThrows("1D invalid selector", [&]() {
            eulereval1d(sca.data(), u.data(), "v", gamma, npe, nc, ne);
        });
    }

    {
        const int npe = 2;
        const int nc = 5;
        const int ne = 1;
        const std::vector<double> u = {
            2.0, 4.0,
            4.0, 8.0,
            6.0, 4.0,
            20.0, 30.0,
            99.0, 77.0
        };
        std::vector<double> sca(static_cast<std::size_t>(npe * ne));

        eulereval2d(sca.data(), u.data(), "v", gamma, npe, nc, ne);
        expectClose(sca[0], 3.0, "2D y-velocity point 0");
        expectClose(sca[1], 1.0, "2D y-velocity point 1");

        eulereval2d(sca.data(), u.data(), "h", gamma, npe, nc, ne);
        expectClose(sca[0], 11.4, "2D enthalpy point 0");
        expectClose(sca[1], 9.5, "2D enthalpy point 1");
    }

    {
        const int npe = 1;
        const int nc = 5;
        const int ne = 2;
        const std::vector<double> u = {
            2.0, 4.0, 6.0, 2.0, 20.0,
            4.0, 8.0, 4.0, 12.0, 50.0
        };
        std::vector<double> sca(static_cast<std::size_t>(npe * ne));

        eulereval3d(sca.data(), u.data(), "w", gamma, npe, nc, ne);
        expectClose(sca[0], 1.0, "3D z-velocity elem 0");
        expectClose(sca[1], 3.0, "3D z-velocity elem 1");

        eulereval3d(sca.data(), u.data(), "c2", gamma, npe, nc, ne);
        expectClose(sca[0], 1.68, "3D c2 elem 0");
        expectClose(sca[1], 3.08, "3D c2 elem 1");

        expectThrows("3D invalid nd", [&]() {
            eulereval(sca.data(), u.data(), "r", gamma, npe, nc, ne, 4);
        });
    }

    std::cout << "eulereval tests passed\n";
    return 0;
}
