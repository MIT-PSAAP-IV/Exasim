#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

using dstype = double;
using Int = int;

#include <backend/Discretization/material_properties.hpp>

namespace {

void requireClose(double a, double b, const char* message)
{
    if (std::abs(a - b) > 5.0e-13) {
        std::fprintf(stderr, "%s: got %.17g expected %.17g\n", message, a, b);
        throw std::runtime_error(message);
    }
}

double p0(double x) { return 2.0 + 3.0*x + 0.5*x*x; }
double dp0(double x) { return 3.0 + x; }
double p1(double x) { return 1.0 - 2.0*x; }
double dp1(double) { return -2.0; }

void runTest()
{
    constexpr Int nstate = 1;
    constexpr Int nprop = 2;
    constexpr Int porder = 2;
    constexpr Int np = porder + 1;
    constexpr Int npe = np;
    constexpr Int ne = 3;

    const std::vector<dstype> axis{0.0, 0.02, 0.10, 0.25, 0.60, 0.77, 1.00};
    const std::vector<Int> elementCounts{ne};
    const std::vector<Int> xelemoffset{0, ne + 1};
    const std::vector<dstype> xelem{axis[0], axis[2], axis[4], axis[6]};

    std::vector<dstype> dgnodes(static_cast<std::size_t>(npe*nstate*ne));
    std::vector<dstype> udg(static_cast<std::size_t>(npe*nprop*ne));
    for (Int e = 0; e < ne; ++e) {
        for (Int a = 0; a < npe; ++a) {
            const dstype x = axis[static_cast<std::size_t>(e*porder + a)];
            dgnodes[a + npe*(0 + nstate*e)] = x;
            udg[a + npe*(0 + nprop*e)] = p0(x);
            udg[a + npe*(1 + nprop*e)] = p1(x);
        }
    }

    const std::vector<dstype> X{0.0, 0.05, 0.10, 0.30, 0.60, 0.90, 1.00};
    const Int ng = static_cast<Int>(X.size());
    std::vector<dstype> U(static_cast<std::size_t>(ng*nprop), -999.0);
    std::vector<dstype> dUdX(static_cast<std::size_t>(ng*nprop*nstate), -999.0);
    std::vector<dstype> tmd(static_cast<std::size_t>(ng*nstate*(2 + 3*np)), 0.0);
    std::vector<Int> tmi(static_cast<std::size_t>(ng*(nstate + 1)), 0);

    materialproperties_kokkos(U.data(), dUdX.data(), X.data(), dgnodes.data(),
        udg.data(), xelem.data(), elementCounts.data(), xelemoffset.data(),
        tmd.data(), tmi.data(), ng, ne, npe, porder, nstate, nprop);
    Kokkos::fence();

    for (Int ig = 0; ig < ng; ++ig) {
        const dstype x = X[static_cast<std::size_t>(ig)];
        requireClose(U[ig + ng*0], p0(x), "property 0 interpolation mismatch");
        requireClose(U[ig + ng*1], p1(x), "property 1 interpolation mismatch");
        requireClose(dUdX[ig + ng*(0 + nprop*0)], dp0(x), "property 0 derivative mismatch");
        requireClose(dUdX[ig + ng*(1 + nprop*0)], dp1(x), "property 1 derivative mismatch");
    }
}

} // namespace

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    int status = 0;
    try {
        runTest();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        status = 1;
    }
    Kokkos::finalize();
    return status;
}
