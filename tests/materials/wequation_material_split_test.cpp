#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <backend/Discretization/residual.hpp>

namespace {

void requireClose(double a, double b, const char* message)
{
    if (std::abs(a - b) > 1.0e-14) {
        std::fprintf(stderr, "%s: got %.17g expected %.17g\n", message, a, b);
        throw std::runtime_error(message);
    }
}

void runTest()
{
    constexpr Int ng = 4;
    constexpr Int ncw = 5;
    constexpr Int ncwa = 2;
    constexpr Int nc = 3;
    std::vector<dstype> fullJac(static_cast<std::size_t>(ng*ncw*ncw));
    std::vector<dstype> compactJac(static_cast<std::size_t>(ng*ncwa*ncwa), -1.0);
    for (Int jw = 0; jw < ncw; ++jw) {
        for (Int iw = 0; iw < ncw; ++iw) {
            for (Int ig = 0; ig < ng; ++ig) {
                fullJac[ig + ng*(iw + ncw*jw)] = 1000.0*ig + 100.0*jw + iw;
            }
        }
    }
    CompactSourcewJacobian<dstype,Int>(compactJac.data(), fullJac.data(), ng,
        ncwa, ncw, 0);
    Kokkos::fence();
    for (Int jw = 0; jw < ncwa; ++jw) {
        for (Int iw = 0; iw < ncwa; ++iw) {
            for (Int ig = 0; ig < ng; ++ig) {
                requireClose(compactJac[ig + ng*(iw + ncwa*jw)],
                    fullJac[ig + ng*(iw + ncw*jw)],
                    "compact Sourcew w-Jacobian mismatch");
            }
        }
    }

    std::vector<dstype> fullUdg(static_cast<std::size_t>(ng*ncw*nc));
    std::vector<dstype> compactUdg(static_cast<std::size_t>(ng*ncwa*nc), -1.0);
    for (Int ju = 0; ju < nc; ++ju) {
        for (Int iw = 0; iw < ncw; ++iw) {
            for (Int ig = 0; ig < ng; ++ig) {
                fullUdg[ig + ng*(iw + ncw*ju)] = 2000.0*ig + 100.0*ju + iw;
            }
        }
    }
    CompactSourcewUdg<dstype,Int>(compactUdg.data(), fullUdg.data(), ng,
        ncwa, ncw, nc, 0);
    Kokkos::fence();
    for (Int ju = 0; ju < nc; ++ju) {
        for (Int iw = 0; iw < ncwa; ++iw) {
            for (Int ig = 0; ig < ng; ++ig) {
                requireClose(compactUdg[ig + ng*(iw + ncwa*ju)],
                    fullUdg[ig + ng*(iw + ncw*ju)],
                    "compact Sourcew udg-Jacobian mismatch");
            }
        }
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
