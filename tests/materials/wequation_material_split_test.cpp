#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <backend/Common/common.h>
#include <backend/Common/cpuimpl.h>
#include <backend/Common/kokkosimpl.h>
#include <backend/Common/pblas.h>
#include <backend/Discretization/residual.hpp>

namespace {

struct ShiftedWModel : exasim::ModelDefaults<ShiftedWModel> {
    static constexpr int nd = 1;
    static constexpr int ncu = 2;
    static constexpr int ncw = 2;
    static constexpr int nparam = 1;
    static constexpr auto disc = exasim::Discretization::HDG;

    KOKKOS_INLINE_FUNCTION static void sourcew(double s[], const double[],
            const double uq[], const double[], const double w[], const double[],
            const double[], double)
    {
        s[0] = 0.2*w[0] + 0.3*w[1] + 1.1*uq[0] - 0.2*uq[1] + 0.5;
        s[1] = -0.4*w[0] + 0.1*w[1] - 0.7*uq[0] + 0.9*uq[1] + 1.25;
    }

    KOKKOS_INLINE_FUNCTION static void sourcew_jac_uq(double s[], const double[],
            const double[], const double[], const double[], const double[],
            const double[], double)
    {
        for (int i = 0; i < ncw*ncu*(1 + nd); ++i)
            s[i] = 0.0;
        s[0] = 1.1;
        s[1] = -0.7;
        s[2] = -0.2;
        s[3] = 0.9;
    }

    KOKKOS_INLINE_FUNCTION static void sourcew_jac_w(double s[], const double[],
            const double[], const double[], const double[], const double[],
            const double[], double)
    {
        s[0] = 0.2;
        s[1] = -0.4;
        s[2] = 0.3;
        s[3] = 0.1;
    }
};

void requireClose(double a, double b, const char* message)
{
    if (std::abs(a - b) > 1.0e-14) {
        std::fprintf(stderr, "%s: got %.17g expected %.17g\n", message, a, b);
        throw std::runtime_error(message);
    }
}

void requireNear(double a, double b, double tolerance, const char* message)
{
    if (std::abs(a - b) > tolerance) {
        std::fprintf(stderr, "%s: got %.17g expected %.17g\n", message, a, b);
        throw std::runtime_error(message);
    }
}

void runNewtonMatrixTest()
{
    constexpr Int ng = 2;
    constexpr Int ncw = 3;
    constexpr dstype shift = 2.75;
    std::vector<dstype> jac(static_cast<std::size_t>(ng*ncw*ncw));
    std::vector<dstype> original(jac.size());
    for (std::size_t i = 0; i < jac.size(); ++i)
        jac[i] = original[i] = 0.125 + 0.01*static_cast<dstype>(i);

    FormWEquationNewtonMatrix<dstype,Int>(jac.data(), ng, ncw, shift, 0);
    Kokkos::fence();
    for (Int input = 0; input < ncw; ++input)
        for (Int output = 0; output < ncw; ++output)
            for (Int ig = 0; ig < ng; ++ig) {
                const std::size_t i = ig + static_cast<std::size_t>(ng) *
                    (output + ncw*input);
                const dstype expected = -original[i] +
                    ((output == input) ? shift : 0.0);
                requireClose(jac[i], expected,
                    "w-equation Newton matrix diagonal shift mismatch");
            }
}

void runNonzeroShiftSolveTest()
{
    constexpr Int ng = 3;
    constexpr Int ncw = 2;
    constexpr Int nc = ShiftedWModel::ncu*(1 + ShiftedWModel::nd);
    constexpr dstype alpha = 1.25;
    constexpr dstype dtfactor = 1.6;
    constexpr dstype beta = 0.5;
    constexpr dstype shift = alpha*dtfactor + beta;
    constexpr dstype j00 = shift - 0.2;
    constexpr dstype j01 = -0.3;
    constexpr dstype j10 = 0.4;
    constexpr dstype j11 = shift - 0.1;
    constexpr dstype determinant = j00*j11 - j01*j10;

    appstruct app;
    commonstruct common;
    common.components.ncu = ShiftedWModel::ncu;
    common.components.nc = nc;
    common.components.ncw = ncw;
    common.components.nco = 0;
    common.components.ncx = ShiftedWModel::nd;
    common.grid.nd = ShiftedWModel::nd;
    common.timeparams.wave = 0;
    common.timeparams.dae_alpha = alpha;
    common.timeparams.dae_beta = beta;
    common.timestate.dtfactor = dtfactor;
    common.timestate.time = 0.0;
    app.materialdb_nprop = 0;

    std::vector<dstype> xdg(ng, 0.0), udg(ng*nc), wsrc(ng*ncw);
    std::vector<dstype> odg(1, 0.0), wdg(ng*ncw), wdgUdg(ng*ncw*nc);
    std::vector<dstype> tempg(ng*(ncw + ncw*ncw), 0.0);
    for (Int ig = 0; ig < ng; ++ig) {
        udg[ig] = 0.3 + 0.2*ig;
        udg[ig + ng] = -0.4 + 0.1*ig;
        udg[ig + 2*ng] = 0.05 + 0.01*ig;
        udg[ig + 3*ng] = -0.08 + 0.02*ig;
        wsrc[ig] = 0.15 - 0.03*ig;
        wsrc[ig + ng] = -0.2 + 0.04*ig;
        wdg[ig] = 1.0 + ig;
        wdg[ig + ng] = -0.5 - ig;
    }

    std::vector<dstype> wdgValue = wdg;
    std::vector<dstype> tempgValue = tempg;
    wEquation<ShiftedWModel>(wdgValue.data(), xdg.data(), udg.data(),
        odg.data(), wsrc.data(), tempgValue.data(), app, common, ng, 0,
        nullptr);
    wEquation<ShiftedWModel>(wdg.data(), wdgUdg.data(), xdg.data(),
        udg.data(), odg.data(), wsrc.data(), tempg.data(), app, common,
        ng, 0, nullptr);
    Kokkos::fence();

    const dstype sourceU[8] = {1.1, -0.7, -0.2, 0.9, 0.0, 0.0, 0.0, 0.0};
    for (Int ig = 0; ig < ng; ++ig) {
        const dstype rhs0 = alpha*wsrc[ig] + 1.1*udg[ig] -
            0.2*udg[ig + ng] + 0.5;
        const dstype rhs1 = alpha*wsrc[ig + ng] - 0.7*udg[ig] +
            0.9*udg[ig + ng] + 1.25;
        const dstype expectedW0 = (j11*rhs0 - j01*rhs1) / determinant;
        const dstype expectedW1 = (-j10*rhs0 + j00*rhs1) / determinant;
        requireNear(wdgValue[ig], expectedW0, 2.0e-13,
            "nonzero-shift value-only w component 0 mismatch");
        requireNear(wdgValue[ig + ng], expectedW1, 2.0e-13,
            "nonzero-shift value-only w component 1 mismatch");
        requireNear(wdg[ig], expectedW0, 2.0e-13,
            "nonzero-shift w solution component 0 mismatch");
        requireNear(wdg[ig + ng], expectedW1, 2.0e-13,
            "nonzero-shift w solution component 1 mismatch");

        for (Int ju = 0; ju < nc; ++ju) {
            const dstype b0 = sourceU[2*ju];
            const dstype b1 = sourceU[2*ju + 1];
            const dstype expected0 = (j11*b0 - j01*b1) / determinant;
            const dstype expected1 = (-j10*b0 + j00*b1) / determinant;
            requireNear(wdgUdg[ig + ng*(0 + ncw*ju)], expected0, 2.0e-13,
                "nonzero-shift w sensitivity component 0 mismatch");
            requireNear(wdgUdg[ig + ng*(1 + ncw*ju)], expected1, 2.0e-13,
                "nonzero-shift w sensitivity component 1 mismatch");
        }
    }
}

void runTest()
{
    runNewtonMatrixTest();
    runNonzeroShiftSolveTest();

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
