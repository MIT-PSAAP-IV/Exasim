#ifdef HAVE_MPI
#include <mpi.h>
MPI_Comm EXASIM_COMM_WORLD = MPI_COMM_NULL;
MPI_Comm EXASIM_COMM_LOCAL = MPI_COMM_NULL;
#endif

#include <fstream>
#include <iostream>

#include <Kokkos_Core.hpp>
#include "../../backend/Discretization/nonmatchinggeometry.hpp"
#include "../../backend/Common/cpuimpl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using exasim::nonmatching::EvaluateFaceShape;
using exasim::nonmatching::FaceToElementReference;
using exasim::nonmatching::InverseElementMap;
using exasim::nonmatching::InverseMapStatus;
using exasim::nonmatching::MapNonmatchingReferenceToPhysical;
using exasim::nonmatching::SimplexNodalBasis2D;

namespace {

int failures = 0;
dstype worstCurvedResidual = 0.0;
dstype worstBasisError = 0.0;
dstype worstTetRestrictionError = 0.0;
dstype worstOtherRestrictionError = 0.0;
dstype worstOffFaceError = 0.0;
dstype worstPolynomialError = 0.0;
dstype worstSimplexExtrapolationError = 0.0;
dstype worstSimplexDerivativeError = 0.0;
dstype worstSimplexLegacyDifference = 0.0;
dstype worstSimplexCondition = 0.0;
dstype worstSimplexInverseResidual = 0.0;

void fail(const std::string& test, dstype got, dstype limit)
{
    std::printf("FAIL %-42s got %.16e, limit %.16e\n",
                test.c_str(), static_cast<double>(got), static_cast<double>(limit));
    ++failures;
}

void expect_le(const std::string& test, dstype got, dstype limit)
{
    if (!(got <= limit)) fail(test, got, limit);
}

void expect_true(const std::string& test, bool value)
{
    if (!value) {
        std::printf("FAIL %-42s expected true\n", test.c_str());
        ++failures;
    }
}

dstype vector_error(const dstype* a, const dstype* b, Int n)
{
    dstype error = 0.0;
    for (Int i = 0; i < n; ++i) error = std::max(error, std::abs(a[i] - b[i]));
    return error;
}

struct ReferenceData {
    Int nd = 0;
    Int elemtype = 0;
    Int porder = 0;
    Int npe = 0;
    Int npf = 0;
    Int nfe = 0;
    std::vector<dstype> xpe;
    std::vector<dstype> xpf;
    std::vector<Int> perm;
};

std::string master_nodes_path()
{
    const std::string source = __FILE__;
    const std::string marker = "/tests/unit/nonmatching_geometry.cpp";
    const std::string::size_type position = source.rfind(marker);
    expect_true("test source path identifies repository", position != std::string::npos);
    return source.substr(0, position) + "/backend/Preprocessing/masternodes.bin";
}

ReferenceData make_reference(Int nd, Int elemtype, Int porder)
{
    ReferenceData data;
    data.nd = nd;
    data.elemtype = elemtype;
    data.porder = porder;
    std::vector<int> telem;
    std::vector<int> tface;
    std::vector<int> perm;
    masternodes(
        data.xpe, telem, data.xpf, tface, perm,
        static_cast<int>(porder), static_cast<int>(nd),
        static_cast<int>(elemtype), master_nodes_path());
    data.npe = static_cast<Int>(data.xpe.size()) / nd;
    const Int faceDim = nd - 1;
    data.npf = static_cast<Int>(data.xpf.size()) / faceDim;
    data.nfe = static_cast<Int>(perm.size()) / data.npf;
    data.perm.assign(perm.begin(), perm.end());
    expect_true("master element has faces", data.nfe > 0);
    return data;
}

void analytic_map(dstype* x, const dstype* zeta, Int nd, bool curved)
{
    if (nd == 2) {
        const dstype r = zeta[0], s = zeta[1];
        x[0] = 0.7 + 1.2 * r + 0.2 * s;
        x[1] = -0.4 + 0.1 * r + 0.9 * s;
        if (curved) {
            x[0] += 0.08 * r * s;
            x[1] += 0.06 * r * r - 0.04 * r * s;
        }
    }
    else {
        const dstype r = zeta[0], s = zeta[1], t = zeta[2];
        x[0] = 0.2 + 1.1 * r + 0.1 * s + 0.05 * t;
        x[1] = -0.3 + 0.15 * r + 0.95 * s + 0.08 * t;
        x[2] = 0.4 + 0.05 * r + 0.12 * s + 1.05 * t;
        if (curved) {
            x[0] += 0.04 * r * s + 0.03 * t * t;
            x[1] += 0.05 * s * t - 0.02 * r * r;
            x[2] += 0.03 * r * t + 0.02 * s * s;
        }
    }
}

std::vector<dstype> make_geometry(const ReferenceData& data, bool curved)
{
    std::vector<dstype> geometry(static_cast<size_t>(data.npe * data.nd));
    for (Int a = 0; a < data.npe; ++a) {
        dstype zeta[3] = {0.0, 0.0, 0.0};
        dstype x[3] = {0.0, 0.0, 0.0};
        for (Int d = 0; d < data.nd; ++d) zeta[d] = data.xpe[a + data.npe * d];
        analytic_map(x, zeta, data.nd, curved);
        for (Int d = 0; d < data.nd; ++d) geometry[a + data.npe * d] = x[d];
    }
    return geometry;
}

void map_element(dstype* x, const dstype* zeta,
                 const ReferenceData& data, const std::vector<dstype>& geometry)
{
    std::vector<dstype> shape(static_cast<size_t>(data.npe));
    std::vector<dstype> dshape(static_cast<size_t>(data.npe * data.nd));
    dstype jacobian[9] = {0.0};
    MapReferenceToPhysical(
        x, jacobian, shape.data(), dshape.data(), zeta,
        geometry.data(), data.xpe.data(), data.nd, data.npe,
        data.elemtype, data.porder);
}

void test_basis_consistency(const ReferenceData& data)
{
    const dstype xi[2] = {0.23, (data.elemtype == 0 ? 0.31 : 0.67)};
    std::vector<dstype> psi(static_cast<size_t>(data.npf));
    std::vector<dstype> phi(static_cast<size_t>(data.npe));
    const bool inside = data.nd == 2
        ? xi[0] >= 0.0 && xi[0] <= 1.0
        : (data.elemtype == 0
            ? xi[0] >= 0.0 && xi[1] >= 0.0 && xi[0] + xi[1] <= 1.0
            : xi[0] >= 0.0 && xi[0] <= 1.0 && xi[1] >= 0.0 && xi[1] <= 1.0);
    expect_true("xi is in reference face", inside);

    for (Int face = 0; face < data.nfe; ++face) {
        const std::string suffix =
            " nd=" + std::to_string(data.nd) +
            " type=" + std::to_string(data.elemtype) +
            " p=" + std::to_string(data.porder) +
            " face=" + std::to_string(face);
        dstype zetaBar[3] = {0.0, 0.0, 0.0};
        FaceToElementReference(
            zetaBar, psi.data(), xi, data.xpe.data(), data.xpf.data(),
            data.perm.data(), face, data.nd, data.npe, data.npf,
            data.elemtype, data.porder);
        EvaluateVolumeShapeAtReferencePoint(
            phi.data(), zetaBar, data.xpe.data(), data.nd, data.npe,
            data.elemtype, data.porder);

        dstype restrictionError = 0.0;
        dstype offFaceError = 0.0;
        for (Int node = 0; node < data.npe; ++node) {
            Int localFaceNode = -1;
            for (Int j = 0; j < data.npf; ++j)
                if (data.perm[j + data.npf * face] == node) localFaceNode = j;
            if (localFaceNode >= 0)
                restrictionError = std::max(
                    restrictionError, std::abs(phi[node] - psi[localFaceNode]));
            else
                offFaceError = std::max(offFaceError, std::abs(phi[node]));
        }
        worstBasisError = std::max(
            worstBasisError, std::max(restrictionError, offFaceError));
        worstOffFaceError = std::max(worstOffFaceError, offFaceError);
        // Tetrahedra are tracked separately: their triangular face basis used
        // to inherit an exact-equality collapsed-vertex test in
        // makemaster.hpp (Phase 11 master-basis amendment) and now meets the
        // same roundoff-level bound as the other families.
        const bool tetrahedron = data.nd == 3 && data.elemtype == 0;
        if (tetrahedron)
            worstTetRestrictionError =
                std::max(worstTetRestrictionError, restrictionError);
        else
            worstOtherRestrictionError =
                std::max(worstOtherRestrictionError, restrictionError);
        expect_le(tetrahedron ? "tet face-basis restriction" + suffix
                              : "real perm restriction" + suffix,
                  restrictionError, static_cast<dstype>(1.0e-13));
        expect_le("real perm off-face zero" + suffix,
                  offFaceError, 2.0e-11);
    }
}

dstype polynomial_value(const dstype* zeta, Int nd, Int degree)
{
    dstype value = 0.7;
    for (Int d = 0; d < nd; ++d)
        value += (d + 1.0) * std::pow(zeta[d], degree);
    if (degree >= 2 && nd >= 2) value += 0.35 * zeta[0] * zeta[1];
    if (degree >= 3 && nd == 3) value -= 0.2 * zeta[0] * zeta[1] * zeta[2];
    return value;
}

void test_polynomial_reproduction(const ReferenceData& data)
{
    const Int degree = data.porder;
    std::vector<dstype> nodal(static_cast<size_t>(data.npe));
    for (Int a = 0; a < data.npe; ++a) {
        dstype zeta[3] = {0.0, 0.0, 0.0};
        for (Int d = 0; d < data.nd; ++d) zeta[d] = data.xpe[a + data.npe * d];
        nodal[a] = polynomial_value(zeta, data.nd, degree);
    }

    const dstype points[2][3] = {{0.21, 0.18, 0.17}, {0.24, -0.015, 0.12}};
    for (Int q = 0; q < 2; ++q) {
        std::vector<dstype> phi(static_cast<size_t>(data.npe));
        EvaluateVolumeShapeAtReferencePoint(
            phi.data(), points[q], data.xpe.data(), data.nd, data.npe,
            data.elemtype, data.porder);
        dstype interpolated = 0.0;
        for (Int a = 0; a < data.npe; ++a) interpolated += phi[a] * nodal[a];
        const dstype error = std::abs(
            interpolated - polynomial_value(points[q], data.nd, degree));
        worstPolynomialError = std::max(worstPolynomialError, error);
        expect_le(q == 0 ? "polynomial reproduction inside" :
                           "polynomial reproduction outside",
                  error, 2.0e-11);
    }
}

void test_inverse_maps(Int nd, Int elemtype)
{
    for (Int porder = 1; porder <= 4; ++porder) {
        const ReferenceData data = make_reference(nd, elemtype, porder);
        const bool curved = porder >= 2;
        const std::vector<dstype> geometry = make_geometry(data, curved);
        dstype zetaExact[3] = {0.24, -0.012, 0.16};
        if (nd == 3 && elemtype == 0) {
            zetaExact[0] = 0.21;
            zetaExact[1] = 0.24;
            zetaExact[2] = -0.012;
        }
        dstype x[3] = {0.0, 0.0, 0.0};
        map_element(x, zetaExact, data, geometry);
        dstype initial[3] = {0.24, 0.0, 0.16};
        dstype zeta[3] = {0.0, 0.0, 0.0};
        dstype residual = 0.0;
        const InverseMapStatus status = InverseElementMap(
            zeta, &residual, x, geometry.data(), data.xpe.data(), initial,
            nd, data.npe, elemtype, porder, 30, 40, 1.0e-13, 0.1);
        expect_true("inverse converged p=" + std::to_string(porder),
                    status == InverseMapStatus::Converged);
        worstCurvedResidual = std::max(worstCurvedResidual, residual);
        expect_le("inverse residual p=" + std::to_string(porder), residual, 1.0e-12);
        expect_le("inverse coordinates p=" + std::to_string(porder),
                  vector_error(zeta, zetaExact, nd), 2.0e-11);
        test_basis_consistency(data);
        test_polynomial_reproduction(data);
    }

    const ReferenceData data = make_reference(nd, elemtype, 1);
    const std::vector<dstype> geometry = make_geometry(data, false);
    const dstype target[3] = {0.1, 0.2, 0.3};
    const dstype initial[3] = {0.25, 0.25, 0.25};
    dstype zeta[3] = {0.0, 0.0, 0.0};
    dstype residual = 0.0;
    std::vector<dstype> collapsed(static_cast<size_t>(data.npe * nd), 0.0);
    expect_true("singular element map reports failure", InverseElementMap(
        zeta, &residual, target, collapsed.data(), data.xpe.data(), initial,
        nd, data.npe, elemtype, 1, 4, 4, 1.0e-13, 0.1) ==
        InverseMapStatus::Failed);
}

void evaluate_total_degree_polynomial(
    dstype& value, dstype* gradient, const dstype* zeta, Int porder)
{
    value = 0.0;
    gradient[0] = 0.0;
    gradient[1] = 0.0;
    for (Int b = 0; b <= porder; ++b) {
        for (Int a = 0; a <= porder - b; ++a) {
            const dstype coefficient = 0.25 + 0.13 * a - 0.07 * b + 0.03 * a * b;
            const dstype ra = std::pow(zeta[0], a);
            const dstype sb = std::pow(zeta[1], b);
            value += coefficient * ra * sb;
            if (a > 0)
                gradient[0] += coefficient * a * std::pow(zeta[0], a - 1) * sb;
            if (b > 0)
                gradient[1] += coefficient * b * ra * std::pow(zeta[1], b - 1);
        }
    }
}

void test_nonmatching_simplex_basis()
{
    const dstype points[][2] = {
        {-0.10, 0.20}, {0.20, -0.10}, {0.60, 0.60}, {1.10, 0.00},
        {1.20, -0.10}, {0.00, 1.05}, {0.00, 1.20}, {0.10, 1.00},
        {-0.10, 1.10}, {0.22, 0.31}, {0.00, 1.00},
        {0.00, 1.0 - 9.0e-9}, {4.0e-9, 1.0 - 9.0e-9}};

    for (Int porder = 1; porder <= 4; ++porder) {
        const ReferenceData data = make_reference(2, 0, porder);
        const SimplexNodalBasis2D basis(data.xpe.data(), data.npe, porder);
        worstSimplexCondition = std::max(worstSimplexCondition, basis.condition_estimate());
        worstSimplexInverseResidual =
            std::max(worstSimplexInverseResidual, basis.inverse_residual());
        std::vector<dstype> nodalValue(static_cast<std::size_t>(data.npe));
        for (Int node = 0; node < data.npe; ++node) {
            const dstype zeta[2] = {data.xpe[node], data.xpe[node + data.npe]};
            dstype gradient[2] = {0.0, 0.0};
            evaluate_total_degree_polynomial(nodalValue[node], gradient, zeta, porder);
        }

        for (const auto& zeta : points) {
            std::vector<dstype> shape(static_cast<std::size_t>(data.npe));
            std::vector<dstype> derivative(static_cast<std::size_t>(2 * data.npe));
            std::vector<dstype> legacy(static_cast<std::size_t>(data.npe));
            basis.evaluate(shape.data(), derivative.data(), zeta);
            dstype interpolated = 0.0;
            dstype interpolatedGradient[2] = {0.0, 0.0};
            dstype interpolatedLinear = 0.0;
            for (Int node = 0; node < data.npe; ++node) {
                interpolated += shape[node] * nodalValue[node];
                interpolatedLinear += shape[node] *
                    (3.0 + 2.0 * data.xpe[node] - 5.0 * data.xpe[node + data.npe]);
                for (Int d = 0; d < 2; ++d)
                    interpolatedGradient[d] +=
                        derivative[node + data.npe * d] * nodalValue[node];
            }
            dstype exact = 0.0, exactGradient[2] = {0.0, 0.0};
            evaluate_total_degree_polynomial(exact, exactGradient, zeta, porder);
            worstSimplexExtrapolationError = std::max(
                worstSimplexExtrapolationError, std::abs(interpolated - exact));
            worstSimplexDerivativeError = std::max(
                worstSimplexDerivativeError,
                vector_error(interpolatedGradient, exactGradient, 2));
            expect_le("simplex polynomial extrapolation p=" + std::to_string(porder),
                      std::abs(interpolated - exact), 2.0e-11);
            expect_le("simplex linear extrapolation p=" + std::to_string(porder),
                      std::abs(interpolatedLinear -
                          (3.0 + 2.0 * zeta[0] - 5.0 * zeta[1])), 2.0e-11);
            expect_le("simplex derivative extrapolation p=" + std::to_string(porder),
                      vector_error(interpolatedGradient, exactGradient, 2), 8.0e-11);

            if (zeta[0] >= 0.0 && zeta[1] >= 0.0 && zeta[0] + zeta[1] <= 1.0) {
                EvaluateVolumeShapeAtReferencePoint(
                    legacy.data(), zeta, data.xpe.data(), 2, data.npe, 0, porder);
                worstSimplexLegacyDifference = std::max(
                    worstSimplexLegacyDifference,
                    vector_error(shape.data(), legacy.data(), data.npe));
            }
        }

        for (Int node = 0; node < data.npe; ++node) {
            const dstype zeta[2] = {data.xpe[node], data.xpe[node + data.npe]};
            std::vector<dstype> shape(static_cast<std::size_t>(data.npe));
            basis.evaluate(shape.data(), nullptr, zeta);
            for (Int a = 0; a < data.npe; ++a)
                expect_le("simplex nodal cardinality p=" + std::to_string(porder),
                          std::abs(shape[a] - (a == node ? 1.0 : 0.0)), 2.0e-11);
        }
    }
    // The legacy collapsed-coordinate evaluator differs near (0,1) by about
    // 1e-9; away from that implementation artifact both bases represent the
    // same nodal polynomial space.
    expect_le("simplex in-element agreement with legacy basis",
              worstSimplexLegacyDifference, 1.0e-8);
}

void test_triangle_inverse_beyond_vertex(bool curved)
{
    const Int porder = 2;
    const ReferenceData data = make_reference(2, 0, porder);
    const SimplexNodalBasis2D basis(data.xpe.data(), data.npe, porder);
    const std::vector<dstype> geometry = make_geometry(data, curved);
    const dstype exact[2] = {0.0, 1.09};
    std::vector<dstype> shape(static_cast<std::size_t>(data.npe));
    std::vector<dstype> derivative(static_cast<std::size_t>(2 * data.npe));
    dstype target[2] = {0.0, 0.0}, jacobian[4] = {0.0};
    MapNonmatchingReferenceToPhysical(
        target, jacobian, shape.data(), derivative.data(), exact,
        geometry.data(), data.xpe.data(), 2, data.npe, 0, porder, &basis);
    const dstype initial[2] = {0.0, 1.0};
    dstype recovered[2] = {0.0, 0.0}, residual = 0.0;
    const dstype scale = std::max(std::hypot(target[0], target[1]), 1.0);
    const dstype tolerance = 64.0 * std::numeric_limits<dstype>::epsilon() * scale;
    const InverseMapStatus status = InverseElementMap(
        recovered, &residual, target, geometry.data(), data.xpe.data(), initial,
        2, data.npe, 0, porder, 30, 40, tolerance, 2.0, &basis);
    expect_true(std::string(curved ? "curved" : "affine") +
                    " triangle beyond-vertex inverse is exact",
                status == InverseMapStatus::Converged);
    expect_le(std::string(curved ? "curved" : "affine") +
                  " triangle beyond-vertex coordinate",
              vector_error(recovered, exact, 2), 2.0e-12);
    expect_le(std::string(curved ? "curved" : "affine") +
                  " triangle beyond-vertex residual",
              residual, tolerance);

    // With unrestricted iterations disabled, an exact projected solution keeps
    // the approved Phase 1 status; a neighborhood-truncated least-squares
    // result has the new explicit status.
    const InverseMapStatus projectedExact = InverseElementMap(
        recovered, &residual, target, geometry.data(), data.xpe.data(), initial,
        2, data.npe, 0, porder, 0, 40, tolerance, 2.0, &basis);
    expect_true("projected exact status remains ConvergedProjected",
                projectedExact == InverseMapStatus::ConvergedProjected);
    const dstype farExact[2] = {0.0, 2.0};
    MapNonmatchingReferenceToPhysical(
        target, jacobian, shape.data(), derivative.data(), farExact,
        geometry.data(), data.xpe.data(), 2, data.npe, 0, porder, &basis);
    const InverseMapStatus projectedInexact = InverseElementMap(
        recovered, &residual, target, geometry.data(), data.xpe.data(), initial,
        2, data.npe, 0, porder, 0, 40, tolerance, 0.05, &basis);
    expect_true("projected inexact has distinct status",
                projectedInexact == InverseMapStatus::ProjectedInexact);
    expect_true("projected inexact retains finite residual",
                std::isfinite(residual) && residual > tolerance);
}

void test_unrestricted_failure_uses_projected_safeguard()
{
    const ReferenceData data = make_reference(2, 0, 2);
    const SimplexNodalBasis2D basis(data.xpe.data(), data.npe, 2);
    std::vector<dstype> geometry(static_cast<std::size_t>(2 * data.npe));
    for (Int node = 0; node < data.npe; ++node) {
        const dstype r = data.xpe[node];
        const dstype s = data.xpe[node + data.npe];
        geometry[node] = (r - 10.0) * (r - 10.0);
        geometry[node + data.npe] = s;
    }
    // From r=0, unrestricted Newton for x=-100 steps to r=10, where the
    // Jacobian is singular. The projected restart remains finite in the
    // reference neighborhood and must return its least-squares point.
    const dstype target[2] = {-100.0, 0.0};
    const dstype initial[2] = {0.0, 0.0};
    dstype zeta[2] = {0.0, 0.0}, residual = 0.0;
    const InverseMapStatus status = InverseElementMap(
        zeta, &residual, target, geometry.data(), data.xpe.data(), initial,
        2, data.npe, 0, 2, 8, 40, 1.0e-12, 0.1, &basis);
    expect_true("unrestricted singular step falls through to safeguard",
                status == InverseMapStatus::ProjectedInexact);
    expect_true("safeguarded result after unrestricted failure is finite",
                std::isfinite(residual) && std::isfinite(zeta[0]) &&
                std::isfinite(zeta[1]));
}

} // namespace

int main()
{
    test_inverse_maps(2, 0); // triangle
    test_inverse_maps(2, 1); // quadrilateral
    test_inverse_maps(3, 0); // tetrahedron
    test_inverse_maps(3, 1); // hexahedron
    test_nonmatching_simplex_basis();
    test_triangle_inverse_beyond_vertex(false);
    test_triangle_inverse_beyond_vertex(true);
    test_unrestricted_failure_uses_projected_safeguard();

    std::printf("worst curved residual    %.16e\n", static_cast<double>(worstCurvedResidual));
    std::printf("worst basis error        %.16e\n", static_cast<double>(worstBasisError));
    std::printf("worst tet restriction    %.16e\n", static_cast<double>(worstTetRestrictionError));
    std::printf("worst other restriction  %.16e\n", static_cast<double>(worstOtherRestrictionError));
    std::printf("worst off-face basis     %.16e\n", static_cast<double>(worstOffFaceError));
    std::printf("worst polynomial error   %.16e\n", static_cast<double>(worstPolynomialError));
    std::printf("worst simplex extrap.   %.16e\n", static_cast<double>(worstSimplexExtrapolationError));
    std::printf("worst simplex derivative %.16e\n", static_cast<double>(worstSimplexDerivativeError));
    std::printf("simplex/legacy max diff %.16e\n", static_cast<double>(worstSimplexLegacyDifference));
    std::printf("simplex max condition   %.16e\n", static_cast<double>(worstSimplexCondition));
    std::printf("simplex inverse residual %.16e\n", static_cast<double>(worstSimplexInverseResidual));
    if (failures == 0) std::printf("PASS nonmatching_geometry\n");
    else std::printf("FAIL nonmatching_geometry: %d check(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
