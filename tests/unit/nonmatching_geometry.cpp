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

using exasim::nonmatching::ClosestPointOnFace;
using exasim::nonmatching::ComputeGeometryPointData;
using exasim::nonmatching::EvaluateFaceShape;
using exasim::nonmatching::FaceToElementReference;
using exasim::nonmatching::GeometryPointStatus;
using exasim::nonmatching::InverseElementMap;
using exasim::nonmatching::InverseMapStatus;
using exasim::nonmatching::MapNonmatchingReferenceToPhysical;
using exasim::nonmatching::MapReferenceFaceToPhysical;
using exasim::nonmatching::PointInReferenceFace;
using exasim::nonmatching::SimplexNodalBasis2D;

namespace {

int failures = 0;
dstype worstAffineError = 0.0;
dstype worstCurvedProjectionError = 0.0;
dstype worstCurvedResidual = 0.0;
dstype worstBasisError = 0.0;
dstype worstTetRestrictionError = 0.0;
dstype worstOtherRestrictionError = 0.0;
dstype worstOffFaceError = 0.0;
dstype worstPolynomialError = 0.0;
dstype worstScaledRelativeError = 0.0;
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

std::vector<dstype> make_face_geometry(
    const ReferenceData& data, const std::vector<dstype>& geometry, Int faceIndex)
{
    std::vector<dstype> face(static_cast<size_t>(data.npf * data.nd));
    for (Int d = 0; d < data.nd; ++d)
        for (Int j = 0; j < data.npf; ++j)
            face[j + data.npf * d] =
                geometry[data.perm[j + data.npf * faceIndex] + data.npe * d];
    return face;
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
    expect_true("xi is in reference face", PointInReferenceFace(xi, data.nd, data.elemtype));

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
        // PointLocator's triangular face basis currently inherits the
        // collapsed-coordinate clamp in makemaster.hpp:289. Keep that known
        // tetrahedral trace limitation isolated from the roundoff-level
        // checks for triangles, quadrilaterals, and hexahedra.
        const bool tetrahedron = data.nd == 3 && data.elemtype == 0;
        if (tetrahedron)
            worstTetRestrictionError =
                std::max(worstTetRestrictionError, restrictionError);
        else
            worstOtherRestrictionError =
                std::max(worstOtherRestrictionError, restrictionError);
        const dstype restrictionTolerance =
            tetrahedron ? static_cast<dstype>(5.0e-9)
                        : static_cast<dstype>(1.0e-13);
        expect_le(tetrahedron ? "tet known face-basis restriction" + suffix
                              : "real perm restriction" + suffix,
                  restrictionError, restrictionTolerance);
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

void test_affine_projection_and_inverse(Int nd, Int elemtype)
{
    const ReferenceData data = make_reference(nd, elemtype, 1);
    const std::vector<dstype> geometry = make_geometry(data, false);
    const std::vector<dstype> face = make_face_geometry(data, geometry, 0);
    const dstype xiExact[2] = {0.27, (elemtype == 0 ? 0.29 : 0.61)};
    std::vector<dstype> psi(static_cast<size_t>(data.npf));
    std::vector<dstype> dpsi(static_cast<size_t>(data.npf * (nd - 1)));
    dstype xbarExact[3] = {0.0, 0.0, 0.0};
    dstype tangent[6] = {0.0};
    MapReferenceFaceToPhysical(
        xbarExact, tangent, psi.data(), dpsi.data(), xiExact,
        face.data(), data.xpf.data(), nd, data.npf, elemtype, 1);

    dstype normal[3] = {0.0, 0.0, 0.0};
    if (nd == 2) {
        normal[0] = -tangent[1];
        normal[1] = tangent[0];
    }
    else {
        normal[0] = tangent[2] * tangent[5] - tangent[4] * tangent[3];
        normal[1] = tangent[4] * tangent[1] - tangent[0] * tangent[5];
        normal[2] = tangent[0] * tangent[3] - tangent[2] * tangent[1];
    }
    dstype normalNorm = std::sqrt(exasim::nonmatching::SquaredNorm(normal, nd));
    for (Int d = 0; d < nd; ++d) normal[d] /= normalNorm;

    for (Int side = -1; side <= 1; ++side) {
        dstype x[3] = {0.0, 0.0, 0.0};
        for (Int d = 0; d < nd; ++d) x[d] = xbarExact[d] + 0.08 * side * normal[d];
        dstype xi[2] = {0.0, 0.0};
        dstype xbar[3] = {0.0, 0.0, 0.0};
        dstype distance = 0.0;
        const bool converged = ClosestPointOnFace(
            xi, xbar, &distance, x, face.data(), data.xpf.data(), nd,
            data.npf, elemtype, 1, 20, 1.0e-14);
        expect_true("affine closest-point converged", converged);
        const dstype xiError = vector_error(xi, xiExact, nd - 1);
        const dstype distanceError = std::abs(distance - 0.08 * std::abs(side));
        worstAffineError = std::max(worstAffineError, std::max(xiError, distanceError));
        expect_le("affine closest-point xi", xiError, 1.0e-13);
        expect_le("affine closest-point distance", distanceError, 1.0e-13);
    }

    dstype zetaExact[3] = {0.22, -0.025, 0.16};
    if (nd == 2) zetaExact[1] = -0.025;
    dstype x[3] = {0.0, 0.0, 0.0};
    map_element(x, zetaExact, data, geometry);
    dstype initial[3] = {0.25, 0.0, 0.2};
    dstype zeta[3] = {0.0, 0.0, 0.0};
    dstype residual = 0.0;
    const InverseMapStatus status = InverseElementMap(
        zeta, &residual, x, geometry.data(), data.xpe.data(), initial,
        nd, data.npe, elemtype, 1, 8, 20, 1.0e-14, 0.1);
    expect_true("affine inverse unrestricted status", status == InverseMapStatus::Converged);
    const dstype inverseError = vector_error(zeta, zetaExact, nd);
    worstAffineError = std::max(worstAffineError, std::max(inverseError, residual));
    expect_le("affine inverse coordinates", inverseError, 1.0e-13);
    expect_le("affine inverse residual", residual, 1.0e-13);
    expect_true("inverse extrapolation may leave element",
                !PointInReferenceElement(zeta, nd, elemtype, 1.0e-14));

    // Force the safeguarded path while keeping the exact point in its neighborhood.
    const InverseMapStatus projectedStatus = InverseElementMap(
        zeta, &residual, x, geometry.data(), data.xpe.data(), initial,
        nd, data.npe, elemtype, 1, 0, 20, 1.0e-13, 0.1);
    expect_true("projected inverse fallback status",
                projectedStatus == InverseMapStatus::ConvergedProjected);
    expect_le("projected inverse residual", residual, 1.0e-12);

    // A point beyond the face edge must project onto the constrained boundary.
    dstype xiOutside[2] = {-0.18, 0.42};
    MapReferenceFaceToPhysical(
        x, tangent, psi.data(), dpsi.data(), xiOutside,
        face.data(), data.xpf.data(), nd, data.npf, elemtype, 1);
    dstype xi[2] = {0.0, 0.0};
    dstype xbar[3] = {0.0, 0.0, 0.0};
    dstype distance = 0.0;
    expect_true("beyond-edge projection converged", ClosestPointOnFace(
        xi, xbar, &distance, x, face.data(), data.xpf.data(), nd,
        data.npf, elemtype, 1, 20, 1.0e-14));
    expect_true("beyond-edge xi constrained", PointInReferenceFace(xi, nd, elemtype));
    expect_true("beyond-edge xi on boundary",
                nd == 2 ? (xi[0] == 0.0 || xi[0] == 1.0) :
                (std::abs(xi[0]) < 1.0e-13 || std::abs(xi[1]) < 1.0e-13 ||
                 (elemtype == 0 && std::abs(xi[0] + xi[1] - 1.0) < 1.0e-13) ||
                 (elemtype == 1 && (std::abs(xi[0] - 1.0) < 1.0e-13 ||
                                     std::abs(xi[1] - 1.0) < 1.0e-13))));

    std::vector<dstype> collapsedFace(static_cast<size_t>(data.npf * nd), 0.0);
    expect_true("singular face map reports failure", !ClosestPointOnFace(
        xi, xbar, &distance, x, collapsedFace.data(), data.xpf.data(), nd,
        data.npf, elemtype, 1, 4, 1.0e-14));
    std::vector<dstype> collapsedElement(static_cast<size_t>(data.npe * nd), 0.0);
    expect_true("singular element map reports failure", InverseElementMap(
        zeta, &residual, x, collapsedElement.data(), data.xpe.data(), initial,
        nd, data.npe, elemtype, 1, 4, 4, 1.0e-13, 0.1) ==
        InverseMapStatus::Failed);

    test_basis_consistency(data);
    test_polynomial_reproduction(data);
}

void test_curved_orders(Int nd, Int elemtype)
{
    for (Int porder = 1; porder <= 4; ++porder) {
        const ReferenceData data = make_reference(nd, elemtype, porder);
        const bool curved = porder >= 2;
        const std::vector<dstype> geometry = make_geometry(data, curved);
        const std::vector<dstype> face = make_face_geometry(data, geometry, 0);

        const dstype xiProjection[2] = {0.28, (elemtype == 0 ? 0.24 : 0.63)};
        std::vector<dstype> faceShape(static_cast<size_t>(data.npf));
        std::vector<dstype> faceDerivative(static_cast<size_t>(data.npf * (nd - 1)));
        dstype facePoint[3] = {0.0, 0.0, 0.0};
        dstype faceTangent[6] = {0.0};
        MapReferenceFaceToPhysical(
            facePoint, faceTangent, faceShape.data(), faceDerivative.data(),
            xiProjection, face.data(), data.xpf.data(), nd, data.npf,
            elemtype, porder);
        dstype faceNormal[3] = {0.0, 0.0, 0.0};
        if (nd == 2) {
            faceNormal[0] = -faceTangent[1];
            faceNormal[1] = faceTangent[0];
        }
        else {
            faceNormal[0] = faceTangent[2] * faceTangent[5] - faceTangent[4] * faceTangent[3];
            faceNormal[1] = faceTangent[4] * faceTangent[1] - faceTangent[0] * faceTangent[5];
            faceNormal[2] = faceTangent[0] * faceTangent[3] - faceTangent[2] * faceTangent[1];
        }
        const dstype faceNormalNorm =
            std::sqrt(exasim::nonmatching::SquaredNorm(faceNormal, nd));
        dstype projectionTarget[3] = {0.0, 0.0, 0.0};
        for (Int d = 0; d < nd; ++d)
            projectionTarget[d] = facePoint[d] + 0.01 * faceNormal[d] / faceNormalNorm;
        dstype xiComputed[2] = {0.0, 0.0};
        dstype xbarComputed[3] = {0.0, 0.0, 0.0};
        dstype projectionDistance = 0.0;
        expect_true("curved projection converged p=" + std::to_string(porder),
                    ClosestPointOnFace(
                        xiComputed, xbarComputed, &projectionDistance,
                        projectionTarget, face.data(), data.xpf.data(), nd,
                        data.npf, elemtype, porder, 40, 1.0e-13));
        worstCurvedProjectionError = std::max(
            worstCurvedProjectionError,
            std::max(vector_error(xiComputed, xiProjection, nd - 1),
                     std::abs(projectionDistance - 0.01)));
        expect_le("curved projection xi p=" + std::to_string(porder),
                  vector_error(xiComputed, xiProjection, nd - 1), 2.0e-10);
        expect_le("curved projection distance p=" + std::to_string(porder),
                  std::abs(projectionDistance - 0.01), 2.0e-11);

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
        expect_true("curved inverse converged p=" + std::to_string(porder),
                    status == InverseMapStatus::Converged);
        worstCurvedResidual = std::max(worstCurvedResidual, residual);
        expect_le("curved inverse residual p=" + std::to_string(porder), residual, 1.0e-12);
        expect_le("curved inverse coordinates p=" + std::to_string(porder),
                  vector_error(zeta, zetaExact, nd), 2.0e-11);

        test_basis_consistency(data);
        test_polynomial_reproduction(data);

        if (porder == 3) {
            // Exercise every output of the combined helper, eqs. (21),
            // (23), (28), and (30), on the face and from both sides.
            const dstype xiExact[2] = {0.28, (elemtype == 0 ? 0.24 : 0.63)};
            std::vector<dstype> psi(static_cast<size_t>(data.npf));
            std::vector<dstype> dpsi(static_cast<size_t>(data.npf * (nd - 1)));
            dstype facePoint[3] = {0.0, 0.0, 0.0};
            dstype tangent[6] = {0.0};
            MapReferenceFaceToPhysical(
                facePoint, tangent, psi.data(), dpsi.data(), xiExact,
                face.data(), data.xpf.data(), nd, data.npf, elemtype, porder);
            dstype normal[3] = {0.0, 0.0, 0.0};
            if (nd == 2) {
                normal[0] = -tangent[1]; normal[1] = tangent[0];
            }
            else {
                normal[0] = tangent[2] * tangent[5] - tangent[4] * tangent[3];
                normal[1] = tangent[4] * tangent[1] - tangent[0] * tangent[5];
                normal[2] = tangent[0] * tangent[3] - tangent[2] * tangent[1];
            }
            const dstype norm = std::sqrt(exasim::nonmatching::SquaredNorm(normal, nd));

            for (Int side = -1; side <= 1; ++side) {
                const dstype signedDistance = 0.01 * side;
                for (Int d = 0; d < nd; ++d)
                    x[d] = facePoint[d] + signedDistance * normal[d] / norm;

                std::vector<dstype> phi(static_cast<size_t>(data.npe));
                std::vector<dstype> phiBar(static_cast<size_t>(data.npe));
                std::vector<dstype> expectedPhi(static_cast<size_t>(data.npe));
                std::vector<dstype> expectedPhiBar(static_cast<size_t>(data.npe));
                std::vector<dstype> expectedPsi(static_cast<size_t>(data.npf));
                dstype xi[2] = {0.0, 0.0};
                dstype zetaComputed[3] = {0.0, 0.0, 0.0};
                dstype zetaBar[3] = {0.0, 0.0, 0.0};
                dstype xbar[3] = {0.0, 0.0, 0.0};
                dstype offset[3] = {0.0, 0.0, 0.0};
                const GeometryPointStatus geometryStatus = ComputeGeometryPointData(
                    xi, zetaComputed, zetaBar, xbar, phi.data(), phiBar.data(),
                    psi.data(), offset, x, geometry.data(), data.xpf.data(),
                    data.xpe.data(), data.perm.data(), 0, nd, data.npe,
                    data.npf, elemtype, porder, 40, 40, 40,
                    1.0e-13, 1.0e-12, 0.1);
                const std::string suffix = " side=" + std::to_string(side);
                expect_true("combined projection" + suffix,
                            geometryStatus.projectionConverged);
                expect_true("combined inverse" + suffix,
                            geometryStatus.inverseStatus == InverseMapStatus::Converged);
                expect_le("combined xi" + suffix,
                          vector_error(xi, xiExact, nd - 1), 2.0e-10);
                expect_le("combined xbar" + suffix,
                          vector_error(xbar, facePoint, nd), 2.0e-11);
                expect_le("combined distance" + suffix,
                          std::abs(geometryStatus.distance - std::abs(signedDistance)),
                          2.0e-11);

                dstype expectedOffset[3] = {0.0, 0.0, 0.0};
                for (Int d = 0; d < nd; ++d)
                    expectedOffset[d] = signedDistance * normal[d] / norm;
                expect_le("combined offset" + suffix,
                          vector_error(offset, expectedOffset, nd), 2.0e-11);

                dstype mappedPoint[3] = {0.0, 0.0, 0.0};
                map_element(mappedPoint, zetaComputed, data, geometry);
                expect_le("combined inverse map" + suffix,
                          vector_error(mappedPoint, x, nd), 2.0e-11);

                EvaluateFaceShape(
                    expectedPsi.data(), xi, data.xpf.data(), nd, data.npf,
                    elemtype, porder);
                EvaluateVolumeShapeAtReferencePoint(
                    expectedPhi.data(), zetaComputed, data.xpe.data(), nd,
                    data.npe, elemtype, porder);
                EvaluateVolumeShapeAtReferencePoint(
                    expectedPhiBar.data(), zetaBar, data.xpe.data(), nd,
                    data.npe, elemtype, porder);
                expect_le("combined psi" + suffix,
                          vector_error(psi.data(), expectedPsi.data(), data.npf),
                          2.0e-13);
                expect_le("combined phi" + suffix,
                          vector_error(phi.data(), expectedPhi.data(), data.npe),
                          2.0e-13);
                expect_le("combined phibar" + suffix,
                          vector_error(phiBar.data(), expectedPhiBar.data(), data.npe),
                          2.0e-13);

                if (side == 0) {
                    expect_le("on-face zeta equals zetabar",
                              vector_error(zetaComputed, zetaBar, nd), 2.0e-11);
                    expect_le("on-face phi equals phibar",
                              vector_error(phi.data(), phiBar.data(), data.npe),
                              1.0e-10);
                }
            }
        }
    }
}

void test_scale_invariance(bool curved)
{
    const Int porder = curved ? 3 : 1;
    const ReferenceData data = make_reference(3, 1, porder);
    const std::vector<dstype> baseGeometry = make_geometry(data, curved);
    const dstype scales[2] = {1.0e-6, 1.0e3};

    for (dstype scale : scales) {
        std::vector<dstype> geometry = baseGeometry;
        for (dstype& value : geometry) value *= scale;
        const std::vector<dstype> face = make_face_geometry(data, geometry, 0);

        const dstype xiExact[2] = {0.31, 0.57};
        std::vector<dstype> psi(static_cast<size_t>(data.npf));
        std::vector<dstype> dpsi(static_cast<size_t>(2 * data.npf));
        dstype facePoint[3] = {0.0, 0.0, 0.0};
        dstype tangent[6] = {0.0};
        MapReferenceFaceToPhysical(
            facePoint, tangent, psi.data(), dpsi.data(), xiExact,
            face.data(), data.xpf.data(), 3, data.npf, 1, porder);
        dstype normal[3] = {
            tangent[2] * tangent[5] - tangent[4] * tangent[3],
            tangent[4] * tangent[1] - tangent[0] * tangent[5],
            tangent[0] * tangent[3] - tangent[2] * tangent[1]};
        const dstype normalNorm =
            std::sqrt(exasim::nonmatching::SquaredNorm(normal, 3));
        dstype target[3] = {0.0, 0.0, 0.0};
        for (Int d = 0; d < 3; ++d)
            target[d] = facePoint[d] + 0.01 * scale * normal[d] / normalNorm;

        dstype xi[2] = {0.0, 0.0};
        dstype xbar[3] = {0.0, 0.0, 0.0};
        dstype distance = 0.0;
        const std::string suffix =
            std::string(curved ? " curved" : " affine") +
            " scale=" + std::to_string(scale);
        expect_true("scaled projection" + suffix, ClosestPointOnFace(
            xi, xbar, &distance, target, face.data(), data.xpf.data(),
            3, data.npf, 1, porder, 40, 1.0e-13));
        const dstype projectionRelativeError = std::max(
            vector_error(xi, xiExact, 2),
            std::abs(distance - 0.01 * scale) / scale);
        worstScaledRelativeError =
            std::max(worstScaledRelativeError, projectionRelativeError);
        expect_le("scaled projection error" + suffix,
                  projectionRelativeError, 2.0e-10);

        const dstype zetaExact[3] = {0.24, 0.36, -0.012};
        dstype physicalPoint[3] = {0.0, 0.0, 0.0};
        map_element(physicalPoint, zetaExact, data, geometry);
        const dstype initial[3] = {0.24, 0.36, 0.0};
        dstype zeta[3] = {0.0, 0.0, 0.0};
        dstype residual = 0.0;
        const InverseMapStatus status = InverseElementMap(
            zeta, &residual, physicalPoint, geometry.data(), data.xpe.data(),
            initial, 3, data.npe, 1, porder, 40, 40,
            scale * 1.0e-13, 0.1);
        expect_true("scaled inverse" + suffix,
                    status == InverseMapStatus::Converged);
        const dstype inverseRelativeError = std::max(
            vector_error(zeta, zetaExact, 3), residual / scale);
        worstScaledRelativeError =
            std::max(worstScaledRelativeError, inverseRelativeError);
        expect_le("scaled inverse error" + suffix,
                  inverseRelativeError, 2.0e-11);
    }
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
    test_affine_projection_and_inverse(2, 0); // triangle
    test_affine_projection_and_inverse(2, 1); // quadrilateral
    test_affine_projection_and_inverse(3, 0); // tetrahedron
    test_affine_projection_and_inverse(3, 1); // hexahedron

    test_curved_orders(2, 0);
    test_curved_orders(2, 1);
    test_curved_orders(3, 0);
    test_curved_orders(3, 1);
    test_scale_invariance(false);
    test_scale_invariance(true);
    test_nonmatching_simplex_basis();
    test_triangle_inverse_beyond_vertex(false);
    test_triangle_inverse_beyond_vertex(true);
    test_unrestricted_failure_uses_projected_safeguard();

    std::printf("worst affine error       %.16e\n", static_cast<double>(worstAffineError));
    std::printf("worst curved projection  %.16e\n", static_cast<double>(worstCurvedProjectionError));
    std::printf("worst curved residual    %.16e\n", static_cast<double>(worstCurvedResidual));
    std::printf("worst basis error        %.16e\n", static_cast<double>(worstBasisError));
    std::printf("worst tet restriction    %.16e\n", static_cast<double>(worstTetRestrictionError));
    std::printf("worst other restriction  %.16e\n", static_cast<double>(worstOtherRestrictionError));
    std::printf("worst off-face basis     %.16e\n", static_cast<double>(worstOffFaceError));
    std::printf("worst polynomial error   %.16e\n", static_cast<double>(worstPolynomialError));
    std::printf("worst scaled rel. error  %.16e\n", static_cast<double>(worstScaledRelativeError));
    std::printf("worst simplex extrap.   %.16e\n", static_cast<double>(worstSimplexExtrapolationError));
    std::printf("worst simplex derivative %.16e\n", static_cast<double>(worstSimplexDerivativeError));
    std::printf("simplex/legacy max diff %.16e\n", static_cast<double>(worstSimplexLegacyDifference));
    std::printf("simplex max condition   %.16e\n", static_cast<double>(worstSimplexCondition));
    std::printf("simplex inverse residual %.16e\n", static_cast<double>(worstSimplexInverseResidual));
    if (failures == 0) std::printf("PASS nonmatching_geometry\n");
    else std::printf("FAIL nonmatching_geometry: %d check(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
