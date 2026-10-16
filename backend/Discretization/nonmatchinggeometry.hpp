/*
    Host-only geometry kernels for non-matching interfaces.

    These routines implement setup Steps 1-3 of
    docs/nonmatching/nonmatching_multiphysics.tex. They intentionally reuse the
    existing PointLocator face-shape and small-system helpers. Two-dimensional
    simplex volume maps use MeshTransfer's total-degree monomials through the
    non-matching nodal wrapper, because the solver's collapsed-coordinate basis
    is not a polynomial continuation outside the reference triangle. No solver
    state is accessed and no reference coordinate is clipped on the primary
    inverse-map path.
*/

#ifndef __NONMATCHINGGEOMETRY_HPP
#define __NONMATCHINGGEOMETRY_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

// makemaster.hpp contains inline permutation helpers whose definitions refer to
// xiny2. The full solver declares it earlier through connectivity.hpp; this
// declaration preserves that include contract for this standalone header.
template <typename T>
void xiny2(int*, const T*, const T*, int, int, int, double);

#include "../PointLocator/pointlocation.hpp"
#include "nonmatchingsimplexbasis.hpp"

namespace exasim {
namespace nonmatching {

enum class InverseMapStatus : int {
    Failed = 0,
    Converged = 1,          // unrestricted solve reached the physical tolerance
    ConvergedProjected = 2, // safeguarded solve reached the physical tolerance
    ProjectedInexact = 3    // finite safeguarded least-squares extrapolation
};

struct GeometryPointStatus {
    bool projectionConverged = false;
    InverseMapStatus inverseStatus = InverseMapStatus::Failed;
    dstype distance = std::numeric_limits<dstype>::max();
};

inline Int FaceElementType(Int nd, Int elemtype)
{
    return (nd == 2) ? 0 : elemtype;
}

inline dstype SquaredNorm(const dstype* x, Int n)
{
    dstype value = 0.0;
    for (Int i = 0; i < n; ++i) value += x[i] * x[i];
    return value;
}

inline bool UsesPolynomialSimplexBasis(Int nd, Int elemtype)
{
    return nd == 2 && elemtype == 0;
}

inline void EvaluateNonmatchingVolumeShapeAndGradients(
    dstype* shape, dstype* derivative, const dstype* zeta, const dstype* xpe,
    Int nd, Int npe, Int elemtype, Int porder,
    const SimplexNodalBasis2D* simplexBasis = nullptr)
{
    if (UsesPolynomialSimplexBasis(nd, elemtype)) {
        if (simplexBasis != nullptr) {
            simplexBasis->evaluate(shape, derivative, zeta);
            return;
        }
        const SimplexNodalBasis2D local(xpe, npe, porder);
        local.evaluate(shape, derivative, zeta);
        return;
    }
    EvaluateVolumeShapeAndGradientsAtReferencePoint(
        shape, derivative, zeta, xpe, nd, npe, elemtype, porder);
}

inline void EvaluateNonmatchingVolumeShape(
    dstype* shape, const dstype* zeta, const dstype* xpe,
    Int nd, Int npe, Int elemtype, Int porder,
    const SimplexNodalBasis2D* simplexBasis = nullptr)
{
    if (UsesPolynomialSimplexBasis(nd, elemtype)) {
        if (simplexBasis != nullptr) {
            simplexBasis->evaluate(shape, nullptr, zeta);
            return;
        }
        const SimplexNodalBasis2D local(xpe, npe, porder);
        local.evaluate(shape, nullptr, zeta);
        return;
    }
    EvaluateVolumeShapeAtReferencePoint(
        shape, zeta, xpe, nd, npe, elemtype, porder);
}

inline void MapNonmatchingReferenceToPhysical(
    dstype* x, dstype* jacobian, dstype* shape, dstype* derivative,
    const dstype* zeta, const dstype* elementNodes, const dstype* xpe,
    Int nd, Int npe, Int elemtype, Int porder,
    const SimplexNodalBasis2D* simplexBasis = nullptr)
{
    EvaluateNonmatchingVolumeShapeAndGradients(
        shape, derivative, zeta, xpe, nd, npe, elemtype, porder, simplexBasis);
    for (Int d = 0; d < nd; ++d) x[d] = 0.0;
    for (Int i = 0; i < nd * nd; ++i) jacobian[i] = 0.0;
    for (Int a = 0; a < npe; ++a) {
        for (Int d = 0; d < nd; ++d) {
            const dstype coordinate = elementNodes[a + npe * d];
            x[d] += shape[a] * coordinate;
            for (Int k = 0; k < nd; ++k)
                jacobian[d * nd + k] += derivative[a + npe * k] * coordinate;
        }
    }
}

// Normalize the local system before using PointLocator's small direct solve.
// This preserves the solution while making its absolute determinant test
// independent of the physical size of a face or element.
inline bool SolveScaleInvariantSmallSystem(
    dstype* x, const dstype* matrix, const dstype* rhs, Int n)
{
    if (x == nullptr || matrix == nullptr || rhs == nullptr || n < 1 || n > 3)
        return false;

    dstype scale = 0.0;
    for (Int i = 0; i < n * n; ++i)
        scale = std::max(scale, std::abs(matrix[i]));
    if (!(scale > 0.0)) return false;

    dstype scaledMatrix[9] = {0.0};
    dstype scaledRhs[3] = {0.0};
    for (Int i = 0; i < n * n; ++i) scaledMatrix[i] = matrix[i] / scale;
    for (Int i = 0; i < n; ++i) scaledRhs[i] = rhs[i] / scale;
    return SolveSmallLinearSystem(x, scaledMatrix, scaledRhs, n);
}

inline void EvaluateFaceShapeAndGradients(
    dstype* psi, dstype* dpsi, const dstype* xi, const dstype* xpf,
    Int nd, Int npf, Int elemtype, Int porder)
{
    const Int ndf = nd - 1;
    EvaluateVolumeShapeAndGradientsAtReferencePoint(
        psi, dpsi, xi, xpf, ndf, npf,
        FaceElementType(nd, elemtype), porder);
}

inline void EvaluateFaceShape(
    dstype* psi, const dstype* xi, const dstype* xpf,
    Int nd, Int npf, Int elemtype, Int porder)
{
    EvaluateVolumeShapeAtReferencePoint(
        psi, xi, xpf, nd - 1, npf,
        FaceElementType(nd, elemtype), porder);
}

inline void MapReferenceFaceToPhysical(
    dstype* x, dstype* tangent, dstype* psi, dstype* dpsi,
    const dstype* xi, const dstype* faceNodes, const dstype* xpf,
    Int nd, Int npf, Int elemtype, Int porder)
{
    const Int ndf = nd - 1;
    EvaluateFaceShapeAndGradients(
        psi, dpsi, xi, xpf, nd, npf, elemtype, porder);

    for (Int d = 0; d < nd; ++d) x[d] = 0.0;
    for (Int i = 0; i < nd * ndf; ++i) tangent[i] = 0.0;

    for (Int a = 0; a < npf; ++a) {
        for (Int d = 0; d < nd; ++d) {
            const dstype ya = faceNodes[a + npf * d];
            x[d] += psi[a] * ya;
            for (Int k = 0; k < ndf; ++k)
                tangent[d * ndf + k] += dpsi[k * npf + a] * ya;
        }
    }
}

inline bool PointInReferenceFace(const dstype* xi, Int nd, Int elemtype)
{
    if (nd == 2) return xi[0] >= 0.0 && xi[0] <= 1.0;
    if (elemtype == 0)
        return xi[0] >= 0.0 && xi[1] >= 0.0 && xi[0] + xi[1] <= 1.0;
    return xi[0] >= 0.0 && xi[0] <= 1.0 && xi[1] >= 0.0 && xi[1] <= 1.0;
}

inline dstype MetricDistanceSquared(
    const dstype* x, const dstype* y, const dstype* metric, Int n)
{
    dstype value = 0.0;
    for (Int i = 0; i < n; ++i)
        for (Int j = 0; j < n; ++j)
            value += (x[i] - y[i]) * metric[i * n + j] * (x[j] - y[j]);
    return value;
}

// Project y onto the reference face in the metric tangent^T*tangent, eq. (22).
inline bool ProjectReferenceFaceMetric(
    dstype* projected, const dstype* y, const dstype* metric,
    Int nd, Int elemtype)
{
    if (projected == nullptr || y == nullptr || metric == nullptr ||
        (nd != 2 && nd != 3))
        return false;
    const Int ndf = nd - 1;
    dstype metricScale = 0.0;
    for (Int i = 0; i < ndf * ndf; ++i)
        metricScale = std::max(metricScale, std::abs(metric[i]));
    if (!(metricScale > 0.0)) return false;

    dstype scaledMetric[4] = {0.0};
    for (Int i = 0; i < ndf * ndf; ++i)
        scaledMetric[i] = metric[i] / metricScale;

    if (ndf == 1) {
        if (!(scaledMetric[0] > 0.0)) return false;
        projected[0] = std::max(static_cast<dstype>(0.0),
                                std::min(static_cast<dstype>(1.0), y[0]));
        return true;
    }
    if (ndf != 2) return false;

    const dstype det =
        scaledMetric[0] * scaledMetric[3] - scaledMetric[1] * scaledMetric[2];
    if (!(scaledMetric[0] > 0.0) ||
        !(det > std::numeric_limits<dstype>::epsilon()))
        return false;

    if (PointInReferenceFace(y, nd, elemtype)) {
        projected[0] = y[0];
        projected[1] = y[1];
        return true;
    }

    const dstype tri[6] = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    const dstype quad[8] = {0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0};
    const dstype* vertices = (elemtype == 0) ? tri : quad;
    const Int nvertices = (elemtype == 0) ? 3 : 4;
    dstype best = std::numeric_limits<dstype>::max();
    bool found = false;

    for (Int edge = 0; edge < nvertices; ++edge) {
        const Int next = (edge + 1) % nvertices;
        const dstype v0[2] = {vertices[2 * edge], vertices[2 * edge + 1]};
        const dstype e[2] = {
            vertices[2 * next] - v0[0], vertices[2 * next + 1] - v0[1]};
        const dstype r[2] = {v0[0] - y[0], v0[1] - y[1]};
        const dstype he[2] = {
            scaledMetric[0] * e[0] + scaledMetric[1] * e[1],
            scaledMetric[2] * e[0] + scaledMetric[3] * e[1]};
        const dstype denom = e[0] * he[0] + e[1] * he[1];
        if (!(denom > std::numeric_limits<dstype>::epsilon())) continue;

        dstype t = -(r[0] * he[0] + r[1] * he[1]) / denom;
        t = std::max(static_cast<dstype>(0.0),
                     std::min(static_cast<dstype>(1.0), t));
        const dstype candidate[2] = {v0[0] + t * e[0], v0[1] + t * e[1]};
        const dstype value =
            MetricDistanceSquared(candidate, y, scaledMetric, 2);
        if (value < best) {
            best = value;
            found = true;
            projected[0] = candidate[0];
            projected[1] = candidate[1];
        }
    }
    return found;
}

// Projected Gauss-Newton closest-point solve on a polynomial face, eqs. (21)-(22).
inline bool ClosestPointOnFace(
    dstype* xi, dstype* xbar, dstype* distance,
    const dstype* x, const dstype* faceNodes, const dstype* xpf,
    Int nd, Int npf, Int elemtype, Int porder,
    Int maxIterations, dstype tolerance)
{
    if (xi == nullptr || xbar == nullptr || distance == nullptr || x == nullptr ||
        faceNodes == nullptr || xpf == nullptr || (nd != 2 && nd != 3) ||
        npf <= 0 || maxIterations <= 0 || tolerance <= 0.0)
        return false;

    const Int ndf = nd - 1;
    xi[0] = (ndf == 1 || elemtype == 1) ? 0.5 : 1.0 / 3.0;
    if (ndf == 2) xi[1] = (elemtype == 1) ? 0.5 : 1.0 / 3.0;

    std::vector<dstype> psi(static_cast<size_t>(npf));
    std::vector<dstype> dpsi(static_cast<size_t>(npf * ndf));
    dstype tangent[6] = {0.0};
    dstype residual[3] = {0.0};
    dstype metric[4] = {0.0};
    dstype gradient[2] = {0.0};
    dstype delta[2] = {0.0};
    dstype trial[2] = {0.0};
    dstype projected[2] = {0.0};

    for (Int iteration = 0; iteration < maxIterations; ++iteration) {
        MapReferenceFaceToPhysical(
            xbar, tangent, psi.data(), dpsi.data(), xi,
            faceNodes, xpf, nd, npf, elemtype, porder);
        for (Int d = 0; d < nd; ++d) residual[d] = xbar[d] - x[d];

        for (Int i = 0; i < ndf; ++i) {
            gradient[i] = 0.0;
            for (Int j = 0; j < ndf; ++j) metric[i * ndf + j] = 0.0;
            for (Int d = 0; d < nd; ++d) {
                gradient[i] += tangent[d * ndf + i] * residual[d];
                for (Int j = 0; j < ndf; ++j)
                    metric[i * ndf + j] +=
                        tangent[d * ndf + i] * tangent[d * ndf + j];
            }
        }

        for (Int i = 0; i < ndf; ++i) gradient[i] = -gradient[i];
        if (!SolveScaleInvariantSmallSystem(delta, metric, gradient, ndf))
            return false;
        for (Int i = 0; i < ndf; ++i) trial[i] = xi[i] + delta[i];
        if (!ProjectReferenceFaceMetric(projected, trial, metric, nd, elemtype))
            return false;

        dstype step2 = 0.0;
        for (Int i = 0; i < ndf; ++i) {
            const dstype step = projected[i] - xi[i];
            step2 += step * step;
            xi[i] = projected[i];
        }
        if (std::sqrt(step2) <= tolerance) {
            MapReferenceFaceToPhysical(
                xbar, tangent, psi.data(), dpsi.data(), xi,
                faceNodes, xpf, nd, npf, elemtype, porder);
            for (Int d = 0; d < nd; ++d) residual[d] = xbar[d] - x[d];
            *distance = std::sqrt(SquaredNorm(residual, nd));
            return true;
        }
    }

    MapReferenceFaceToPhysical(
        xbar, tangent, psi.data(), dpsi.data(), xi,
        faceNodes, xpf, nd, npf, elemtype, porder);
    for (Int d = 0; d < nd; ++d) residual[d] = xbar[d] - x[d];
    *distance = std::sqrt(SquaredNorm(residual, nd));
    return false;
}

// chi_a: reference face -> face a of the reference element.
inline void FaceToElementReference(
    dstype* zetaBar, dstype* psi, const dstype* xi,
    const dstype* xpe, const dstype* xpf, const Int* perm,
    Int face, Int nd, Int npe, Int npf, Int elemtype, Int porder)
{
    EvaluateFaceShape(psi, xi, xpf, nd, npf, elemtype, porder);
    for (Int d = 0; d < nd; ++d) {
        zetaBar[d] = 0.0;
        for (Int j = 0; j < npf; ++j) {
            const Int node = perm[j + npf * face];
            zetaBar[d] += psi[j] * xpe[node + npe * d];
        }
    }
}

inline void ProjectToReferenceNeighborhood(
    dstype* projected, const dstype* xi, Int nd, Int elemtype, dstype margin)
{
    if (elemtype == 1) {
        for (Int d = 0; d < nd; ++d)
            projected[d] = std::max(
                -margin,
                std::min(static_cast<dstype>(1.0) + margin, xi[d]));
        return;
    }

    // The expanded simplex is zeta_d >= -margin and
    // sum(zeta_d) <= 1 + margin. Thus its slanted face moves a normal
    // distance margin/sqrt(nd), while each coordinate face moves margin.
    const dstype scale =
        static_cast<dstype>(1.0) + static_cast<dstype>(nd + 1) * margin;
    dstype transformed[3] = {0.0, 0.0, 0.0};
    dstype simplexProjection[3] = {0.0, 0.0, 0.0};
    dstype work[3] = {0.0, 0.0, 0.0};
    for (Int d = 0; d < nd; ++d)
        transformed[d] = (xi[d] + margin) / scale;
    ProjectPointToReferenceElement(
        simplexProjection, transformed, work, nd, static_cast<Int>(0));
    for (Int d = 0; d < nd; ++d)
        projected[d] = scale * simplexProjection[d] - margin;
}

enum class NewtonMapStatus : int {
    Failed = 0,
    Exact = 1,
    Inexact = 2
};

inline NewtonMapStatus NewtonInverseMap(
    dstype* zeta, dstype* residualNorm,
    const dstype* x, const dstype* elementNodes, const dstype* xpe,
    Int nd, Int npe, Int elemtype, Int porder,
    Int maxIterations, dstype tolerance,
    bool project, dstype projectionMargin,
    const SimplexNodalBasis2D* simplexBasis)
{
    std::vector<dstype> shape(static_cast<size_t>(npe));
    std::vector<dstype> dshape(static_cast<size_t>(npe * nd));
    dstype xcur[3] = {0.0, 0.0, 0.0};
    dstype jacobian[9] = {0.0};
    dstype rhs[3] = {0.0, 0.0, 0.0};
    dstype delta[3] = {0.0, 0.0, 0.0};
    dstype trial[3] = {0.0, 0.0, 0.0};
    dstype bestZeta[3] = {0.0, 0.0, 0.0};
    dstype bestResidual = std::numeric_limits<dstype>::max();

    for (Int iteration = 0; iteration < maxIterations; ++iteration) {
        MapNonmatchingReferenceToPhysical(
            xcur, jacobian, shape.data(), dshape.data(), zeta,
            elementNodes, xpe, nd, npe, elemtype, porder, simplexBasis);
        for (Int d = 0; d < nd; ++d) rhs[d] = x[d] - xcur[d];
        *residualNorm = std::sqrt(SquaredNorm(rhs, nd));
        if (*residualNorm < bestResidual) {
            bestResidual = *residualNorm;
            for (Int d = 0; d < nd; ++d) bestZeta[d] = zeta[d];
        }
        if (*residualNorm <= tolerance) return NewtonMapStatus::Exact;
        if (!SolveScaleInvariantSmallSystem(delta, jacobian, rhs, nd))
            return NewtonMapStatus::Failed;

        for (Int d = 0; d < nd; ++d) trial[d] = zeta[d] + delta[d];
        if (!project) {
            for (Int d = 0; d < nd; ++d) zeta[d] = trial[d];
            continue;
        }

        // Safeguarded projected Gauss-Newton for the least-squares problem of
        // eq. (29). Retain the best point and backtrack until the
        // physical residual decreases. Stagnation is a valid projected result.
        bool accepted = false;
        dstype alpha = 1.0;
        for (Int line = 0; line < 32; ++line) {
            for (Int d = 0; d < nd; ++d) trial[d] = zeta[d] + alpha * delta[d];
            ProjectToReferenceNeighborhood(trial, trial, nd, elemtype, projectionMargin);
            MapNonmatchingReferenceToPhysical(
                xcur, jacobian, shape.data(), dshape.data(), trial,
                elementNodes, xpe, nd, npe, elemtype, porder, simplexBasis);
            for (Int d = 0; d < nd; ++d) rhs[d] = x[d] - xcur[d];
            const dstype trialResidual = std::sqrt(SquaredNorm(rhs, nd));
            if (trialResidual < bestResidual) {
                bestResidual = trialResidual;
                for (Int d = 0; d < nd; ++d) bestZeta[d] = trial[d];
            }
            if (trialResidual < *residualNorm) {
                for (Int d = 0; d < nd; ++d) zeta[d] = trial[d];
                *residualNorm = trialResidual;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if (!accepted) {
            for (Int d = 0; d < nd; ++d) zeta[d] = bestZeta[d];
            *residualNorm = bestResidual;
            return NewtonMapStatus::Inexact;
        }
    }

    MapNonmatchingReferenceToPhysical(
        xcur, jacobian, shape.data(), dshape.data(), zeta,
        elementNodes, xpe, nd, npe, elemtype, porder, simplexBasis);
    for (Int d = 0; d < nd; ++d) rhs[d] = x[d] - xcur[d];
    *residualNorm = std::sqrt(SquaredNorm(rhs, nd));
    if (*residualNorm < bestResidual) {
        bestResidual = *residualNorm;
        for (Int d = 0; d < nd; ++d) bestZeta[d] = zeta[d];
    }
    for (Int d = 0; d < nd; ++d) zeta[d] = bestZeta[d];
    *residualNorm = bestResidual;
    return *residualNorm <= tolerance ? NewtonMapStatus::Exact : NewtonMapStatus::Inexact;
}

// Equation (29), first unrestricted and then projected only as a safeguard.
inline InverseMapStatus InverseElementMap(
    dstype* zeta, dstype* residualNorm,
    const dstype* x, const dstype* elementNodes, const dstype* xpe,
    const dstype* initialGuess, Int nd, Int npe, Int elemtype, Int porder,
    Int maxNewtonIterations, Int maxProjectedIterations,
    dstype tolerance, dstype projectionMargin,
    const SimplexNodalBasis2D* suppliedSimplexBasis = nullptr)
{
    if (zeta == nullptr || residualNorm == nullptr || x == nullptr ||
        elementNodes == nullptr || xpe == nullptr || initialGuess == nullptr ||
        (nd != 2 && nd != 3) || npe <= 0 || tolerance <= 0.0 ||
        projectionMargin < 0.0)
        return InverseMapStatus::Failed;

    SimplexNodalBasis2D localSimplexBasis;
    const SimplexNodalBasis2D* simplexBasis = suppliedSimplexBasis;
    if (UsesPolynomialSimplexBasis(nd, elemtype) && simplexBasis == nullptr) {
        localSimplexBasis.initialize(xpe, npe, porder);
        simplexBasis = &localSimplexBasis;
    }

    for (Int d = 0; d < nd; ++d) zeta[d] = initialGuess[d];
    if (maxNewtonIterations > 0) {
        const NewtonMapStatus unrestricted = NewtonInverseMap(
                zeta, residualNorm, x, elementNodes, xpe, nd, npe,
                elemtype, porder, maxNewtonIterations, tolerance, false, 0.0,
                simplexBasis);
        if (unrestricted == NewtonMapStatus::Exact) return InverseMapStatus::Converged;
        // A diverged or locally singular unrestricted path still gets the
        // safeguarded restart. Only failure of that projected solve is final.
    }

    for (Int d = 0; d < nd; ++d) zeta[d] = initialGuess[d];
    ProjectToReferenceNeighborhood(
        zeta, zeta, nd, elemtype, projectionMargin);
    if (maxProjectedIterations <= 0) return InverseMapStatus::Failed;
    const NewtonMapStatus projected = NewtonInverseMap(
            zeta, residualNorm, x, elementNodes, xpe, nd, npe,
            elemtype, porder, maxProjectedIterations, tolerance, true,
            projectionMargin, simplexBasis);
    if (projected == NewtonMapStatus::Failed) return InverseMapStatus::Failed;
    return projected == NewtonMapStatus::Exact
        ? InverseMapStatus::ConvergedProjected
        : InverseMapStatus::ProjectedInexact;
}

inline GeometryPointStatus ComputeGeometryPointData(
    dstype* xi, dstype* zeta, dstype* zetaBar, dstype* xbar,
    dstype* phi, dstype* phiBar, dstype* psi, dstype* offset,
    const dstype* x, const dstype* elementNodes,
    const dstype* xpf, const dstype* xpe, const Int* perm,
    Int face, Int nd, Int npe, Int npf, Int elemtype, Int porder,
    Int maxProjectionIterations, Int maxNewtonIterations,
    Int maxProjectedIterations, dstype projectionTolerance,
    dstype inverseTolerance, dstype projectionMargin,
    const SimplexNodalBasis2D* suppliedSimplexBasis = nullptr)
{
    GeometryPointStatus status;
    if (xi == nullptr || zeta == nullptr || zetaBar == nullptr ||
        xbar == nullptr || phi == nullptr || phiBar == nullptr ||
        psi == nullptr || offset == nullptr || x == nullptr ||
        elementNodes == nullptr || xpf == nullptr || xpe == nullptr ||
        perm == nullptr || face < 0 || (nd != 2 && nd != 3) ||
        npe <= 0 || npf <= 0)
        return status;

    // The donor-face geometry must use the element-local ordering selected by
    // perm. Gathering it here prevents an independently ordered face array
    // from making psi(xi) inconsistent with phibar(chi_a(xi)).
    std::vector<dstype> faceNodes(static_cast<size_t>(npf * nd));
    for (Int j = 0; j < npf; ++j) {
        const Int node = perm[j + npf * face];
        if (node < 0 || node >= npe) return status;
        for (Int d = 0; d < nd; ++d)
            faceNodes[j + npf * d] = elementNodes[node + npe * d];
    }

    status.projectionConverged = ClosestPointOnFace(
        xi, xbar, &status.distance, x, faceNodes.data(), xpf,
        nd, npf, elemtype, porder, maxProjectionIterations,
        projectionTolerance);
    if (!status.projectionConverged) return status;

    FaceToElementReference(
        zetaBar, psi, xi, xpe, xpf, perm, face,
        nd, npe, npf, elemtype, porder);
    dstype residualNorm = std::numeric_limits<dstype>::max();
    SimplexNodalBasis2D localSimplexBasis;
    const SimplexNodalBasis2D* simplexBasis = suppliedSimplexBasis;
    if (UsesPolynomialSimplexBasis(nd, elemtype) && simplexBasis == nullptr) {
        localSimplexBasis.initialize(xpe, npe, porder);
        simplexBasis = &localSimplexBasis;
    }
    status.inverseStatus = InverseElementMap(
        zeta, &residualNorm, x, elementNodes, xpe, zetaBar,
        nd, npe, elemtype, porder, maxNewtonIterations,
        maxProjectedIterations, inverseTolerance, projectionMargin, simplexBasis);
    if (status.inverseStatus == InverseMapStatus::Failed) return status;

    EvaluateNonmatchingVolumeShape(
        phi, zeta, xpe, nd, npe, elemtype, porder, simplexBasis);
    EvaluateNonmatchingVolumeShape(
        phiBar, zetaBar, xpe, nd, npe, elemtype, porder, simplexBasis);
    for (Int d = 0; d < nd; ++d) offset[d] = x[d] - xbar[d]; // eq. (23)
    return status;
}

} // namespace nonmatching
} // namespace exasim

#endif
