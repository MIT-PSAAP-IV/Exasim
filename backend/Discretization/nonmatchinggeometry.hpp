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
    return (nd == 2 || nd == 3) && elemtype == 0;
}

inline void EvaluateNonmatchingVolumeShapeAndGradients(
    dstype* shape, dstype* derivative, const dstype* zeta, const dstype* xpe,
    Int nd, Int npe, Int elemtype, Int porder,
    const PolynomialNodalBasis* simplexBasis = nullptr)
{
    if (simplexBasis != nullptr) {
        simplexBasis->evaluate(shape, derivative, zeta);
        return;
    }
    if (UsesPolynomialSimplexBasis(nd, elemtype)) {
        if (nd == 2) {
            const SimplexNodalBasis2D local(xpe, npe, porder);
            local.evaluate(shape, derivative, zeta);
        }
        else {
            const SimplexNodalBasis3D local(xpe, npe, porder);
            local.evaluate(shape, derivative, zeta);
        }
        return;
    }
    EvaluateVolumeShapeAndGradientsAtReferencePoint(
        shape, derivative, zeta, xpe, nd, npe, elemtype, porder);
}

inline void EvaluateNonmatchingVolumeShape(
    dstype* shape, const dstype* zeta, const dstype* xpe,
    Int nd, Int npe, Int elemtype, Int porder,
    const PolynomialNodalBasis* simplexBasis = nullptr)
{
    if (simplexBasis != nullptr) {
        simplexBasis->evaluate(shape, nullptr, zeta);
        return;
    }
    if (UsesPolynomialSimplexBasis(nd, elemtype)) {
        if (nd == 2) {
            const SimplexNodalBasis2D local(xpe, npe, porder);
            local.evaluate(shape, nullptr, zeta);
        }
        else {
            const SimplexNodalBasis3D local(xpe, npe, porder);
            local.evaluate(shape, nullptr, zeta);
        }
        return;
    }
    EvaluateVolumeShapeAtReferencePoint(
        shape, zeta, xpe, nd, npe, elemtype, porder);
}

inline void MapNonmatchingReferenceToPhysical(
    dstype* x, dstype* jacobian, dstype* shape, dstype* derivative,
    const dstype* zeta, const dstype* elementNodes, const dstype* xpe,
    Int nd, Int npe, Int elemtype, Int porder,
    const PolynomialNodalBasis* simplexBasis = nullptr)
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
    Int nd, Int npf, Int elemtype, Int porder,
    const SurfaceNodalBasis* surfaceBasis = nullptr)
{
    const Int ndf = nd - 1;
    if (nd == 3) {
        if (surfaceBasis != nullptr) {
            surfaceBasis->evaluate(psi, dpsi, xi);
            return;
        }
        const SurfaceNodalBasis local(xpf, npf, porder, elemtype);
        local.evaluate(psi, dpsi, xi);
        return;
    }
    EvaluateVolumeShapeAndGradientsAtReferencePoint(
        psi, dpsi, xi, xpf, ndf, npf,
        FaceElementType(nd, elemtype), porder);
}

inline void EvaluateFaceShape(
    dstype* psi, const dstype* xi, const dstype* xpf,
    Int nd, Int npf, Int elemtype, Int porder,
    const SurfaceNodalBasis* surfaceBasis = nullptr)
{
    if (nd == 3) {
        if (surfaceBasis != nullptr) {
            surfaceBasis->evaluate(psi, nullptr, xi);
            return;
        }
        const SurfaceNodalBasis local(xpf, npf, porder, elemtype);
        local.evaluate(psi, nullptr, xi);
        return;
    }
    EvaluateVolumeShapeAtReferencePoint(
        psi, xi, xpf, nd - 1, npf,
        FaceElementType(nd, elemtype), porder);
}

// chi_a: reference face -> face a of the reference element.
inline void FaceToElementReference(
    dstype* zetaBar, dstype* psi, const dstype* xi,
    const dstype* xpe, const dstype* xpf, const Int* perm,
    Int face, Int nd, Int npe, Int npf, Int elemtype, Int porder,
    const SurfaceNodalBasis* surfaceBasis = nullptr)
{
    EvaluateFaceShape(psi, xi, xpf, nd, npf, elemtype, porder, surfaceBasis);
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
    const PolynomialNodalBasis* simplexBasis)
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
    const PolynomialNodalBasis* suppliedSimplexBasis = nullptr)
{
    if (zeta == nullptr || residualNorm == nullptr || x == nullptr ||
        elementNodes == nullptr || xpe == nullptr || initialGuess == nullptr ||
        (nd != 2 && nd != 3) || npe <= 0 || tolerance <= 0.0 ||
        projectionMargin < 0.0)
        return InverseMapStatus::Failed;

    SimplexNodalBasis2D localSimplexBasis2D;
    SimplexNodalBasis3D localSimplexBasis3D;
    const PolynomialNodalBasis* simplexBasis = suppliedSimplexBasis;
    if (UsesPolynomialSimplexBasis(nd, elemtype) && simplexBasis == nullptr) {
        if (nd == 2) {
            localSimplexBasis2D.initialize(xpe, npe, porder);
            simplexBasis = &localSimplexBasis2D;
        }
        else {
            localSimplexBasis3D.initialize(xpe, npe, porder);
            simplexBasis = &localSimplexBasis3D;
        }
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

} // namespace nonmatching
} // namespace exasim

#endif
