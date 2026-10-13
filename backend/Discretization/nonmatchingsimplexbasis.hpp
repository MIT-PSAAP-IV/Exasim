/*
    Polynomially extended nodal basis for 2D non-matching simplex setup.

    The modal basis is shared with MeshTransfer: total-degree monomials
    zeta_1^a zeta_2^b, a+b <= p. Unlike the solver's collapsed-coordinate
    basis, this representation remains polynomial outside the reference
    triangle, as required by non-matching extrapolation.
*/

#ifndef __NONMATCHINGSIMPLEXBASIS_HPP
#define __NONMATCHINGSIMPLEXBASIS_HPP

#include "../MeshTransfer/include/basis.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace exasim {
namespace nonmatching {

class SimplexNodalBasis2D {
public:
    SimplexNodalBasis2D() = default;

    SimplexNodalBasis2D(const dstype* xpe, Int npe, Int porder)
    {
        initialize(xpe, npe, porder);
    }

    void initialize(const dstype* xpe, Int npe, Int porder)
    {
        const Int expected = meshtransfer::detail::expected_nodes<dstype>(0, porder, 2);
        if (xpe == nullptr || porder < 1 || porder > 4 || npe != expected)
            throw std::runtime_error("non-matching simplex basis: invalid master element");

        porder_ = porder;
        npe_ = npe;
        std::vector<dstype> matrix(static_cast<std::size_t>(npe * npe));
        inverse_.assign(static_cast<std::size_t>(npe * npe), 0.0);
        std::vector<dstype> modal(static_cast<std::size_t>(npe));
        dstype xi[2] = {0.0, 0.0};
        for (Int a = 0; a < npe; ++a) {
            xi[0] = xpe[a];
            xi[1] = xpe[a + npe];
            meshtransfer::detail::modal_basis(
                xi, modal.data(), static_cast<dstype*>(nullptr), 0, porder, 2);
            for (Int m = 0; m < npe; ++m)
                matrix[static_cast<std::size_t>(a * npe + m)] = modal[m];
            inverse_[static_cast<std::size_t>(a * npe + a)] = 1.0;
        }
        const std::vector<dstype> original = matrix;

        matrixNorm_ = infinity_norm(matrix, npe);
        std::vector<dstype> rowScale(static_cast<std::size_t>(npe), 0.0);
        for (Int row = 0; row < npe; ++row)
            for (Int col = 0; col < npe; ++col)
                rowScale[row] = std::max(
                    rowScale[row], std::abs(matrix[static_cast<std::size_t>(row * npe + col)]));

        const dstype eps = std::numeric_limits<dstype>::epsilon();
        for (Int col = 0; col < npe; ++col) {
            Int pivot = -1;
            dstype bestRatio = -1.0;
            for (Int row = col; row < npe; ++row) {
                if (!(rowScale[row] > 0.0)) continue;
                const dstype ratio =
                    std::abs(matrix[static_cast<std::size_t>(row * npe + col)]) / rowScale[row];
                if (ratio > bestRatio) {
                    bestRatio = ratio;
                    pivot = row;
                }
            }
            if (pivot < 0 || !(bestRatio > 64.0 * eps))
                throw std::runtime_error("non-matching simplex basis: singular Vandermonde matrix");
            if (pivot != col) {
                for (Int j = 0; j < npe; ++j) {
                    std::swap(matrix[static_cast<std::size_t>(col * npe + j)],
                              matrix[static_cast<std::size_t>(pivot * npe + j)]);
                    std::swap(inverse_[static_cast<std::size_t>(col * npe + j)],
                              inverse_[static_cast<std::size_t>(pivot * npe + j)]);
                }
                std::swap(rowScale[col], rowScale[pivot]);
            }
            const dstype diagonal = matrix[static_cast<std::size_t>(col * npe + col)];
            for (Int j = 0; j < npe; ++j) {
                matrix[static_cast<std::size_t>(col * npe + j)] /= diagonal;
                inverse_[static_cast<std::size_t>(col * npe + j)] /= diagonal;
            }
            for (Int row = 0; row < npe; ++row) {
                if (row == col) continue;
                const dstype factor = matrix[static_cast<std::size_t>(row * npe + col)];
                for (Int j = 0; j < npe; ++j) {
                    matrix[static_cast<std::size_t>(row * npe + j)] -=
                        factor * matrix[static_cast<std::size_t>(col * npe + j)];
                    inverse_[static_cast<std::size_t>(row * npe + j)] -=
                        factor * inverse_[static_cast<std::size_t>(col * npe + j)];
                }
            }
        }

        conditionEstimate_ = matrixNorm_ * infinity_norm(inverse_, npe);
        inverseResidual_ = product_identity_residual(original, inverse_, npe);
        const dstype conditionLimit = 1.0 / (4096.0 * eps);
        const dstype residualLimit = 512.0 * eps * std::max<dstype>(1.0, conditionEstimate_);
        if (conditionEstimate_ > conditionLimit || inverseResidual_ > residualLimit)
            throw std::runtime_error("non-matching simplex basis: ill-conditioned Vandermonde matrix");
    }

    bool initialized() const noexcept { return !inverse_.empty(); }
    dstype condition_estimate() const noexcept { return conditionEstimate_; }
    dstype inverse_residual() const noexcept { return inverseResidual_; }

    void evaluate(dstype* shape, dstype* derivative, const dstype* zeta) const
    {
        if (!initialized() || shape == nullptr || zeta == nullptr)
            throw std::runtime_error("non-matching simplex basis: uninitialized evaluation");
        std::vector<dstype> modal(static_cast<std::size_t>(npe_));
        std::vector<dstype> dmodal;
        if (derivative) dmodal.resize(static_cast<std::size_t>(2 * npe_));
        meshtransfer::detail::modal_basis(
            zeta, modal.data(), derivative ? dmodal.data() : nullptr, 0, porder_, 2);
        for (Int a = 0; a < npe_; ++a) {
            dstype value = 0.0;
            for (Int m = 0; m < npe_; ++m)
                value += modal[m] * inverse_[static_cast<std::size_t>(m * npe_ + a)];
            shape[a] = value;
            if (derivative) {
                for (Int d = 0; d < 2; ++d) {
                    dstype gradient = 0.0;
                    for (Int m = 0; m < npe_; ++m)
                        gradient += dmodal[m + npe_ * d] *
                            inverse_[static_cast<std::size_t>(m * npe_ + a)];
                    derivative[a + npe_ * d] = gradient;
                }
            }
        }
    }

private:
    static dstype infinity_norm(const std::vector<dstype>& matrix, Int n)
    {
        dstype norm = 0.0;
        for (Int row = 0; row < n; ++row) {
            dstype sum = 0.0;
            for (Int col = 0; col < n; ++col)
                sum += std::abs(matrix[static_cast<std::size_t>(row * n + col)]);
            norm = std::max(norm, sum);
        }
        return norm;
    }

    static dstype product_identity_residual(
        const std::vector<dstype>& matrix,
        const std::vector<dstype>& inverse,
        Int n)
    {
        dstype residual = 0.0;
        for (Int row = 0; row < n; ++row) {
            dstype rowSum = 0.0;
            for (Int col = 0; col < n; ++col) {
                dstype value = 0.0;
                for (Int k = 0; k < n; ++k)
                    value += matrix[static_cast<std::size_t>(row * n + k)] *
                             inverse[static_cast<std::size_t>(k * n + col)];
                rowSum += std::abs(value - (row == col ? 1.0 : 0.0));
            }
            residual = std::max(residual, rowSum);
        }
        return residual;
    }

    Int porder_ = 0;
    Int npe_ = 0;
    std::vector<dstype> inverse_;
    dstype matrixNorm_ = 0.0;
    dstype conditionEstimate_ = 0.0;
    dstype inverseResidual_ = 0.0;
};

} // namespace nonmatching
} // namespace exasim

#endif
