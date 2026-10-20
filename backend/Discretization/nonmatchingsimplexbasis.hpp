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

class PolynomialNodalBasis {
public:
    PolynomialNodalBasis() = default;

    PolynomialNodalBasis(
        const dstype* nodes, Int nnode, Int porder, Int dimension, Int elemtype)
    {
        initialize(nodes, nnode, porder, dimension, elemtype);
    }

    void initialize(
        const dstype* nodes, Int nnode, Int porder, Int dimension, Int elemtype)
    {
        const Int expected = meshtransfer::detail::expected_nodes<dstype>(
            elemtype, porder, dimension);
        if (nodes == nullptr || porder < 1 || porder > 4 ||
            (dimension != 2 && dimension != 3) ||
            (elemtype != 0 && elemtype != 1) || nnode != expected)
            throw std::runtime_error("non-matching polynomial basis: invalid master element");

        porder_ = porder;
        nnode_ = nnode;
        dimension_ = dimension;
        elemtype_ = elemtype;
        if (elemtype_ == 1) {
            initialize_tensor(nodes);
            return;
        }
        std::vector<dstype> matrix(static_cast<std::size_t>(nnode * nnode));
        inverse_.assign(static_cast<std::size_t>(nnode * nnode), 0.0);
        std::vector<dstype> modal(static_cast<std::size_t>(nnode));
        dstype xi[3] = {0.0, 0.0, 0.0};
        for (Int a = 0; a < nnode; ++a) {
            for (Int d = 0; d < dimension; ++d)
                xi[d] = nodes[a + nnode * d];
            meshtransfer::detail::modal_basis(
                xi, modal.data(), static_cast<dstype*>(nullptr),
                elemtype, porder, dimension);
            for (Int m = 0; m < nnode; ++m)
                matrix[static_cast<std::size_t>(a * nnode + m)] = modal[m];
            inverse_[static_cast<std::size_t>(a * nnode + a)] = 1.0;
        }
        const std::vector<dstype> original = matrix;

        matrixNorm_ = infinity_norm(matrix, nnode);
        std::vector<dstype> rowScale(static_cast<std::size_t>(nnode), 0.0);
        for (Int row = 0; row < nnode; ++row)
            for (Int col = 0; col < nnode; ++col)
                rowScale[row] = std::max(
                    rowScale[row], std::abs(matrix[static_cast<std::size_t>(row * nnode + col)]));

        const dstype eps = std::numeric_limits<dstype>::epsilon();
        for (Int col = 0; col < nnode; ++col) {
            Int pivot = -1;
            dstype bestRatio = -1.0;
            for (Int row = col; row < nnode; ++row) {
                if (!(rowScale[row] > 0.0)) continue;
                const dstype ratio =
                    std::abs(matrix[static_cast<std::size_t>(row * nnode + col)]) / rowScale[row];
                if (ratio > bestRatio) {
                    bestRatio = ratio;
                    pivot = row;
                }
            }
            if (pivot < 0 || !(bestRatio > 64.0 * eps))
                throw std::runtime_error("non-matching polynomial basis: singular Vandermonde matrix");
            if (pivot != col) {
                for (Int j = 0; j < nnode; ++j) {
                    std::swap(matrix[static_cast<std::size_t>(col * nnode + j)],
                              matrix[static_cast<std::size_t>(pivot * nnode + j)]);
                    std::swap(inverse_[static_cast<std::size_t>(col * nnode + j)],
                              inverse_[static_cast<std::size_t>(pivot * nnode + j)]);
                }
                std::swap(rowScale[col], rowScale[pivot]);
            }
            const dstype diagonal = matrix[static_cast<std::size_t>(col * nnode + col)];
            for (Int j = 0; j < nnode; ++j) {
                matrix[static_cast<std::size_t>(col * nnode + j)] /= diagonal;
                inverse_[static_cast<std::size_t>(col * nnode + j)] /= diagonal;
            }
            for (Int row = 0; row < nnode; ++row) {
                if (row == col) continue;
                const dstype factor = matrix[static_cast<std::size_t>(row * nnode + col)];
                for (Int j = 0; j < nnode; ++j) {
                    matrix[static_cast<std::size_t>(row * nnode + j)] -=
                        factor * matrix[static_cast<std::size_t>(col * nnode + j)];
                    inverse_[static_cast<std::size_t>(row * nnode + j)] -=
                        factor * inverse_[static_cast<std::size_t>(col * nnode + j)];
                }
            }
        }

        conditionEstimate_ = matrixNorm_ * infinity_norm(inverse_, nnode);
        inverseResidual_ = product_identity_residual(original, inverse_, nnode);
        const dstype conditionLimit = 1.0 / (4096.0 * eps);
        const dstype residualLimit = 512.0 * eps * std::max<dstype>(1.0, conditionEstimate_);
        if (conditionEstimate_ > conditionLimit || inverseResidual_ > residualLimit)
            throw std::runtime_error("non-matching polynomial basis: ill-conditioned Vandermonde matrix");
    }

    bool initialized() const noexcept { return !inverse_.empty() || !tensorInverse_.empty(); }
    dstype condition_estimate() const noexcept { return conditionEstimate_; }
    dstype inverse_residual() const noexcept { return inverseResidual_; }

    void evaluate(dstype* shape, dstype* derivative, const dstype* zeta) const
    {
        if (!initialized() || shape == nullptr || zeta == nullptr)
            throw std::runtime_error("non-matching polynomial basis: uninitialized evaluation");
        if (elemtype_ == 1) {
            evaluate_tensor(shape, derivative, zeta);
            return;
        }
        // p <= 4: a 3D tensor element has at most (p+1)^3 = 125 modes.
        dstype modal[125] = {0.0};
        dstype dmodal[375] = {0.0};
        meshtransfer::detail::modal_basis(
            zeta, modal, derivative ? dmodal : nullptr,
            elemtype_, porder_, dimension_);
        for (Int a = 0; a < nnode_; ++a) {
            dstype value = 0.0;
            for (Int m = 0; m < nnode_; ++m)
                value += modal[m] * inverse_[static_cast<std::size_t>(m * nnode_ + a)];
            shape[a] = value;
            if (derivative) {
                for (Int d = 0; d < dimension_; ++d) {
                    dstype gradient = 0.0;
                    for (Int m = 0; m < nnode_; ++m)
                        gradient += dmodal[m + nnode_ * d] *
                            inverse_[static_cast<std::size_t>(m * nnode_ + a)];
                    derivative[a + nnode_ * d] = gradient;
                }
            }
        }
    }

private:
    void initialize_tensor(const dstype* nodes)
    {
        const Int n1 = porder_ + 1;
        tensorInverse_.assign(static_cast<std::size_t>(dimension_ * n1 * n1), 0.0);
        tensorNodeIndex_.assign(static_cast<std::size_t>(dimension_ * nnode_), -1);
        conditionEstimate_ = 1.0;
        inverseResidual_ = 0.0;
        const dstype eps = std::numeric_limits<dstype>::epsilon();

        for (Int d = 0; d < dimension_; ++d) {
            std::vector<dstype> coordinates;
            coordinates.reserve(n1);
            for (Int a = 0; a < nnode_; ++a) coordinates.push_back(nodes[a + nnode_ * d]);
            std::sort(coordinates.begin(), coordinates.end());
            std::vector<dstype> unique;
            for (dstype value : coordinates) {
                const dstype scale = std::max<dstype>(1.0, std::abs(value));
                if (unique.empty() || std::abs(value - unique.back()) > 64.0 * eps * scale)
                    unique.push_back(value);
            }
            if (static_cast<Int>(unique.size()) != n1)
                throw std::runtime_error("non-matching polynomial basis: invalid tensor nodes");
            for (Int a = 0; a < nnode_; ++a) {
                Int best = 0;
                for (Int i = 1; i < n1; ++i)
                    if (std::abs(nodes[a + nnode_ * d] - unique[i]) <
                        std::abs(nodes[a + nnode_ * d] - unique[best])) best = i;
                tensorNodeIndex_[a + nnode_ * d] = best;
            }

            std::vector<dstype> matrix(static_cast<std::size_t>(n1 * n1));
            std::vector<dstype> original(static_cast<std::size_t>(n1 * n1));
            std::vector<dstype> inverse(static_cast<std::size_t>(n1 * n1), 0.0);
            for (Int i = 0; i < n1; ++i) {
                dstype value[meshtransfer::detail::max_polynomial_order + 1] = {0.0};
                dstype gradient[meshtransfer::detail::max_polynomial_order + 1] = {0.0};
                meshtransfer::detail::shifted_legendre(unique[i], porder_, value, gradient);
                for (Int m = 0; m < n1; ++m) matrix[i * n1 + m] = value[m];
                inverse[i * n1 + i] = 1.0;
            }
            original = matrix;
            const dstype norm = infinity_norm(matrix, n1);
            std::vector<dstype> scale(static_cast<std::size_t>(n1), 0.0);
            for (Int row = 0; row < n1; ++row)
                for (Int col = 0; col < n1; ++col)
                    scale[row] = std::max(scale[row], std::abs(matrix[row * n1 + col]));
            for (Int col = 0; col < n1; ++col) {
                Int pivot = -1;
                dstype ratio = -1.0;
                for (Int row = col; row < n1; ++row) {
                    if (!(scale[row] > 0.0)) continue;
                    const dstype candidate = std::abs(matrix[row * n1 + col]) / scale[row];
                    if (candidate > ratio) { ratio = candidate; pivot = row; }
                }
                if (pivot < 0 || !(ratio > 64.0 * eps))
                    throw std::runtime_error("non-matching polynomial basis: singular tensor Vandermonde matrix");
                if (pivot != col) {
                    for (Int j = 0; j < n1; ++j) {
                        std::swap(matrix[pivot * n1 + j], matrix[col * n1 + j]);
                        std::swap(inverse[pivot * n1 + j], inverse[col * n1 + j]);
                    }
                    std::swap(scale[pivot], scale[col]);
                }
                const dstype diagonal = matrix[col * n1 + col];
                for (Int j = 0; j < n1; ++j) {
                    matrix[col * n1 + j] /= diagonal;
                    inverse[col * n1 + j] /= diagonal;
                }
                for (Int row = 0; row < n1; ++row) {
                    if (row == col) continue;
                    const dstype factor = matrix[row * n1 + col];
                    for (Int j = 0; j < n1; ++j) {
                        matrix[row * n1 + j] -= factor * matrix[col * n1 + j];
                        inverse[row * n1 + j] -= factor * inverse[col * n1 + j];
                    }
                }
            }
            const dstype oneCondition = norm * infinity_norm(inverse, n1);
            conditionEstimate_ *= oneCondition;
            inverseResidual_ = std::max(
                inverseResidual_, product_identity_residual(original, inverse, n1));
            std::copy(inverse.begin(), inverse.end(),
                tensorInverse_.begin() + static_cast<std::size_t>(d * n1 * n1));
        }
        matrixNorm_ = conditionEstimate_;
        const dstype conditionLimit = 1.0 / (4096.0 * eps);
        const dstype residualLimit = 512.0 * eps * std::max<dstype>(1.0, conditionEstimate_);
        if (conditionEstimate_ > conditionLimit || inverseResidual_ > residualLimit)
            throw std::runtime_error("non-matching polynomial basis: ill-conditioned tensor Vandermonde matrix");
    }

    void evaluate_tensor(dstype* shape, dstype* derivative, const dstype* zeta) const
    {
        const Int n1 = porder_ + 1;
        dstype value[3][meshtransfer::detail::max_polynomial_order + 1] = {{0.0}};
        dstype gradient[3][meshtransfer::detail::max_polynomial_order + 1] = {{0.0}};
        for (Int d = 0; d < dimension_; ++d) {
            dstype modal[meshtransfer::detail::max_polynomial_order + 1] = {0.0};
            dstype dmodal[meshtransfer::detail::max_polynomial_order + 1] = {0.0};
            meshtransfer::detail::shifted_legendre(zeta[d], porder_, modal, dmodal);
            const dstype* inverse = &tensorInverse_[static_cast<std::size_t>(d * n1 * n1)];
            for (Int i = 0; i < n1; ++i)
                for (Int m = 0; m < n1; ++m) {
                    value[d][i] += modal[m] * inverse[m * n1 + i];
                    gradient[d][i] += dmodal[m] * inverse[m * n1 + i];
                }
        }
        for (Int a = 0; a < nnode_; ++a) {
            shape[a] = 1.0;
            for (Int d = 0; d < dimension_; ++d)
                shape[a] *= value[d][tensorNodeIndex_[a + nnode_ * d]];
            if (!derivative) continue;
            for (Int d = 0; d < dimension_; ++d) {
                dstype result = gradient[d][tensorNodeIndex_[a + nnode_ * d]];
                for (Int q = 0; q < dimension_; ++q)
                    if (q != d) result *= value[q][tensorNodeIndex_[a + nnode_ * q]];
                derivative[a + nnode_ * d] = result;
            }
        }
    }

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
    Int nnode_ = 0;
    Int dimension_ = 0;
    Int elemtype_ = 0;
    std::vector<dstype> inverse_;
    std::vector<dstype> tensorInverse_;
    std::vector<Int> tensorNodeIndex_;
    dstype matrixNorm_ = 0.0;
    dstype conditionEstimate_ = 0.0;
    dstype inverseResidual_ = 0.0;
};

class SimplexNodalBasis2D : public PolynomialNodalBasis {
public:
    SimplexNodalBasis2D() = default;
    SimplexNodalBasis2D(const dstype* xpe, Int npe, Int porder)
        : PolynomialNodalBasis(xpe, npe, porder, 2, 0) {}
    void initialize(const dstype* xpe, Int npe, Int porder)
    {
        PolynomialNodalBasis::initialize(xpe, npe, porder, 2, 0);
    }
};

class SimplexNodalBasis3D : public PolynomialNodalBasis {
public:
    SimplexNodalBasis3D() = default;
    SimplexNodalBasis3D(const dstype* xpe, Int npe, Int porder)
        : PolynomialNodalBasis(xpe, npe, porder, 3, 0) {}
    void initialize(const dstype* xpe, Int npe, Int porder)
    {
        PolynomialNodalBasis::initialize(xpe, npe, porder, 3, 0);
    }
};

// Nodal polynomial basis on a three-dimensional element face. Triangle faces
// use total-degree monomials and quadrilateral faces use a tensor-product
// basis, both constructed from the model's actual face nodes.
class SurfaceNodalBasis : public PolynomialNodalBasis {
public:
    SurfaceNodalBasis() = default;

    SurfaceNodalBasis(
        const dstype* xpf, Int npf, Int porder, Int volumeElemtype)
        : PolynomialNodalBasis(xpf, npf, porder, 2, volumeElemtype) {}

    void initialize(
        const dstype* xpf, Int npf, Int porder, Int volumeElemtype)
    {
        PolynomialNodalBasis::initialize(
            xpf, npf, porder, 2, volumeElemtype);
    }
};

} // namespace nonmatching
} // namespace exasim

#endif
