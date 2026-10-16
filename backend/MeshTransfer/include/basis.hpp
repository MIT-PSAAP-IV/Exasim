#pragma once

#include "types.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>

namespace exasim::meshtransfer::detail {

inline constexpr int max_polynomial_order = 8;
inline constexpr int max_element_nodes = 256;

template <class Scalar, class Index>
KOKKOS_INLINE_FUNCTION
Index expected_nodes(Index elemtype, Index porder, Index nd)
{
    if (elemtype != 0) {
        Index n = 1;
        for (Index d = 0; d < nd; ++d)
            n *= porder + 1;
        return n;
    }
    if (nd == 1)
        return porder + 1;
    if (nd == 2)
        return (porder + 1) * (porder + 2) / 2;
    return (porder + 1) * (porder + 2) * (porder + 3) / 6;
}

template <class Scalar, class Index>
KOKKOS_INLINE_FUNCTION
void shifted_legendre(Scalar x, Index porder, Scalar* value, Scalar* derivative)
{
    value[0] = Scalar(1);
    derivative[0] = Scalar(0);
    if (porder == 0)
        return;
    const Scalar z = Scalar(2) * x - Scalar(1);
    value[1] = z;
    derivative[1] = Scalar(2);
    for (Index n = 1; n < porder; ++n) {
        const Scalar a = Scalar(2 * n + 1) / Scalar(n + 1);
        const Scalar b = Scalar(n) / Scalar(n + 1);
        value[n + 1] = a * z * value[n] - b * value[n - 1];
        derivative[n + 1] =
            a * (Scalar(2) * value[n] + z * derivative[n]) -
            b * derivative[n - 1];
    }
}

template <class Scalar, class Index>
KOKKOS_INLINE_FUNCTION
void modal_basis(
    const Scalar* xi,
    Scalar* phi,
    Scalar* dphi,
    Index elemtype,
    Index porder,
    Index nd)
{
    if (elemtype != 0) {
        Scalar values[3][max_polynomial_order + 1];
        Scalar derivatives[3][max_polynomial_order + 1];
        for (Index d = 0; d < nd; ++d)
            shifted_legendre(xi[d], porder, values[d], derivatives[d]);

        Index m = 0;
        if (nd == 1) {
            for (Index i = 0; i <= porder; ++i, ++m) {
                phi[m] = values[0][i];
                if (dphi) dphi[m] = derivatives[0][i];
            }
        }
        else if (nd == 2) {
            for (Index j = 0; j <= porder; ++j)
                for (Index i = 0; i <= porder; ++i, ++m) {
                    phi[m] = values[0][i] * values[1][j];
                    if (dphi) {
                        dphi[m] = derivatives[0][i] * values[1][j];
                        dphi[m + expected_nodes<Scalar>(elemtype, porder, nd)] =
                            values[0][i] * derivatives[1][j];
                    }
                }
        }
        else {
            const Index npe = expected_nodes<Scalar>(elemtype, porder, nd);
            for (Index k = 0; k <= porder; ++k)
                for (Index j = 0; j <= porder; ++j)
                    for (Index i = 0; i <= porder; ++i, ++m) {
                        phi[m] = values[0][i] * values[1][j] * values[2][k];
                        if (dphi) {
                            dphi[m] = derivatives[0][i] * values[1][j] * values[2][k];
                            dphi[m + npe] =
                                values[0][i] * derivatives[1][j] * values[2][k];
                            dphi[m + 2 * npe] =
                                values[0][i] * values[1][j] * derivatives[2][k];
                        }
                    }
        }
        return;
    }

    Scalar powers[3][max_polynomial_order + 1];
    for (Index d = 0; d < nd; ++d) {
        powers[d][0] = Scalar(1);
        for (Index i = 1; i <= porder; ++i)
            powers[d][i] = powers[d][i - 1] * xi[d];
    }

    Index m = 0;
    if (nd == 1) {
        for (Index i = 0; i <= porder; ++i, ++m) {
            phi[m] = powers[0][i];
            if (dphi)
                dphi[m] = i == 0 ? Scalar(0) : Scalar(i) * powers[0][i - 1];
        }
    }
    else if (nd == 2) {
        const Index npe = expected_nodes<Scalar>(elemtype, porder, nd);
        for (Index j = 0; j <= porder; ++j)
            for (Index i = 0; i <= porder - j; ++i, ++m) {
                phi[m] = powers[0][i] * powers[1][j];
                if (dphi) {
                    dphi[m] = i == 0 ? Scalar(0)
                        : Scalar(i) * powers[0][i - 1] * powers[1][j];
                    dphi[m + npe] = j == 0 ? Scalar(0)
                        : Scalar(j) * powers[0][i] * powers[1][j - 1];
                }
            }
    }
    else {
        const Index npe = expected_nodes<Scalar>(elemtype, porder, nd);
        for (Index k = 0; k <= porder; ++k)
            for (Index j = 0; j <= porder - k; ++j)
                for (Index i = 0; i <= porder - j - k; ++i, ++m) {
                    phi[m] = powers[0][i] * powers[1][j] * powers[2][k];
                    if (dphi) {
                        dphi[m] = i == 0 ? Scalar(0)
                            : Scalar(i) * powers[0][i - 1] * powers[1][j] * powers[2][k];
                        dphi[m + npe] = j == 0 ? Scalar(0)
                            : Scalar(j) * powers[0][i] * powers[1][j - 1] * powers[2][k];
                        dphi[m + 2 * npe] = k == 0 ? Scalar(0)
                            : Scalar(k) * powers[0][i] * powers[1][j] * powers[2][k - 1];
                    }
                }
    }
}

} // namespace exasim::meshtransfer::detail
