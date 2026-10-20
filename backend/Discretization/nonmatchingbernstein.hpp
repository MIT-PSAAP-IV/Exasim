#ifndef __NONMATCHINGBERNSTEIN_HPP
#define __NONMATCHINGBERNSTEIN_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace exasim {
namespace nonmatching {
namespace preprocess {

inline int binomial(int n, int k)
{
    if (k < 0 || k > n) return 0;
    int value = 1;
    for (int i = 1; i <= k; ++i) value = value * (n - k + i) / i;
    return value;
}

template <class T>
inline T power(T value, int exponent)
{
    T result = 1;
    for (int i = 0; i < exponent; ++i) result *= value;
    return result;
}

template <class T>
inline void bernstein_values(T* values, const T* xi, int p, int ndf, int faceType)
{
    int a = 0;
    if (ndf == 1) {
        for (int i = 0; i <= p; ++i)
            values[a++] = static_cast<T>(binomial(p, i)) * power(xi[0], i) *
                          power(static_cast<T>(1) - xi[0], p - i);
        return;
    }
    if (faceType == 0) {
        const T w = static_cast<T>(1) - xi[0] - xi[1];
        for (int i = 0; i <= p; ++i)
            for (int j = 0; j <= p - i; ++j) {
                const int k = p - i - j;
                values[a++] = static_cast<T>(binomial(p, i) * binomial(p - i, j)) *
                              power(xi[0], i) * power(xi[1], j) * power(w, k);
            }
        return;
    }
    for (int j = 0; j <= p; ++j)
        for (int i = 0; i <= p; ++i)
            values[a++] = static_cast<T>(binomial(p, i) * binomial(p, j)) *
                          power(xi[0], i) * power(static_cast<T>(1) - xi[0], p - i) *
                          power(xi[1], j) * power(static_cast<T>(1) - xi[1], p - j);
}

template <class T>
inline T bernstein_1d_term(int p, int i, int derivative, T x)
{
    T sum = 0;
    for (int left = 0; left <= derivative; ++left) {
        const int right = derivative - left;
        if (left > i || right > p - i) continue;
        T coefficient = static_cast<T>(binomial(derivative, left));
        for (int q = 0; q < left; ++q) coefficient *= static_cast<T>(i - q);
        for (int q = 0; q < right; ++q) coefficient *= static_cast<T>(p - i - q);
        if (right % 2) coefficient = -coefficient;
        sum += coefficient * power(x, i - left) *
            power(static_cast<T>(1) - x, p - i - right);
    }
    return static_cast<T>(binomial(p, i)) * sum;
}

template <class T>
inline void bernstein_values_derivatives(T* values, T* derivatives, T* second,
                                         const T* xi, int p, int ndf, int faceType)
{
    int a = 0;
    if (ndf == 1) {
        for (int i = 0; i <= p; ++i) {
            values[a] = bernstein_1d_term(p, i, 0, xi[0]);
            derivatives[a] = bernstein_1d_term(p, i, 1, xi[0]);
            second[a] = bernstein_1d_term(p, i, 2, xi[0]);
            ++a;
        }
        return;
    }
    if (faceType == 0) {
        const T u = xi[0], v = xi[1], w = static_cast<T>(1) - u - v;
        for (int i = 0; i <= p; ++i)
            for (int j = 0; j <= p - i; ++j) {
                const int k = p - i - j;
                const T coefficient = static_cast<T>(binomial(p, i) * binomial(p - i, j));
                values[a] = coefficient * power(u, i) * power(v, j) * power(w, k);
                const T commonU = i > 0 ? static_cast<T>(i) * power(u, i - 1) *
                    power(v, j) * power(w, k) : static_cast<T>(0);
                const T commonV = j > 0 ? static_cast<T>(j) * power(u, i) *
                    power(v, j - 1) * power(w, k) : static_cast<T>(0);
                const T wterm = k > 0 ? static_cast<T>(k) * power(u, i) *
                    power(v, j) * power(w, k - 1) : static_cast<T>(0);
                derivatives[a] = coefficient * (commonU - wterm);
                derivatives[a + (p + 1) * (p + 2) / 2] = coefficient * (commonV - wterm);
                const int npf = (p + 1) * (p + 2) / 2;
                const T uu = (i > 1 ? static_cast<T>(i*(i-1))*power(u,i-2)*power(v,j)*power(w,k) : 0) -
                    (i > 0 && k > 0 ? static_cast<T>(2*i*k)*power(u,i-1)*power(v,j)*power(w,k-1) : 0) +
                    (k > 1 ? static_cast<T>(k*(k-1))*power(u,i)*power(v,j)*power(w,k-2) : 0);
                const T vv = (j > 1 ? static_cast<T>(j*(j-1))*power(u,i)*power(v,j-2)*power(w,k) : 0) -
                    (j > 0 && k > 0 ? static_cast<T>(2*j*k)*power(u,i)*power(v,j-1)*power(w,k-1) : 0) +
                    (k > 1 ? static_cast<T>(k*(k-1))*power(u,i)*power(v,j)*power(w,k-2) : 0);
                const T uv = (i > 0 && j > 0 ? static_cast<T>(i*j)*power(u,i-1)*power(v,j-1)*power(w,k) : 0) -
                    (i > 0 && k > 0 ? static_cast<T>(i*k)*power(u,i-1)*power(v,j)*power(w,k-1) : 0) -
                    (j > 0 && k > 0 ? static_cast<T>(j*k)*power(u,i)*power(v,j-1)*power(w,k-1) : 0) +
                    (k > 1 ? static_cast<T>(k*(k-1))*power(u,i)*power(v,j)*power(w,k-2) : 0);
                second[a] = coefficient*uu;
                second[a+npf] = coefficient*uv;
                second[a+2*npf] = coefficient*uv;
                second[a+3*npf] = coefficient*vv;
                ++a;
            }
        return;
    }
    const T u = xi[0], v = xi[1];
    const int npf = (p + 1) * (p + 1);
    for (int j = 0; j <= p; ++j)
        for (int i = 0; i <= p; ++i) {
            const T bu=bernstein_1d_term(p,i,0,u), bv=bernstein_1d_term(p,j,0,v);
            const T du=bernstein_1d_term(p,i,1,u), dv=bernstein_1d_term(p,j,1,v);
            const T ddu=bernstein_1d_term(p,i,2,u), ddv=bernstein_1d_term(p,j,2,v);
            values[a]=bu*bv; derivatives[a]=du*bv; derivatives[a+npf]=bu*dv;
            second[a]=ddu*bv; second[a+npf]=du*dv;
            second[a+2*npf]=du*dv; second[a+3*npf]=bu*ddv;
            ++a;
        }
}

template <class T>
inline bool invert_matrix(std::vector<T>& inverse, const std::vector<T>& matrix, int n,
                          T& condition)
{
    std::vector<T> work(matrix);
    inverse.assign(static_cast<std::size_t>(n) * n, static_cast<T>(0));
    for (int i = 0; i < n; ++i) inverse[static_cast<std::size_t>(i) * n + i] = 1;
    T norm = 0;
    for (int i = 0; i < n; ++i) {
        T row = 0;
        for (int j = 0; j < n; ++j) row += std::abs(matrix[static_cast<std::size_t>(i) * n + j]);
        norm = std::max(norm, row);
    }
    for (int column = 0; column < n; ++column) {
        int pivot = column;
        T pivotValue = std::abs(work[static_cast<std::size_t>(column) * n + column]);
        for (int row = column + 1; row < n; ++row) {
            const T value = std::abs(work[static_cast<std::size_t>(row) * n + column]);
            if (value > pivotValue) { pivot = row; pivotValue = value; }
        }
        if (!(pivotValue > static_cast<T>(64) * std::numeric_limits<T>::epsilon() *
                           std::max(norm, static_cast<T>(1))))
            return false;
        if (pivot != column)
            for (int j = 0; j < n; ++j) {
                std::swap(work[static_cast<std::size_t>(pivot) * n + j],
                          work[static_cast<std::size_t>(column) * n + j]);
                std::swap(inverse[static_cast<std::size_t>(pivot) * n + j],
                          inverse[static_cast<std::size_t>(column) * n + j]);
            }
        const T diagonal = work[static_cast<std::size_t>(column) * n + column];
        for (int j = 0; j < n; ++j) {
            work[static_cast<std::size_t>(column) * n + j] /= diagonal;
            inverse[static_cast<std::size_t>(column) * n + j] /= diagonal;
        }
        for (int row = 0; row < n; ++row) {
            if (row == column) continue;
            const T factor = work[static_cast<std::size_t>(row) * n + column];
            for (int j = 0; j < n; ++j) {
                work[static_cast<std::size_t>(row) * n + j] -=
                    factor * work[static_cast<std::size_t>(column) * n + j];
                inverse[static_cast<std::size_t>(row) * n + j] -=
                    factor * inverse[static_cast<std::size_t>(column) * n + j];
            }
        }
    }
    T inverseNorm = 0;
    for (int i = 0; i < n; ++i) {
        T row = 0;
        for (int j = 0; j < n; ++j) row += std::abs(inverse[static_cast<std::size_t>(i) * n + j]);
        inverseNorm = std::max(inverseNorm, row);
    }
    condition = norm * inverseNorm;
    return condition < static_cast<T>(1) /
        (static_cast<T>(4096) * std::numeric_limits<T>::epsilon());
}

} // namespace preprocess
} // namespace nonmatching
} // namespace exasim

#endif
