#ifndef __NONMATCHINGPREPROCESSGEOMETRY_HPP
#define __NONMATCHINGPREPROCESSGEOMETRY_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "nonmatchingsimplexbasis.hpp"

namespace exasim {
namespace nonmatching {
namespace preprocess {

// CPU preprocessing is limited to p <= 4, hence at most 25 face nodes and
// six fixed starts (center, closest sample, and one start per topological edge).
constexpr int MaxFaceNodes = 25;
constexpr int MaxFaceDimension = 2;
constexpr int MaxPhysicalDimension = 3;
constexpr int MaxStarts = 6;

template <class T>
inline T down(T value)
{
    return std::nextafter(value, std::numeric_limits<T>::lowest());
}

template <class T>
inline T up(T value)
{
    return std::nextafter(value, std::numeric_limits<T>::max());
}

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

template <class T, class I>
struct PreparedFaceMaster {
    I nd = 0, ndf = 0, npf = 0, faceType = 0, porder = 0, sampleResolution = 0;
    std::vector<T> xpf;             // [npf x ndf]
    std::vector<T> sampleXi;        // [ns x ndf]
    std::vector<T> samplePsi;       // [npf x ns]
    std::vector<T> sampleDpsi;      // [npf x ndf x ns]
    std::vector<I> edgePtr;         // [nedge+1]
    std::vector<I> edgeSample;      // sample indices in fixed edge order
    std::vector<T> centerXi;        // [ndf]
    std::vector<T> bernsteinInverse;// row-major [npf x npf]
    std::unique_ptr<SurfaceNodalBasis> surfaceBasis;
    T bernsteinCondition = 0;

    I samples() const { return static_cast<I>(sampleXi.size() / static_cast<std::size_t>(ndf)); }
    I edges() const { return static_cast<I>(edgePtr.size() > 0 ? edgePtr.size() - 1 : 0); }
};

template <class T, class I>
inline void evaluate_basis(const PreparedFaceMaster<T, I>& master, const T* xi,
                           T* psi, T* dpsi)
{
    if (master.nd == 3) {
        master.surfaceBasis->evaluate(psi, dpsi, xi);
        return;
    }
    for (I j = 0; j < master.npf; ++j) {
        T value = 1, derivative = 0;
        for (I k = 0; k < master.npf; ++k) {
            if (k == j) continue;
            const T denominator = master.xpf[j] - master.xpf[k];
            value *= (xi[0] - master.xpf[k]) / denominator;
            T term = 1 / denominator;
            for (I m = 0; m < master.npf; ++m)
                if (m != j && m != k)
                    term *= (xi[0] - master.xpf[m]) / (master.xpf[j] - master.xpf[m]);
            derivative += term;
        }
        psi[j] = value;
        if (dpsi) dpsi[j] = derivative;
    }
}

template <class T, class I>
inline PreparedFaceMaster<T, I> prepare_face_master(const T* xpf, I nd, I npf,
                                                     I faceType, I porder)
{
    if ((nd != 2 && nd != 3) || porder < 1 || porder > 4 || npf < 2 || npf > MaxFaceNodes)
        throw std::runtime_error("non-matching preprocessing: unsupported face master");
    PreparedFaceMaster<T, I> master;
    master.nd = nd; master.ndf = nd - 1; master.npf = npf;
    master.faceType = faceType; master.porder = porder;
    master.sampleResolution = 2 * porder;
    master.xpf.assign(xpf, xpf + static_cast<std::size_t>(npf) * master.ndf);
    if (nd == 3)
        master.surfaceBasis = std::make_unique<SurfaceNodalBasis>(xpf, npf, porder, faceType);

    const I L = master.sampleResolution;
    if (master.ndf == 1) {
        for (I i = 0; i <= L; ++i) master.sampleXi.push_back(static_cast<T>(i) / L);
        master.centerXi = {static_cast<T>(0.5)};
        master.edgePtr = {0, 1, 2};
        master.edgeSample = {0, L};
    }
    else if (faceType == 0) {
        for (I i = 0; i <= L; ++i)
            for (I j = 0; j <= L - i; ++j) {
                master.sampleXi.push_back(static_cast<T>(i) / L);
                master.sampleXi.push_back(static_cast<T>(j) / L);
            }
        master.centerXi = {static_cast<T>(1.0 / 3.0), static_cast<T>(1.0 / 3.0)};
        master.edgePtr.assign(4, 0);
        for (I edge = 0; edge < 3; ++edge) {
            for (I s = 0; s < master.samples(); ++s) {
                const T u = master.sampleXi[2*s], v = master.sampleXi[2*s+1];
                if ((edge == 0 && v == 0) || (edge == 1 && u + v == 1) ||
                    (edge == 2 && u == 0)) master.edgeSample.push_back(s);
            }
            master.edgePtr[edge + 1] = static_cast<I>(master.edgeSample.size());
        }
    }
    else {
        for (I j = 0; j <= L; ++j)
            for (I i = 0; i <= L; ++i) {
                master.sampleXi.push_back(static_cast<T>(i) / L);
                master.sampleXi.push_back(static_cast<T>(j) / L);
            }
        master.centerXi = {static_cast<T>(0.5), static_cast<T>(0.5)};
        master.edgePtr.assign(5, 0);
        for (I edge = 0; edge < 4; ++edge) {
            for (I s = 0; s < master.samples(); ++s) {
                const T u = master.sampleXi[2*s], v = master.sampleXi[2*s+1];
                if ((edge == 0 && v == 0) || (edge == 1 && u == 1) ||
                    (edge == 2 && v == 1) || (edge == 3 && u == 0))
                    master.edgeSample.push_back(s);
            }
            master.edgePtr[edge + 1] = static_cast<I>(master.edgeSample.size());
        }
    }

    const I ns = master.samples();
    master.samplePsi.resize(static_cast<std::size_t>(npf) * ns);
    master.sampleDpsi.resize(static_cast<std::size_t>(npf) * master.ndf * ns);
    for (I s = 0; s < ns; ++s)
        evaluate_basis(master, &master.sampleXi[static_cast<std::size_t>(s) * master.ndf],
                       &master.samplePsi[static_cast<std::size_t>(s) * npf],
                       &master.sampleDpsi[static_cast<std::size_t>(s) * npf * master.ndf]);

    std::vector<T> collocation(static_cast<std::size_t>(npf) * npf);
    T values[MaxFaceNodes] = {0};
    for (I a = 0; a < npf; ++a) {
        T xi[2] = {xpf[a], master.ndf == 2 ? xpf[a + npf] : static_cast<T>(0)};
        bernstein_values(values, xi, porder, master.ndf, faceType);
        for (I b = 0; b < npf; ++b)
            collocation[static_cast<std::size_t>(a) * npf + b] = values[b];
    }
    if (!invert_matrix(master.bernsteinInverse, collocation, npf, master.bernsteinCondition))
        throw std::runtime_error("non-matching preprocessing: Bernstein collocation matrix is ill-conditioned");
    return master;
}

template <class T, class I>
struct PreparedFaces {
    PreparedFaceMaster<T, I> master;
    I count = 0;
    std::vector<I> rank, elem, localFace, globalFace;
    std::vector<T> nodes;            // [npf x nd x count]
    std::vector<T> control;          // Bernstein coefficients [npf x nd x count]
    std::vector<T> physicalSamples;  // [nd x ns x count]
    std::vector<T> centerPhysical;   // [nd x count]
    std::vector<T> boxMin, boxMax;   // [nd x count]
    std::vector<T> cmax, beta, hface;// [count]
};

template <class T, class I, class FaceSet>
inline PreparedFaces<T, I> prepare_faces(const FaceSet& faces, const T* xpf,
                                         I faceType, I porder)
{
    PreparedFaces<T, I> out;
    out.master = prepare_face_master<T, I>(xpf, faces.nd, faces.npf, faceType, porder);
    out.count = faces.size(); out.rank = faces.rank; out.elem = faces.elem;
    out.localFace = faces.lface; out.globalFace = faces.global; out.nodes = faces.nodes;
    const I nd = out.master.nd, npf = out.master.npf, ns = out.master.samples();
    out.physicalSamples.assign(static_cast<std::size_t>(nd) * ns * out.count, 0);
    out.centerPhysical.assign(static_cast<std::size_t>(nd) * out.count, 0);
    out.control.assign(static_cast<std::size_t>(npf) * nd * out.count, 0);
    out.boxMin.assign(static_cast<std::size_t>(nd) * out.count, 0);
    out.boxMax.assign(static_cast<std::size_t>(nd) * out.count, 0);
    out.cmax.assign(out.count, 0); out.beta.assign(out.count, 0); out.hface.assign(out.count, 0);
    for (I face = 0; face < out.count; ++face) {
        const T* y = &out.nodes[static_cast<std::size_t>(npf) * nd * face];
        for (I s = 0; s < ns; ++s)
            for (I d = 0; d < nd; ++d)
                for (I a = 0; a < npf; ++a)
                    out.physicalSamples[d + nd * (s + static_cast<std::size_t>(ns) * face)] +=
                        out.master.samplePsi[a + npf*s] * y[a + npf*d];

        T* control = &out.control[static_cast<std::size_t>(npf) * nd * face];
        T scale = 0;
        for (I d = 0; d < nd; ++d)
            for (I a = 0; a < npf; ++a) {
                scale = std::max(scale, std::abs(y[a + npf*d]));
                for (I b = 0; b < npf; ++b)
                    control[b + npf*d] += out.master.bernsteinInverse[static_cast<std::size_t>(b)*npf+a] *
                                           y[a + npf*d];
            }
        long double centerValues[MaxFaceNodes] = {0};
        long double centerDerivatives[MaxFaceNodes * MaxFaceDimension] = {0};
        long double centerSecond[MaxFaceNodes * MaxFaceDimension * MaxFaceDimension] = {0};
        const long double centerXi[2] = {
            static_cast<long double>(out.master.centerXi[0]),
            out.master.ndf == 2 ? static_cast<long double>(out.master.centerXi[1]) : 0};
        bernstein_values_derivatives(centerValues, centerDerivatives, centerSecond,
            centerXi, porder, out.master.ndf, faceType);
        long double center[MaxPhysicalDimension] = {0, 0, 0};
        for (I a = 0; a < npf; ++a)
            for (I d = 0; d < nd; ++d)
                center[d] += centerValues[a] * control[a + npf*d];
        for (I d = 0; d < nd; ++d)
            out.centerPhysical[d + static_cast<std::size_t>(nd) * face] =
                static_cast<T>(center[d]);
        const T beta = up(static_cast<T>(32*npf) * std::numeric_limits<T>::epsilon() *
                          std::max(static_cast<T>(1), out.master.bernsteinCondition) *
                          std::max(static_cast<T>(1), scale));
        out.beta[face] = beta;
        T diagonal2 = 0;
        for (I d = 0; d < nd; ++d) {
            T lo = std::numeric_limits<T>::max(), hi = std::numeric_limits<T>::lowest();
            for (I a = 0; a < npf; ++a) {
                const T c = control[a + npf*d];
                out.cmax[face] = std::max(out.cmax[face], std::abs(c));
                lo = std::min(lo, down(c - beta)); hi = std::max(hi, up(c + beta));
            }
            out.boxMin[d + nd*face] = lo; out.boxMax[d + nd*face] = hi;
            diagonal2 += (hi-lo)*(hi-lo);
        }
        out.hface[face] = std::sqrt(diagonal2);
    }
    return out;
}

template <class T, class I>
struct ProjectionWorkspace {
    long double psi[MaxFaceNodes] = {0};
    long double dpsi[MaxFaceNodes * MaxFaceDimension] = {0};
    long double ddpsi[MaxFaceNodes * MaxFaceDimension * MaxFaceDimension] = {0};
    long double physical[MaxPhysicalDimension] = {0};
    long double tangent[MaxPhysicalDimension * MaxFaceDimension] = {0};
    long double curvature[MaxPhysicalDimension * MaxFaceDimension * MaxFaceDimension] = {0};
    T starts[MaxStarts * MaxFaceDimension] = {0};
    T lastProjectedStep = 0;
    I lastIteration = 0;
    long long projectionStarts = 0;
    long long projectionIterations = 0;
    std::vector<I> faceOrder;
    std::vector<T> faceLower;
};

template <class T>
inline bool solve_small(T* x, const T* matrix, const T* rhs, int n)
{
    T scale = 0;
    for (int i = 0; i < n*n; ++i) scale = std::max(scale, std::abs(matrix[i]));
    if (!(scale > 0)) return false;
    if (n == 1) { x[0] = rhs[0] / matrix[0]; return true; }
    const T a = matrix[0]/scale, b = matrix[1]/scale;
    const T c = matrix[2]/scale, d = matrix[3]/scale;
    const T determinant = a*d-b*c;
    if (!(std::abs(determinant) > static_cast<T>(64)*std::numeric_limits<T>::epsilon())) return false;
    const T r0 = rhs[0]/scale, r1 = rhs[1]/scale;
    x[0] = (d*r0-b*r1)/determinant; x[1] = (-c*r0+a*r1)/determinant;
    return true;
}

template <class T>
inline bool inside_face(const T* xi, int ndf, int faceType)
{
    if (ndf == 1) return xi[0] >= 0 && xi[0] <= 1;
    if (faceType == 0) return xi[0] >= 0 && xi[1] >= 0 && xi[0]+xi[1] <= 1;
    return xi[0] >= 0 && xi[0] <= 1 && xi[1] >= 0 && xi[1] <= 1;
}

template <class T>
inline bool project_metric(T *projected, const T *point, const T *metric, int ndf, int faceType) {
    if (ndf == 1) {
        projected[0] = std::max<T>(0, std::min<T>(1, point[0]));
        return metric[0] > 0;
    }
    if (inside_face(point, ndf, faceType)) {
        projected[0] = point[0];
        projected[1] = point[1];
        return true;
    }
    const T tri[6] = {0, 0, 1, 0, 0, 1}, quad[8] = {0, 0, 1, 0, 1, 1, 0, 1};
    const T *vertices = faceType == 0 ? tri : quad;
    const int nv = faceType == 0 ? 3 : 4;
    T best = std::numeric_limits<T>::max();
    bool found = false;
    for (int edge = 0; edge < nv; ++edge) {
        const int next = (edge + 1) % nv;
        const T v0[2] = {vertices[2 * edge], vertices[2 * edge + 1]};
        const T e[2] = {vertices[2 * next] - v0[0], vertices[2 * next + 1] - v0[1]};
        const T r[2] = {v0[0] - point[0], v0[1] - point[1]};
        const T he[2] = {metric[0] * e[0] + metric[1] * e[1], metric[2] * e[0] + metric[3] * e[1]};
        const T denom = e[0] * he[0] + e[1] * he[1];
        if (!(denom > 0))
            continue;
        T t = -(r[0] * he[0] + r[1] * he[1]) / denom;
        t = std::max<T>(0, std::min<T>(1, t));
        const T z[2] = {v0[0] + t * e[0], v0[1] + t * e[1]},
                q[2] = {z[0] - point[0], z[1] - point[1]};
        const T value = q[0] * (metric[0] * q[0] + metric[1] * q[1]) +
                        q[1] * (metric[2] * q[0] + metric[3] * q[1]);
        if (value < best) {
            best = value;
            projected[0] = z[0];
            projected[1] = z[1];
            found = true;
        }
    }
    return found;
}

template <class T, class I>
inline T map_face(const PreparedFaceMaster<T, I> &master, const T *control, const T *target,
                  const T *xi, ProjectionWorkspace<T, I> &work) {
    const long double xiWide[2] = {static_cast<long double>(xi[0]),
                                   master.ndf == 2 ? static_cast<long double>(xi[1]) : 0};
    bernstein_values_derivatives(work.psi, work.dpsi, work.ddpsi, xiWide, master.porder, master.ndf,
                                 master.faceType);
    std::fill(work.physical, work.physical + master.nd, static_cast<long double>(0));
    std::fill(work.tangent, work.tangent + master.nd * master.ndf, static_cast<long double>(0));
    std::fill(work.curvature, work.curvature + master.nd * master.ndf * master.ndf,
              static_cast<long double>(0));
    for (I a = 0; a < master.npf; ++a)
        for (I d = 0; d < master.nd; ++d) {
            const long double y = control[a + master.npf * d];
            work.physical[d] += work.psi[a] * y;
            for (I k = 0; k < master.ndf; ++k)
                work.tangent[d * master.ndf + k] += work.dpsi[a + master.npf * k] * y;
            for (I k = 0; k < master.ndf; ++k)
                for (I j = 0; j < master.ndf; ++j)
                    work.curvature[j + master.ndf * (k + master.ndf * d)] +=
                        work.ddpsi[a + master.npf * (j + master.ndf * k)] * y;
        }
    long double value = 0;
    for (I d = 0; d < master.nd; ++d) {
        const long double r = work.physical[d] - target[d];
        value += r * r;
    }
    return static_cast<T>(value);
}

template <class T, class I>
inline bool project_from_start(const PreparedFaceMaster<T, I> &master, const T *control,
                               const T *target, T controlScale, const T *start, T tolerance,
                               I maxIterations, ProjectionWorkspace<T, I> &work, T *xi,
                               T &distance) {
    ++work.projectionStarts;
    for (I k = 0; k < master.ndf; ++k)
        xi[k] = start[k];
    for (I iteration = 0; iteration < maxIterations; ++iteration) {
        ++work.projectionIterations;
        work.lastIteration = iteration;
        const T current = map_face(master, control, target, xi, work);
        long double metricWide[4] = {0, 0, 0, 0}, rhsWide[2] = {0, 0}, deltaWide[2] = {0, 0};
        T metric[4] = {0, 0, 0, 0}, delta[2] = {0, 0}, trial[2] = {0, 0}, projected[2] = {0, 0};
        for (I i = 0; i < master.ndf; ++i)
            for (I d = 0; d < master.nd; ++d) {
                const long double tangent = work.tangent[d * master.ndf + i];
                rhsWide[i] -= tangent * (work.physical[d] - target[d]);
                for (I j = 0; j < master.ndf; ++j)
                    metricWide[i * master.ndf + j] +=
                        tangent * static_cast<long double>(work.tangent[d * master.ndf + j]);
            }
        if (!solve_small(deltaWide, metricWide, rhsWide, master.ndf))
            return false;
        for (I i = 0; i < master.ndf; ++i) {
            delta[i] = static_cast<T>(deltaWide[i]);
            for (I j = 0; j < master.ndf; ++j)
                metric[i * master.ndf + j] = static_cast<T>(metricWide[i * master.ndf + j]);
        }
        for (I k = 0; k < master.ndf; ++k)
            trial[k] = xi[k] + delta[k];
        if (!project_metric(projected, trial, metric, master.ndf, master.faceType))
            return false;
        T step2 = 0;
        for (I k = 0; k < master.ndf; ++k) {
            const T q = projected[k] - xi[k];
            step2 += q * q;
        }
        work.lastProjectedStep = std::sqrt(step2);
        long double targetScale = 0;
        for (I d = 0; d < master.nd; ++d)
            targetScale = std::max(targetScale, std::abs(static_cast<long double>(target[d])));
        const long double coordinateScale =
            std::max(targetScale, static_cast<long double>(controlScale));
        long double lambdaMin = metricWide[0];
        if (master.ndf == 2) {
            const long double a = metricWide[0], b = metricWide[1], c = metricWide[3];
            const long double discriminant = std::hypot(a - c, 2 * b);
            const long double lambdaMax = (a + c + discriminant) / 2;
            const long double determinant = a * c - b * b;
            lambdaMin = lambdaMax > 0 ? determinant / lambdaMax : 0;
        }
        // Coordinate round-off cannot resolve a smaller reference-space step.
        const T floorStep =
            lambdaMin > 0 ? static_cast<T>(16 * std::numeric_limits<T>::epsilon() *
                                           coordinateScale / std::sqrt(lambdaMin))
                          : static_cast<T>(0);
        if (work.lastProjectedStep <= std::max(tolerance, floorStep)) {
            distance = std::sqrt(current);
            return true;
        }
        bool accepted = false;
        T alpha = 1;
        for (int line = 0; line < 60; ++line, alpha *= static_cast<T>(0.5)) {
            for (I k = 0; k < master.ndf; ++k)
                trial[k] = xi[k] + alpha * delta[k];
            if (!project_metric(projected, trial, metric, master.ndf, master.faceType))
                break;
            const T next = map_face(master, control, target, projected, work);
            if (next < current) {
                for (I k = 0; k < master.ndf; ++k)
                    xi[k] = projected[k];
                accepted = true;
                break;
            }
        }
        if (!accepted) {
            // At nonzero normal distance Gauss--Newton can reach the objective's
            // roundoff floor before its reference step reaches tolerance. Use
            // one exact-Hessian stationarity polish, then return to the same
            // projected Gauss--Newton stopping test on the next iteration.
            map_face(master, control, target, xi, work);
            long double hessian[4] = {0, 0, 0, 0}, newtonRhs[2] = {0, 0}, newtonStep[2] = {0, 0};
            for (I i = 0; i < master.ndf; ++i)
                for (I j = 0; j < master.ndf; ++j) {
                    hessian[i * master.ndf + j] = metricWide[i * master.ndf + j];
                    for (I d = 0; d < master.nd; ++d)
                        hessian[i * master.ndf + j] +=
                            (work.physical[d] - target[d]) *
                            work.curvature[j + master.ndf * (i + master.ndf * d)];
                }
            for (I i = 0; i < master.ndf; ++i)
                newtonRhs[i] = rhsWide[i];
            if (!solve_small(newtonStep, hessian, newtonRhs, master.ndf))
                return false;
            for (I k = 0; k < master.ndf; ++k)
                trial[k] = xi[k] + static_cast<T>(newtonStep[k]);
            if (!project_metric(projected, trial, metric, master.ndf, master.faceType))
                return false;
            const T polished = map_face(master, control, target, projected, work);
            const long double coordinateRoundoff =
                16 * std::numeric_limits<T>::epsilon() * coordinateScale;
            const long double distanceWide = std::sqrt(static_cast<long double>(current));
            const T roundoff = static_cast<T>(
                coordinateRoundoff * (2 * distanceWide + coordinateRoundoff));
            if (polished > current + roundoff)
                return false;
            for (I k = 0; k < master.ndf; ++k)
                xi[k] = projected[k];
        }
    }
    distance = std::sqrt(map_face(master, control, target, xi, work));
    return false;
}

template <class T, class I> struct FaceProjection {
    T xi[2] = {0, 0};
    T distance = std::numeric_limits<T>::max();
    T centerDistance = std::numeric_limits<T>::max();
    I start = -1;
    bool converged = false;
};

template <class T> inline T tie_scale(T cmaxA, T cmaxB) { return std::max(cmaxA, cmaxB); }

template <class T> inline T tie_window(T distance, T cmaxA, T cmaxB) {
    return static_cast<T>(128) * std::numeric_limits<T>::epsilon() *
           std::max({static_cast<T>(1), std::abs(distance), tie_scale(cmaxA, cmaxB)});
}

template <class T> inline bool tied(T a, T b, T cmaxA, T cmaxB) {
    return std::abs(a - b) <=
           static_cast<T>(128) * std::numeric_limits<T>::epsilon() *
               std::max({static_cast<T>(1), std::abs(a), std::abs(b), tie_scale(cmaxA, cmaxB)});
}

template <class T, class I>
inline FaceProjection<T, I> project_face_multistart(const PreparedFaces<T, I> &faces, I face,
                                                    const T *target, T tolerance, I maxIterations,
                                                    ProjectionWorkspace<T, I> &work) {
    const auto &master = faces.master;
    const I ns = master.samples();
    const T *control = &faces.control[static_cast<std::size_t>(master.npf) * master.nd * face];
    I nstart = 0;
    auto add = [&](const T *z) {
        for (I s = 0; s < nstart; ++s) {
            T q = 0;
            for (I k = 0; k < master.ndf; ++k) {
                const T d = work.starts[s * master.ndf + k] - z[k];
                q += d * d;
            }
            if (std::sqrt(q) <= static_cast<T>(4) * tolerance)
                return;
        }
        for (I k = 0; k < master.ndf; ++k)
            work.starts[nstart * master.ndf + k] = z[k];
        ++nstart;
    };
    add(master.centerXi.data());
    I closest = 0;
    T best = std::numeric_limits<T>::max();
    for (I s = 0; s < ns; ++s) {
        T q = 0;
        for (I d = 0; d < master.nd; ++d) {
            const T r =
                faces.physicalSamples[d + master.nd * (s + static_cast<std::size_t>(ns) * face)] -
                target[d];
            q += r * r;
        }
        if (q < best) {
            best = q;
            closest = s;
        }
    }
    add(&master.sampleXi[static_cast<std::size_t>(closest) * master.ndf]);
    for (I edge = 0; edge < master.edges(); ++edge) {
        I edgeBest = master.edgeSample[master.edgePtr[edge]];
        T edgeValue = std::numeric_limits<T>::max();
        for (I k = master.edgePtr[edge]; k < master.edgePtr[edge + 1]; ++k) {
            const I s = master.edgeSample[k];
            T q = 0;
            for (I d = 0; d < master.nd; ++d) {
                const T r =
                    faces.physicalSamples[d +
                                          master.nd * (s + static_cast<std::size_t>(ns) * face)] -
                    target[d];
                q += r * r;
            }
            if (q < edgeValue) {
                edgeValue = q;
                edgeBest = s;
            }
        }
        add(&master.sampleXi[static_cast<std::size_t>(edgeBest) * master.ndf]);
    }
    FaceProjection<T, I> winner;
    const T *centerPhysical = &faces.centerPhysical[static_cast<std::size_t>(master.nd) * face];
    for (I s = 0; s < nstart; ++s) {
        T xi[2] = {0, 0}, distance = 0;
        const bool ok = project_from_start(master, control, target, faces.cmax[face],
                                           &work.starts[s * master.ndf], tolerance, maxIterations,
                                           work, xi, distance);
        if (!ok)
            continue;
        map_face(master, control, centerPhysical, xi, work);
        T cd = 0;
        for (I d = 0; d < master.nd; ++d) {
            const long double delta = work.physical[d] - centerPhysical[d];
            cd += static_cast<T>(delta * delta);
        }
        cd = std::sqrt(cd);
        bool take = !winner.converged || distance < winner.distance;
        if (winner.converged &&
            tied(distance, winner.distance, faces.cmax[face], faces.cmax[face])) {
            take = cd < winner.centerDistance;
            if (tied(cd, winner.centerDistance, faces.cmax[face], faces.cmax[face])) {
                take = std::lexicographical_compare(xi, xi + master.ndf, winner.xi,
                                                    winner.xi + master.ndf) ||
                       (std::equal(xi, xi + master.ndf, winner.xi) && s < winner.start);
            }
        }
        if (take) {
            winner.converged = true;
            winner.distance = distance;
            winner.centerDistance = cd;
            winner.start = s;
            winner.xi[0] = xi[0];
            winner.xi[1] = xi[1];
        }
    }
    return winner;
}

template <class T, class I>
inline T box_distance(const PreparedFaces<T, I> &faces, I face, const T *x) {
    T value = 0;
    for (I d = 0; d < faces.master.nd; ++d) {
        T gap = 0;
        if (x[d] < faces.boxMin[d + faces.master.nd * face])
            gap = faces.boxMin[d + faces.master.nd * face] - x[d];
        else if (x[d] > faces.boxMax[d + faces.master.nd * face])
            gap = x[d] - faces.boxMax[d + faces.master.nd * face];
        value += gap * gap;
    }
    return down(std::sqrt(value));
}

template <class T> inline T screen_margin(T cmax, T beta, T xabs, int nd, int porder) {
    const T eps = std::numeric_limits<T>::epsilon();
    const T coefficient = nd == 2 ? static_cast<T>(8 * (porder + 1)) : static_cast<T>(4096);
    const T scale = nd == 2 ? cmax + xabs : std::max<T>(1, cmax + xabs);
    return up(coefficient * eps * scale + static_cast<T>(4) * beta);
}

template <class T, class I>
inline T face_margin(const PreparedFaces<T, I> &faces, I face, const T *x) {
    T xabs = 0;
    for (I d = 0; d < faces.master.nd; ++d)
        xabs = std::max(xabs, std::abs(x[d]));
    return screen_margin(faces.cmax[face], faces.beta[face], xabs, faces.master.nd,
                         faces.master.porder);
}

template <class T, class I> struct LocalSearchResult {
    I face = -1;
    I failedFace = -1;
    FaceProjection<T, I> projection;
    T feasibleUpper = std::numeric_limits<T>::max();
    I projectedFaces = 0, prunedFaces = 0;
    long long projectionStarts = 0, projectionIterations = 0;
    bool failed = false;
};

template <class T, class I>
inline LocalSearchResult<T, I> search_local_faces(const PreparedFaces<T, I> &faces, const T *x,
                                                  T tolerance, I maxIterations,
                                                  ProjectionWorkspace<T, I> &work) {
    LocalSearchResult<T, I> result;
    if (faces.count == 0) {
        result.failed = true;
        return result;
    }
    const long long startsBefore = work.projectionStarts;
    const long long iterationsBefore = work.projectionIterations;
    work.faceOrder.resize(static_cast<std::size_t>(faces.count));
    std::iota(work.faceOrder.begin(), work.faceOrder.end(), 0);
    work.faceLower.resize(static_cast<std::size_t>(faces.count));
    for (I m = 0; m < faces.count; ++m)
        work.faceLower[m] = box_distance(faces, m, x) - face_margin(faces, m, x);
    std::sort(work.faceOrder.begin(), work.faceOrder.end(), [&](I a, I b) {
        return std::tie(work.faceLower[a], faces.globalFace[a]) <
               std::tie(work.faceLower[b], faces.globalFace[b]);
    });
    for (I m : work.faceOrder) {
        if (result.face >= 0) {
            const T window =
                tie_window(result.feasibleUpper, faces.cmax[m], faces.cmax[result.face]);
            if (work.faceLower[m] > up(result.feasibleUpper + window)) {
                ++result.prunedFaces;
                continue;
            }
        }
        const auto candidate = project_face_multistart(faces, m, x, tolerance, maxIterations, work);
        ++result.projectedFaces;
        if (!candidate.converged) {
            result.failed = true;
            if (result.failedFace < 0 || faces.globalFace[m] < faces.globalFace[result.failedFace])
                result.failedFace = m;
            continue;
        }
        result.feasibleUpper = std::min(result.feasibleUpper, up(candidate.distance));
        const bool tie = result.face >= 0 && tied(candidate.distance, result.projection.distance,
                                                  faces.cmax[m], faces.cmax[result.face]);
        if (result.face < 0 || (!tie && candidate.distance < result.projection.distance) ||
            (tie && faces.globalFace[m] < faces.globalFace[result.face])) {
            result.face = m;
            result.projection = candidate;
        }
    }
    result.projectionStarts = work.projectionStarts - startsBefore;
    result.projectionIterations = work.projectionIterations - iterationsBefore;
    return result;
}

} // namespace preprocess
} // namespace nonmatching
} // namespace exasim

#endif
