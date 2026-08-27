#include "l2eprojection.hpp"
#include "dgprojection.hpp"   // volgeom_det (shared, already unit-tested)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {
// Solve A X = B for X, A n x n and B n x m (both column-major); A/B overwritten.
// Gaussian elimination with partial pivoting. Throws if A is singular.
void solveDense(double* A, double* B, int n, int m)
{
    for (int k = 0; k < n; ++k) {
        int piv = k; double amax = std::fabs(A[k + n * k]);
        for (int i = k + 1; i < n; ++i) { double a = std::fabs(A[i + n * k]); if (a > amax) { amax = a; piv = i; } }
        if (amax == 0.0) throw std::runtime_error("l2eprojection: singular element mass matrix");
        if (piv != k) {
            for (int j = 0; j < n; ++j) std::swap(A[k + n * j], A[piv + n * j]);
            for (int j = 0; j < m; ++j) std::swap(B[k + n * j], B[piv + n * j]);
        }
        double akk = A[k + n * k];
        for (int i = k + 1; i < n; ++i) {
            double f = A[i + n * k] / akk;
            if (f != 0.0) {
                for (int j = k + 1; j < n; ++j) A[i + n * j] -= f * A[k + n * j];
                for (int j = 0; j < m; ++j)     B[i + n * j] -= f * B[k + n * j];
            }
        }
    }
    for (int col = 0; col < m; ++col)
        for (int i = n - 1; i >= 0; --i) {
            double s = B[i + n * col];
            for (int j = i + 1; j < n; ++j) s -= A[i + n * j] * B[j + n * col];
            B[i + n * col] = s / A[i + n * i];
        }
}
} // namespace

void l2eprojection(double* UDG, const double* fg, const double* dgnodes,
                   const double* shapv, const double* dshapv, const double* gw,
                   int npe, int nge, int nd, int ncx, int nc, int ne)
{
    if (nd < 1 || nd > 3) throw std::runtime_error("l2eprojection: nd must be 1, 2, or 3");

    std::vector<double> M(npe * npe), F(npe * nc), J(nd * nd);

    for (int e = 0; e < ne; ++e) {
        const double* Xe  = dgnodes + static_cast<std::size_t>(npe) * ncx * e;
        const double* fge = fg      + static_cast<std::size_t>(nge) * nc  * e;

        std::fill(M.begin(), M.end(), 0.0);
        std::fill(F.begin(), F.end(), 0.0);

        for (int g = 0; g < nge; ++g) {
            // Jacobian J[i*nd+j] = dx_j/dxi_i at Gauss point g
            for (int i = 0; i < nd; ++i)
                for (int j = 0; j < nd; ++j) {
                    double s = 0.0;
                    for (int a = 0; a < npe; ++a)
                        s += dshapv[g + nge * (a + npe * i)] * Xe[a + npe * j];
                    J[i * nd + j] = s;
                }
            const double wj = gw[g] * volgeom_det(J.data(), nd);   // weight * jac

            for (int b = 0; b < npe; ++b) {
                const double sb = shapv[g + nge * b];
                for (int a = 0; a < npe; ++a)
                    M[a + npe * b] += shapv[g + nge * a] * sb * wj;   // M += phi_a phi_b w jac
            }
            for (int c = 0; c < nc; ++c) {
                const double fc = fge[g + nge * c] * wj;
                for (int a = 0; a < npe; ++a)
                    F[a + npe * c] += shapv[g + nge * a] * fc;        // F += phi_a f_c w jac
            }
        }

        solveDense(M.data(), F.data(), npe, nc);   // F <- M^{-1} F

        double* U = UDG + static_cast<std::size_t>(npe) * nc * e;
        for (int c = 0; c < nc; ++c)
            for (int a = 0; a < npe; ++a)
                U[a + npe * c] = F[a + npe * c];
    }
}
