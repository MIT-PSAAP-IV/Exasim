#include "dgprojection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// volgeom_det: determinant of the nd x nd Jacobian, J[a*nd + b] = dx_b/dxi_a.
// Matches frontends/Matlab/Utilities/volgeom.m (the `jac` output) for nd 1/2/3.
// ---------------------------------------------------------------------------
double volgeom_det(const double* J, int nd)
{
    switch (nd) {
        case 1:
            return J[0];
        case 2:
            // Jg11*Jg22 - Jg12*Jg21
            return J[0] * J[3] - J[1] * J[2];
        case 3:
            // standard 3x3 determinant; algebraically identical to volgeom.m
            return J[0] * (J[4] * J[8] - J[5] * J[7])
                 - J[1] * (J[3] * J[8] - J[5] * J[6])
                 + J[2] * (J[3] * J[7] - J[4] * J[6]);
        default:
            throw std::runtime_error("volgeom_det: dimension not implemented (nd must be 1, 2, or 3)");
    }
}

namespace {

// Solve A X = B for X, where A is n x n and B is n x m, both column-major.
// A and B are overwritten (A with its LU factors, B with the solution X).
// Gaussian elimination with partial pivoting. Throws if A is singular.
void solveDense(double* A, double* B, int n, int m)
{
    const double tiny = 0.0;
    for (int k = 0; k < n; ++k) {
        // find pivot in column k
        int piv = k;
        double amax = std::fabs(A[k + n * k]);
        for (int i = k + 1; i < n; ++i) {
            double a = std::fabs(A[i + n * k]);
            if (a > amax) { amax = a; piv = i; }
        }
        if (amax <= tiny) {
            throw std::runtime_error("dgprojection: singular element mass matrix");
        }
        // swap rows k and piv in A and B
        if (piv != k) {
            for (int j = 0; j < n; ++j) std::swap(A[k + n * j], A[piv + n * j]);
            for (int j = 0; j < m; ++j) std::swap(B[k + n * j], B[piv + n * j]);
        }
        // eliminate below
        double akk = A[k + n * k];
        for (int i = k + 1; i < n; ++i) {
            double f = A[i + n * k] / akk;
            if (f != 0.0) {
                for (int j = k + 1; j < n; ++j) A[i + n * j] -= f * A[k + n * j];
                for (int j = 0; j < m; ++j)     B[i + n * j] -= f * B[k + n * j];
            }
        }
    }
    // back substitution
    for (int col = 0; col < m; ++col) {
        for (int i = n - 1; i >= 0; --i) {
            double s = B[i + n * col];
            for (int j = i + 1; j < n; ++j) s -= A[i + n * j] * B[j + n * col];
            B[i + n * col] = s / A[i + n * i];
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// dgprojection: element-by-element L2 projection between nodal bases.
// See dgprojection.hpp for the array layouts. Port of dgprojection.m.
// ---------------------------------------------------------------------------
void dgprojection(double* U1,
                  const double* U,
                  const double* dgnodes,
                  const double* shape_t,
                  const double* dshape_t,
                  const double* shape_s,
                  const double* gw,
                  int npe_t,
                  int npe_s,
                  int nge,
                  int nd,
                  int ncx,
                  int nc,
                  int ne)
{
    if (nd < 1 || nd > 3) {
        throw std::runtime_error("dgprojection: dimension not implemented (nd must be 1, 2, or 3)");
    }

    std::vector<double> jac(nge);           // det J at each Gauss point
    std::vector<double> M(npe_t * npe_t);   // element mass matrix (target x target)
    std::vector<double> C(npe_t * npe_s);   // cross-mass matrix (target x source)
    std::vector<double> L(npe_t * nc);      // C * U(:,:,e), also holds the solve output
    std::vector<double> J(nd * nd);         // Jacobian at one Gauss point

    for (int e = 0; e < ne; ++e) {
        const double* Xe = dgnodes + static_cast<std::size_t>(npe_t) * ncx * e;
        const double* Ue = U       + static_cast<std::size_t>(npe_s) * nc  * e;

        // --- geometry: jac at each Gauss point (dgprojection.m lines 22-24) ---
        for (int g = 0; g < nge; ++g) {
            for (int a = 0; a < nd; ++a) {        // xi-derivative direction
                for (int b = 0; b < nd; ++b) {    // physical coordinate
                    double s = 0.0;
                    for (int i = 0; i < npe_t; ++i) {
                        // dshape_t[i,g,a] * dgnodes[i,b,e]
                        s += dshape_t[i + npe_t * (g + nge * a)] * Xe[i + npe_t * b];
                    }
                    J[a * nd + b] = s;            // J[a*nd+b] = dx_b/dxi_a
                }
            }
            jac[g] = volgeom_det(J.data(), nd);
        }

        // --- M = shape_t * diag(gw.*jac) * shape_t^T  (line 26) ---
        for (int q = 0; q < npe_t; ++q) {
            for (int p = 0; p < npe_t; ++p) {
                double s = 0.0;
                for (int g = 0; g < nge; ++g) {
                    s += shape_t[p + npe_t * g] * shape_t[q + npe_t * g] * (gw[g] * jac[g]);
                }
                M[p + npe_t * q] = s;
            }
        }

        // --- C = shape_t * diag(gw.*jac) * shape_s^T  (line 27) ---
        for (int sidx = 0; sidx < npe_s; ++sidx) {
            for (int p = 0; p < npe_t; ++p) {
                double s = 0.0;
                for (int g = 0; g < nge; ++g) {
                    s += shape_t[p + npe_t * g] * shape_s[sidx + npe_s * g] * (gw[g] * jac[g]);
                }
                C[p + npe_t * sidx] = s;
            }
        }

        // --- L = C * U(:,:,e)  (line 28) ---
        for (int c = 0; c < nc; ++c) {
            for (int p = 0; p < npe_t; ++p) {
                double s = 0.0;
                for (int sidx = 0; sidx < npe_s; ++sidx) {
                    s += C[p + npe_t * sidx] * Ue[sidx + npe_s * c];
                }
                L[p + npe_t * c] = s;
            }
        }

        // --- U1(:,:,e) = M \ L  (line 29) ---
        solveDense(M.data(), L.data(), npe_t, nc);

        double* U1e = U1 + static_cast<std::size_t>(npe_t) * nc * e;
        for (int c = 0; c < nc; ++c) {
            for (int p = 0; p < npe_t; ++p) {
                U1e[p + npe_t * c] = L[p + npe_t * c];
            }
        }
    }
}
