// dgprojection_backend_test.cpp
//
// Validates the *batched-primitive decomposition* used by DGProjection
// (dgprojection_backend.hpp) without needing the Kokkos build. It reimplements
// the backend primitives (Node2Gauss, Gauss2Node, ShapJac, ArrayGemmBatch1,
// ArrayMatrixMultiplication1, Inverse, ArraySetValue) with the EXACT semantics
// and array layouts of the real kernels, replays the same straight-mesh and
// curved-mesh algorithms DGProjection uses, and checks the result against the
// portable scalar reference dgprojection() (the trusted oracle).
//
// The real DGProjection calls the same primitives with the same arguments, so a
// match here pins the decomposition (strides, which shape matrix, accumulate vs
// overwrite, straight/curved split, Jacobian weighting). On-device numerical
// validation of the Kokkos path is a separate ctest (needs a full build).

#include "dgprojection.hpp"   // volgeom_det + dgprojection (oracle)

#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool nearlyEqual(double a, double b, double tol = 1.0e-10)
{
    return std::abs(a - b) <= tol * (1.0 + std::abs(b));
}
void expectClose(double a, double b, const std::string& l)
{
    if (!nearlyEqual(a, b))
        throw std::runtime_error(l + ": expected " + std::to_string(b) + ", got " + std::to_string(a));
}

// ---- faithful host mirrors of the backend primitives -----------------------

// Node2Gauss: ug[ng x nn] = shapt[ng x np] * un[np x nn]   (all column-major)
void h_Node2Gauss(double* ug, const double* un, const double* shapt, int ng, int np, int nn)
{
    for (int j = 0; j < nn; ++j)
        for (int i = 0; i < ng; ++i) {
            double s = 0.0;
            for (int k = 0; k < np; ++k) s += shapt[i + ng * k] * un[k + np * j];
            ug[i + ng * j] = s;
        }
}
// Gauss2Node: un[np x nn] = shapg[np x ng] * ug[ng x nn]
void h_Gauss2Node(double* un, const double* ug, const double* shapg, int ng, int np, int nn)
{
    for (int j = 0; j < nn; ++j)
        for (int i = 0; i < np; ++i) {
            double s = 0.0;
            for (int k = 0; k < ng; ++k) s += shapg[i + np * k] * ug[k + ng * j];
            un[i + np * j] = s;
        }
}
// ShapJac: out[n + nge*m + nge*npe*k] = shapegt[n + nge*m] * jac[n + nge*k]
void h_ShapJac(double* out, const double* shapegt, const double* jac, int nge, int npe, int ne)
{
    for (int k = 0; k < ne; ++k)
        for (int m = 0; m < npe; ++m)
            for (int n = 0; n < nge; ++n)
                out[n + nge * m + nge * npe * k] = shapegt[n + nge * m] * jac[n + nge * k];
}
// ArrayGemmBatch1 (accumulates): C[i+I*j+IJ*s] += A[i+I*k+IK*s] * B[k+K*j+KJ*s]
void h_GemmBatch1(double* C, const double* A, const double* B, int I, int J, int K, int S)
{
    for (int s = 0; s < S; ++s)
        for (int j = 0; j < J; ++j)
            for (int i = 0; i < I; ++i) {
                double sum = C[i + I * j + I * J * s];
                for (int k = 0; k < K; ++k)
                    sum += A[i + I * k + I * K * s] * B[k + K * j + K * J * s];
                C[i + I * j + I * J * s] = sum;
            }
}
// ArrayMatrixMultiplication1 (accumulates): C[i+I*j] += A[i+I*k] * B[k+K*j]
void h_MatMul1(double* C, const double* A, const double* B, int I, int J, int K)
{
    for (int j = 0; j < J; ++j)
        for (int i = 0; i < I; ++i) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) sum += A[i + I * k] * B[k + K * j];
            C[i + I * j] += sum;
        }
}
void h_SetValue(double* y, double a, int n) { for (int i = 0; i < n; ++i) y[i] = a; }
// Inverse in place, per n x n block (Gauss-Jordan; matches Inverse() semantics)
void h_Inverse(double* A, int n, int batch)
{
    for (int b = 0; b < batch; ++b) {
        double* M = A + static_cast<std::size_t>(n) * n * b;
        std::vector<double> Inv(n * n, 0.0);
        for (int i = 0; i < n; ++i) Inv[i + n * i] = 1.0;
        for (int col = 0; col < n; ++col) {
            int piv = col; double amax = std::abs(M[col + n * col]);
            for (int r = col + 1; r < n; ++r) { double a = std::abs(M[r + n * col]); if (a > amax) { amax = a; piv = r; } }
            if (piv != col)
                for (int c = 0; c < n; ++c) { std::swap(M[col + n * c], M[piv + n * c]); std::swap(Inv[col + n * c], Inv[piv + n * c]); }
            double d = M[col + n * col];
            for (int c = 0; c < n; ++c) { M[col + n * c] /= d; Inv[col + n * c] /= d; }
            for (int r = 0; r < n; ++r) {
                if (r == col) continue;
                double f = M[r + n * col];
                for (int c = 0; c < n; ++c) { M[r + n * c] -= f * M[col + n * c]; Inv[r + n * c] -= f * Inv[col + n * c]; }
            }
        }
        for (int i = 0; i < n * n; ++i) M[i] = Inv[i];
    }
}

// ---- the DGProjection algorithm, replayed with the host primitives ---------
// Layouts mirror the real master struct:
//   shapegt_t [nge x npe_t]  (gauss-major values, = master.shapegt block 0)
//   dshapegt  [nge x npe_t]  (gauss-major d/dxi, 1D only; = master.shapegt block 1)
//   shapegw_t [npe_t x nge]  (node-major, weighted; = master.shapegw)
//   shapegs   [nge x npe_s]  (gauss-major source values at target gauss pts)
//   dgnodes   [npe_t x ne]   (1D coords)
void batched_1d(double* U1, const double* U, const double* shapegt_t, const double* dshapegt,
                const double* shapegw_t, const double* shapegs, const double* dgnodes,
                int npe_t, int npe_s, int nge, int nc, int ne, bool curved)
{
    if (!curved) {
        // P0 = M0^{-1} C0, shared across all elements
        std::vector<double> M0(npe_t * npe_t), C0(npe_t * npe_s), P0(npe_t * npe_s, 0.0);
        h_Gauss2Node(M0.data(), shapegt_t, shapegw_t, nge, npe_t, npe_t);
        h_Inverse(M0.data(), npe_t, 1);
        h_Gauss2Node(C0.data(), shapegs, shapegw_t, nge, npe_t, npe_s);
        h_MatMul1(P0.data(), M0.data(), C0.data(), npe_t, npe_s, npe_t);
        // U1(:,:,e) = P0 * U(:,:,e) for the whole block at once
        h_Gauss2Node(U1, U, P0.data(), npe_s, npe_t, nc * ne);
        return;
    }
    // curved: per-element M, C, then M^{-1}(C U)
    std::vector<double> jac(nge * ne), workt(nge * npe_t * ne), works(nge * npe_s * ne);
    std::vector<double> Minv(npe_t * npe_t * ne), C(npe_t * npe_s * ne), L(npe_t * nc * ne);
    // geometry: Jg = Node2Gauss(dgnodes, dshapegt); jac = Jg (ElemGeom1D) -- the
    // real DGProjection path (Node2Gauss with the derivative shape block).
    h_Node2Gauss(jac.data(), dgnodes, dshapegt, nge, npe_t, ne);
    h_ShapJac(workt.data(), shapegt_t, jac.data(), nge, npe_t, ne);
    h_Gauss2Node(Minv.data(), workt.data(), shapegw_t, nge, npe_t, npe_t * ne);
    h_Inverse(Minv.data(), npe_t, ne);
    h_ShapJac(works.data(), shapegs, jac.data(), nge, npe_s, ne);
    h_Gauss2Node(C.data(), works.data(), shapegw_t, nge, npe_t, npe_s * ne);
    h_SetValue(L.data(), 0.0, npe_t * nc * ne);
    h_GemmBatch1(L.data(), C.data(), U, npe_t, nc, npe_s, ne);
    h_SetValue(U1, 0.0, npe_t * nc * ne);
    h_GemmBatch1(U1, Minv.data(), L.data(), npe_t, nc, npe_t, ne);
}

// ---- 1D bases + 3-pt Gauss (shared with the oracle inputs) -----------------
struct Gauss1D {
    std::array<double, 3> xi, w;
    Gauss1D() { double a = std::sqrt(0.6); xi = {0.5 * (1 - a), 0.5, 0.5 * (1 + a)}; w = {5.0/18, 4.0/9, 5.0/18}; }
};
std::array<double, 2> p1(double x) { return {1 - x, x}; }
std::array<double, 2> dp1(double)  { return {-1.0, 1.0}; }
std::array<double, 3> p2(double x) { return {2*x*x - 3*x + 1, -4*x*x + 4*x, 2*x*x - x}; }
std::array<double, 3> dp2(double x){ return {4*x - 3, -8*x + 4, 4*x - 1}; }

} // namespace

int main()
{
    Gauss1D g;
    const int nge = 3;
    std::vector<double> gw(g.w.begin(), g.w.end());

    // Oracle inputs (node-major): shape_t [npe x nge], dshape_t [npe x nge x 1], shape_s [npe_s x nge]
    auto nodeShape = [&](int np, std::function<std::vector<double>(double)> b) {
        std::vector<double> S(np * nge);
        for (int q = 0; q < nge; ++q) { auto v = b(g.xi[q]); for (int i = 0; i < np; ++i) S[i + np * q] = v[i]; }
        return S;
    };
    // Backend inputs: shapegt [nge x np] (gauss-major), shapegw [np x nge] weighted
    auto gaussShape = [&](int np, std::function<std::vector<double>(double)> b) {
        std::vector<double> S(nge * np);
        for (int q = 0; q < nge; ++q) { auto v = b(g.xi[q]); for (int i = 0; i < np; ++i) S[q + nge * i] = v[i]; }
        return S;
    };
    auto weightShape = [&](int np, std::function<std::vector<double>(double)> b) {
        std::vector<double> S(np * nge);
        for (int q = 0; q < nge; ++q) { auto v = b(g.xi[q]); for (int i = 0; i < np; ++i) S[i + np * q] = v[i] * gw[q]; }
        return S;
    };
    auto v2 = [](std::function<std::array<double,2>(double)> f){ return [f](double x){ auto a=f(x); return std::vector<double>(a.begin(),a.end()); }; };
    auto v3 = [](std::function<std::array<double,3>(double)> f){ return [f](double x){ auto a=f(x); return std::vector<double>(a.begin(),a.end()); }; };

    // oracle-layout shapes
    std::vector<double> nS_p1 = nodeShape(2, v2(p1)),  nS_p2 = nodeShape(3, v3(p2));
    std::vector<double> nD_p1(2 * nge * 1), nD_p2(3 * nge * 1);
    for (int q = 0; q < nge; ++q) { auto d = dp1(g.xi[q]); for (int i = 0; i < 2; ++i) nD_p1[i + 2 * (q + nge * 0)] = d[i]; }
    for (int q = 0; q < nge; ++q) { auto d = dp2(g.xi[q]); for (int i = 0; i < 3; ++i) nD_p2[i + 3 * (q + nge * 0)] = d[i]; }
    // backend-layout shapes
    std::vector<double> gS_p1 = gaussShape(2, v2(p1)), gS_p2 = gaussShape(3, v3(p2));
    std::vector<double> gD_p1 = gaussShape(2, v2(dp1)), gD_p2 = gaussShape(3, v3(dp2));
    std::vector<double> gW_p1 = weightShape(2, v2(p1)), gW_p2 = weightShape(3, v3(p2));

    auto runCheck = [&](const std::string& name, bool curved,
                        const std::vector<double>& X_p2 /*3 target nodes*/,
                        int npe_s, const std::vector<double>& U /*[npe_s x nc]*/, int nc) {
        // oracle: target = p2, source depends on npe_s
        const double* shp_s_node = (npe_s == 2) ? nS_p1.data() : nS_p2.data();
        const double* shp_s_gauss = (npe_s == 2) ? gS_p1.data() : gS_p2.data();
        std::vector<double> ref(3 * nc, 0.0), got(3 * nc, 0.0);
        dgprojection(ref.data(), U.data(), X_p2.data(),
                     nS_p2.data(), nD_p2.data(), shp_s_node, gw.data(),
                     3, npe_s, nge, 1, 1, nc, 1);
        batched_1d(got.data(), U.data(), gS_p2.data(), gD_p2.data(), gW_p2.data(),
                   shp_s_gauss, X_p2.data(), 3, npe_s, nge, nc, 1, curved);
        for (int i = 0; i < 3 * nc; ++i) expectClose(got[i], ref[i], name + " idx " + std::to_string(i));
    };

    // affine element -> straight path AND curved path must both match the oracle
    std::vector<double> Xaff = {2.0, 3.5, 5.0};
    runCheck("affine p1->p2 straight", false, Xaff, 2, {5.0, 11.0}, 1);
    runCheck("affine p1->p2 curved",   true,  Xaff, 2, {5.0, 11.0}, 1);
    runCheck("affine p2->p2 straight (2 comp)", false, Xaff, 3, {7.0, -3.0, 2.0,  1.0, 4.0, 9.0}, 2);

    // curved element (node 0.5 pulled off-centre -> jac varies): only the curved
    // path is valid, and it must match the oracle (exercises jac weighting).
    std::vector<double> Xcur = {0.0, 0.4, 1.0};
    runCheck("curved p2->p2 identity", true, Xcur, 3, {7.0, -3.0, 2.0}, 1);
    runCheck("curved p1->p2 cross-basis", true, Xcur, 2, {5.0, 11.0}, 1);
    runCheck("curved p1->p2 cross-basis (2 comp)", true, Xcur, 2, {5.0, 11.0,  -2.0, 8.0}, 2);

    std::cout << "dgprojection backend-decomposition tests passed\n";
    return 0;
}
