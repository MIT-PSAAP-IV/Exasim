/*
    nonmatchingsetup.hpp

    Setup of the non-conformal, non-matching interface solver
    (docs/nonmatching/nonmatching_multiphysics.tex). Phase 2: donor search.

    When interfacetype = 1 (app.problem[34]), every rank that owns faces of
    its model's coupled interface computes, for each quadrature point x_gl of
    each own face, the donor tuple (r, e, f, a) of eq. (56) by the closest-face
    rule (3), with ties broken by the smallest global face index. It also
    records dist(x_gl, F_f) and the reference coordinate xi of the closest
    point (eq. (21)). From the tuples it forms K_l (eqs. (4), (6)), R_l and
    G_{l,r} (eq. (60)) and fills the owner-side fields of the frozen struct
    (nonmatchinginterface.hpp). The same code runs on both models, so both
    directions are covered: a rank of model 1 searches Gamma^2 for its points,
    a rank of model 2 searches Gamma^1.

    First version (phase prompt, task 3): every rank gathers all coupled
    interface faces of both models with MPI_Allgatherv on EXASIM_COMM_WORLD,
    as build_runtime_interface_partition does, and searches the faces of the
    other model locally, projecting every point onto every candidate face.
    Every world rank takes part, including ranks without interface faces; this
    is a limitation of the first version. Phase 7 replaces the all-gather and
    the exhaustive search with the bounding-box search on an interface
    communicator (Section 3.2, Steps D1-D5).

    Requirements: include after Common/common.h and interfacepartition.hpp.
    Host-only setup code.
*/

#ifndef __NONMATCHINGSETUP_HPP
#define __NONMATCHINGSETUP_HPP

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "nonmatchinginterface.hpp"
#include "nonmatchinggeometry.hpp"
#include "nonmatchingruntime.hpp" // [nonmatching] phase 8 transactional host/runtime rebuild
#include "nonmatchingquadrature.hpp" // [nonmatching] phase 9 A1 composite owner quadrature
#include "nonmatchingmastercompatibility.hpp" // [nonmatching] phase 10 mixed element families

namespace exasim {
namespace nonmatching {

// Parameters of the donor search (phase 2, Revision 3).
//
// Supported range (enforced in donor_search): nd = 2 (line faces) and
// 1 <= porder <= 4. The certified distance bounds of the tie rule
// (certified_distance_bounds) use the Bernstein form of a line face and the
// convex hull of its control points in the plane; 3D faces (triangle and
// square patches) need their own enclosure and tests before they are enabled.
// The fixed-size work arrays of the Bernstein routines hold degree <= 4.
// Phase 7: Distributed (Steps D1-D5 on comm_Gamma, production default) or
// Exhaustive (all faces gathered on comm_Gamma, then donor_search; oracle and
// diagnostics only). Both use the same certified evaluator and tie rule.
enum class SearchMode { Distributed = 0, Exhaustive = 1 };

struct DonorSearchParams {
    SearchMode searchMode = SearchMode::Distributed;
    dstype onFaceRelTol = 1.0e-10;    // diagnostics only: point on its donor face if dist <= onFaceRelTol * h_f
    Int maxProjectionIterations = 100;
    dstype projectionTolerance = 1.0e-13; // step norm in reference-face units (eqs. (21)-(22))
    Int maxPorder = 4;                // supported range, see above
    Int maxInverseIterations = 50;
    Int maxProjectedInverseIterations = 100;
    // Forward-error allowance for a degree <= 4 map (at most 25 nodes), its
    // residual norm, and the small Newton solve. The physical tolerance is
    // this factor times eps times max(|x|_inf, donor-element extent).
    dstype inverseToleranceFactor = 64.0;
    // Fallback neighborhood in dimensionless reference-element coordinates.
    // Unrestricted Newton is attempted first; the fallback retains the best
    // finite projected least-squares point when an exact inverse is unavailable.
    dstype inverseProjectionMargin = 2.0;
    // [nonmatching] Phase 9 A1 (opt-in, plan amendment r1 revision 3): composite owner-face
    // quadrature on coincident 2D interfaces. Must be equal on every rank (checked).
    bool compositeOwnerQuadrature = false;
    Int maxCompositePiecesPerFace = 64; // more pieces on one owner face: uniform setup error
    // Projection iteration limit of the A1 endpoint (reverse) search; -1 uses
    // maxProjectionIterations. A test-facing control only (forces endpoint candidate failures).
    Int endpointMaxProjectionIterations = -1;
};

// Interface faces of one rank (own faces) or of all ranks (gathered).
// Per face f: nodes[j + npf*d + npf*nd*f], face nodes in the element-local
// order of mesh.perm, so that xi refers to the same face parameterization as
// chi_a (Phase 1) and as the face Gauss data of the solver.
template <class T = ::dstype, class I = ::Int>
struct InterfaceFaceSet {
    I nd = 0, npf = 0;
    std::vector<I> rank;     // world rank that owns the face
    std::vector<I> model;    // 0: model 1 (fileoffset == 0), 1: model 2
    std::vector<I> elem;     // local element on the owning rank
    std::vector<I> lface;    // local face number in elem (0..nfe-1)
    std::vector<I> global;   // global face index nfe*elempart[elem] + lface (per model)
    std::vector<T> nodes;    // [npf x nd x nfaces]

    I size() const { return static_cast<I>(rank.size()); }
};

// Result of the donor search for the own faces of one rank. Point p = g + ngf*l
// is quadrature point g of own face l (l in the order of mesh.intfaces).
// Coordinate-like arrays are point-first, as in the frozen header.
template <class T = ::dstype, class I = ::Int>
struct DonorSearchResult {
    I nd = 0, ngf = 0, nfaces = 0, npts = 0;
    I direction = -1;            // 0: own faces on Gamma^1 (model 1), 1: on Gamma^2
    std::vector<I> ownglobal;    // [nfaces] global index of own face l
    std::vector<T> x;            // [npts x nd] x_gl, x[p + npts*d]
    std::vector<I> r;            // [npts] donor rank r_gl (world)                eq. (56)
    std::vector<I> e;            // [npts] local donor element on r                eq. (56)
    std::vector<I> f;            // [npts] global donor face index f_gl = k_gl     eqs. (3), (56)
    std::vector<I> a;            // [npts] local face number of f in e             eq. (56)
    std::vector<T> dist;         // [npts] dist(x_gl, F_f)                         eq. (3)
    std::vector<T> xi;           // [npts x (nd-1)] closest point in the reference face, eq. (21)
    std::vector<T> hface;        // [npts] diagonal of the donor face box (local length scale)
    std::vector<char> converged; // [npts] projection onto the donor face converged
    // Phase 3 inverse-map diagnostics, indexed by coupling direction. These
    // remain outside the frozen run-time data structure.
    I inversePoints[2] = {0, 0};
    I projectedExactInversePoints[2] = {0, 0};
    I projectedInexactInversePoints[2] = {0, 0};
    T maxInverseResidual[2] = {0, 0};
    T maxRelativeInverseResidual[2] = {0, 0};
    T maxReferenceExtrapolation[2] = {0, 0};
    // [nonmatching] Phase 9 A1: composite owner points (empty on the uniform path, where
    // point p = g + ngf*l). facePointPtr[l]..facePointPtr[l+1]-1 are the points of own face
    // l, piece-major then Gauss index; pointUniform marks faces without breakpoints, whose
    // points are the uniform master points (computed with the uniform expressions).
    std::vector<I> facePointPtr;  // [nfaces + 1]
    std::vector<I> pointFace, pointGauss, pointPiece; // [npts]
    std::vector<T> pointS, pointLen;                  // [npts] owner reference s, piece length
    std::vector<char> pointUniform;                   // [npts]
    std::vector<T> pointTest;                         // [npf x npts] psi_i(s_p)
};

// [nonmatching] Phase 9 A1: point range of own face l (uniform: g + ngf*l).
template <class T, class I>
inline void face_point_range(const DonorSearchResult<T, I>& res, I l, I& begin, I& end)
{
    if (res.facePointPtr.empty()) { begin = res.ngf * l; end = begin + res.ngf; }
    else { begin = res.facePointPtr[l]; end = res.facePointPtr[l + 1]; }
}

// [nonmatching] Phase 9 A1: own face, Gauss index and piece of point p.
template <class T, class I>
inline void point_face_gauss(const DonorSearchResult<T, I>& res, I p, I& l, I& g, I& piece)
{
    if (res.facePointPtr.empty()) { l = p / res.ngf; g = p - res.ngf * l; piece = 0; }
    else { l = res.pointFace[p]; g = res.pointGauss[p]; piece = res.pointPiece[p]; }
}

// [nonmatching] Phase 9 A1: 1D Lagrange basis on the master face nodes xpf (npf values in
// [0, 1]) and its derivative at the reference coordinate s.
template <class T, class I>
inline void face_lagrange_1d(T* psi, T* dpsi, const T* xpf, I npf, T s)
{
    for (I j = 0; j < npf; ++j) {
        T value = 1, derivative = 0;
        for (I k = 0; k < npf; ++k) {
            if (k == j) continue;
            const T denom = xpf[j] - xpf[k];
            T term = 1 / denom;
            for (I m = 0; m < npf; ++m)
                if (m != j && m != k) term *= (s - xpf[m]) / (xpf[j] - xpf[m]);
            derivative += term;
            value *= (s - xpf[k]) / denom;
        }
        psi[j] = value;
        if (dpsi) dpsi[j] = derivative;
    }
}

// Phase 7: optional counters of the search (tests and reports only; no effect on results).
struct SearchStatistics {
    int member = 0;                    // this rank belongs to comm_Gamma
    int interfaceSize = 0;             // |P_Gamma|
    long long interfaceCollectives = 0;// collectives this rank entered on comm_Gamma
    long long boxBytes = 0;            // D2 bytes received
    long long countBytes = 0;          // Alltoall count bytes (sent + received)
    long long queryBytes = 0;          // D3 + D4 query payload sent
    long long responseBytes = 0;       // D3 + D4 response payload sent
    long long ownerPoints = 0;         // own quadrature points searched
    long long d4Queries = 0;           // D4 (point, processor) queries sent
    long long maxD4FanOut = 0;         // max D4 processors of one point
    long long facesPerPointSum = 0;    // sum over own points of faces evaluated for it
    long long maxFacesPerPoint = 0;    // max faces evaluated for one own point
    long long facesEvaluated = 0;      // faces evaluated by this rank (all received points)
    long long remoteFaceGeometry = 0;  // remote faces whose node coordinates this rank stores
    double maxGapRatio = 0;            // max observed gap ratio gapUpper / gapTarget of accepted faces
    double searchSeconds = 0;          // wall time of the search (D1-D5 or exhaustive), excluding D6-D7
    double endpointSeconds = 0;        // [nonmatching] Phase 9 A1: composite owner quadrature setup
};

// Phase 7: identity of the first failing candidate of each owner point (smallest
// donor world rank, then smallest global face), -1 if the point did not fail.
template <class I>
struct SearchFailures {
    std::vector<I> donorRank, face;
    void reset(I npts) { donorRank.assign(npts, -1); face.assign(npts, -1); }
    void record(I p, I rank, I global)
    {
        if (donorRank[p] < 0 || std::tie(rank, global) < std::tie(donorRank[p], face[p])) {
            donorRank[p] = rank;
            face[p] = global;
        }
    }
};

// Rounding scale of a computed point-to-face distance d = |G(xi) - x| evaluated
// with the production face basis, G(xi) = sum_j psi_j(xi) y_j (nd = 2):
// r = 2 (2 npf eps |S|_2 + npf eps |Y|_2 + (nd+2) eps d), S_d = sum_j |psi_j||y_jd|,
// Y_d = sum_j |y_jd|. It assumes the per-value basis accuracy
// |fl(psi_j) - psi_j| <= npf eps (|psi_j| + 1), which the test checks against an
// analytic Lagrange evaluation (an absolute term is needed near the zeros of
// psi_j, where the measured error reaches 3.5 eps for p = 4 and a purely relative
// bound fails). It is used
// only to decide when the safeguarded projection is needed. The tie rule uses
// the certified bounds of certified_distance_bounds, which do not depend on the
// production basis.
template <class T, class I>
inline T rounding_scale(const T* xi, const T* x, const T* y, const T* xpf, I nd, I npf, I elemtype, I porder)
{
    const T eps = std::numeric_limits<T>::epsilon();
    std::vector<T> psi(static_cast<std::size_t>(npf));
    EvaluateFaceShape(psi.data(), xi, xpf, nd, npf, elemtype, porder);
    T S2 = 0, Y2 = 0, d2 = 0;
    for (I d = 0; d < nd; d++) {
        T Sd = 0, Yd = 0, g = 0;
        for (I j = 0; j < npf; j++) {
            Sd += std::abs(psi[j]) * std::abs(y[j + npf * d]);
            Yd += std::abs(y[j + npf * d]);
            g += psi[j] * y[j + npf * d];
        }
        S2 += Sd * Sd;
        Y2 += Yd * Yd;
        d2 += (g - x[d]) * (g - x[d]);
    }
    return 2 * (2 * static_cast<T>(npf) * eps * std::sqrt(S2) + static_cast<T>(npf) * eps * std::sqrt(Y2) +
                static_cast<T>(nd + 2) * eps * std::sqrt(d2));
}

// ---------------------------------------------------------------------------
// Certified distance bounds for a line face (nd = 2), Revisions 3 and 4.
//
// The face is the polynomial curve G of degree p through its nodes
// (xpf_j, y_j), the same polynomial as sum_j psi_j y_j in exact arithmetic.
// G is written in the Bernstein basis on [0, 1], G = sum_k c_k B_k^p.
//
// Exact-arithmetic argument (Revision 3). On [a, b] the curve G([a, b]) lies in
// the convex hull of its Bernstein control points on [a, b],
// P_k = blossom(a^(p-k), b^k). For any vector u != 0, every point q of that hull
// satisfies |q - x| >= u.(q - x)/|u| >= min_k u.(P_k - x)/|u| (a linear function
// attains its minimum over a convex hull at a vertex). Hence
//     lb([a, b]) = max(0, max_u min_k u.(P_k - x)/|u|)
// is a lower bound of min over [a, b] of |G - x|, for ANY choice of directions u.
// It equals dist(x, hull) for the direction from x to the nearest hull point (the
// directions tried are those to the nearest points of all segments between control
// points, which include it), and it is 0 when x is inside the hull. Branch and
// bound on [0, 1] (split the leaf with the smallest bound, evaluating G at
// midpoints) gives a global lower bound LB = min over leaves; the upper bound UB
// is |G(t) - x| at evaluated parameters t (the projected xi, 0, 1, midpoints).
// A leaf is discarded only when lb > UB, so LB stays global. No curvature bound,
// finite difference, local stationarity argument or orientation predicate is used.
//
// Floating point (Revision 4). Every quantity that enters LB or UB is computed in
// interval arithmetic with outward rounding (struct Interval below): each
// operation +, -, *, /, sqrt is evaluated in round-to-nearest (IEEE 754: the
// result is within 1/2 ulp of the exact value of the operands) and the result is
// widened by TWO ulps with std::nextafter in each direction. One ulp suffices
// under IEEE 754; the second covers the double rounding of reciprocal-math
// (a/b = a*(1/b)). [nonmatching] Phase 7 Revision 2: this enclosure argument holds
// under standard round-to-nearest arithmetic. The production library is built with
// -ffast-math (maintainer decision 2026-10-09), which may contract, reassociate or
// assume finite values; there the bounds are conservative best-effort margins, not a
// formal guarantee. The chain is:
//   1. Collocation matrix Bc_jk = C(p,k) t_j^k (1-t_j)^(p-k), t_j = xpf_j, as
//      intervals (powers as repeated products, binomials exact).
//   2. Coefficients: c~ = M Y with an approximate inverse M (point matrix). The
//      exact c* satisfies c* - c~ = Bc^-1 R, R = Y - Bc c~ (interval). If
//      ||I - M Bc||_inf <= q < 1 (interval upper bound), then
//      |c* - c~|_inf <= ||M R||_inf/(1 - q) =: beta (upper end of an interval
//      computation). Each coefficient is enclosed by [c~ - beta, c~ + beta].
//   3. Control points on [a, b]: de Casteljau with interval coefficients and
//      interval parameters a, b, 1 - a, 1 - b (boxes enclosing the exact P_k).
//   4. lb: for each direction u (any doubles), the lower end of the interval
//      min_k u.(P_k - x) divided by the upper end of |u|, clipped at 0.
//   5. UB: upper end of the interval |G(t) - x|, G(t) by interval de Casteljau.
// If x lies on or near the hull boundary, every u gives a small or negative
// value and lb is 0 or tiny: an uncertain hull test is resolved conservatively by
// construction. Inputs are assumed finite and valid, and the argument also assumes
// that no intermediate exact result overflows the finite range: ia_up of an
// overflowed result steps toward max() and returns max(), which is not an outward
// enclosure. Finite valid geometry includes this precondition. There is no
// NaN/infinity or overflow validation in production code (Exasim's residual-norm
// check is the production detector).
// ---------------------------------------------------------------------------

// Closed interval [lo, hi] with outward rounding (see above). Host setup only.
template <class T>
struct Interval {
    T lo = 0, hi = 0;
    Interval() = default;
    explicit Interval(T v) : lo(v), hi(v) {}
    Interval(T a, T b) : lo(a), hi(b) {}
};

template <class T> inline T ia_down(T v)
{
    const T lo = std::numeric_limits<T>::lowest();
    return std::nextafter(std::nextafter(v, lo), lo);
}
template <class T> inline T ia_up(T v)
{
    const T hi = std::numeric_limits<T>::max();
    return std::nextafter(std::nextafter(v, hi), hi);
}
template <class T> inline Interval<T> ia_add(const Interval<T>& a, const Interval<T>& b)
{
    return Interval<T>(ia_down(a.lo + b.lo), ia_up(a.hi + b.hi));
}
template <class T> inline Interval<T> ia_sub(const Interval<T>& a, const Interval<T>& b)
{
    return Interval<T>(ia_down(a.lo - b.hi), ia_up(a.hi - b.lo));
}
template <class T> inline Interval<T> ia_mul(const Interval<T>& a, const Interval<T>& b)
{
    const T p1 = a.lo * b.lo, p2 = a.lo * b.hi, p3 = a.hi * b.lo, p4 = a.hi * b.hi;
    return Interval<T>(ia_down(std::min(std::min(p1, p2), std::min(p3, p4))),
                       ia_up(std::max(std::max(p1, p2), std::max(p3, p4))));
}
template <class T> inline Interval<T> ia_sqr(const Interval<T>& a)
{
    const T l = std::abs(a.lo), h = std::abs(a.hi);
    const T mx = std::max(l, h), mn = (a.lo <= 0 && a.hi >= 0) ? static_cast<T>(0) : std::min(l, h);
    return Interval<T>(mn == 0 ? static_cast<T>(0) : ia_down(mn * mn), ia_up(mx * mx));
}
// Division by an interval with b.lo > 0 (callers check).
template <class T> inline Interval<T> ia_div_pos(const Interval<T>& a, const Interval<T>& b)
{
    const T q1 = a.lo / b.lo, q2 = a.lo / b.hi, q3 = a.hi / b.lo, q4 = a.hi / b.hi;
    return Interval<T>(ia_down(std::min(std::min(q1, q2), std::min(q3, q4))),
                       ia_up(std::max(std::max(q1, q2), std::max(q3, q4))));
}
template <class T> inline Interval<T> ia_sqrt(const Interval<T>& a)
{
    const T l = std::max(static_cast<T>(0), a.lo), h = std::max(static_cast<T>(0), a.hi);
    return Interval<T>(l == 0 ? static_cast<T>(0) : std::max(static_cast<T>(0), ia_down(std::sqrt(l))),
                       ia_up(std::sqrt(h)));
}
template <class T> inline T ia_mag(const Interval<T>& a) { return std::max(std::abs(a.lo), std::abs(a.hi)); }

// Bernstein collocation data for the face nodes xpf (line face, degree p).
template <class T>
struct BernsteinMaster {
    int p = 0;
    std::vector<T> Bc;              // [(p+1) x (p+1)] Bc[j + (p+1)*k] ~ B_k^p(xpf_j) (point values)
    std::vector<Interval<T>> BcI;   // enclosures of the exact B_k^p(xpf_j)
    std::vector<T> M;               // [(p+1) x (p+1)] approximate inverse of Bc (point matrix)
    T normM = 0;                    // ||M||_inf (rounded up)
    T normIMB = 0;                  // upper bound of ||I - M Bc||_inf over the enclosures
    bool ok = false;                // normIMB < 1/2
};

// Point value of B_k^p(t) (informational; the bounds use bernstein_basis_interval).
template <class T>
inline T bernstein_basis(int p, int k, T t)
{
    T binom = 1;
    for (int i = 1; i <= k; i++) binom = binom * static_cast<T>(p - k + i) / static_cast<T>(i);
    T v = binom;
    for (int i = 0; i < k; i++) v *= t;
    for (int i = 0; i < p - k; i++) v *= (1 - t);
    return v;
}

// Enclosure of B_k^p(t) for a double t: binomial (exact integer for p <= 4) times
// interval powers of t and 1 - t.
template <class T>
inline Interval<T> bernstein_basis_interval(int p, int k, T t)
{
    long long binom = 1;
    for (int i = 1; i <= k; i++) binom = binom * (p - k + i) / i;
    Interval<T> v(static_cast<T>(binom));
    const Interval<T> ti(t), omt = ia_sub(Interval<T>(static_cast<T>(1)), Interval<T>(t));
    for (int i = 0; i < k; i++) v = ia_mul(v, ti);
    for (int i = 0; i < p - k; i++) v = ia_mul(v, omt);
    return v;
}

template <class T, class I>
inline BernsteinMaster<T> make_bernstein_master(const T* xpf, I porder)
{
    BernsteinMaster<T> bm;
    const int p = static_cast<int>(porder), n = p + 1;
    bm.p = p;
    if (p < 1 || p > 4) return bm;
    bm.Bc.assign(static_cast<std::size_t>(n * n), 0);
    bm.BcI.assign(static_cast<std::size_t>(n * n), Interval<T>());
    for (int j = 0; j < n; j++)
        for (int k = 0; k < n; k++) {
            bm.BcI[j + n * k] = bernstein_basis_interval(p, k, xpf[j]);
            bm.Bc[j + n * k] = static_cast<T>(0.5) * (bm.BcI[j + n * k].lo + bm.BcI[j + n * k].hi);
        }
    // Approximate inverse: Gauss-Jordan with partial pivoting on [Bc | I], one refinement step.
    std::vector<T> A(bm.Bc), Minv(static_cast<std::size_t>(n * n), 0);
    for (int i = 0; i < n; i++) Minv[i + n * i] = 1;
    for (int c = 0; c < n; c++) {
        int piv = c;
        for (int r = c + 1; r < n; r++) if (std::abs(A[r + n * c]) > std::abs(A[piv + n * c])) piv = r;
        if (!(std::abs(A[piv + n * c]) > 0)) return bm;
        for (int k = 0; k < n; k++) { std::swap(A[c + n * k], A[piv + n * k]); std::swap(Minv[c + n * k], Minv[piv + n * k]); }
        const T d = A[c + n * c];
        for (int k = 0; k < n; k++) { A[c + n * k] /= d; Minv[c + n * k] /= d; }
        for (int r = 0; r < n; r++) {
            if (r == c) continue;
            const T f = A[r + n * c];
            for (int k = 0; k < n; k++) { A[r + n * k] -= f * A[c + n * k]; Minv[r + n * k] -= f * Minv[c + n * k]; }
        }
    }
    std::vector<T> E(static_cast<std::size_t>(n * n), 0);
    bm.M.assign(static_cast<std::size_t>(n * n), 0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            T s = (i == j) ? 1 : 0;
            for (int k = 0; k < n; k++) s -= bm.Bc[i + n * k] * Minv[k + n * j];
            E[i + n * j] = s;
        }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            T s = Minv[i + n * j];
            for (int k = 0; k < n; k++) s += Minv[i + n * k] * E[k + n * j];
            bm.M[i + n * j] = s;
        }
    // Certified norms: ||I - M BcI||_inf (interval) and ||M||_inf, rounded up.
    for (int i = 0; i < n; i++) {
        Interval<T> rowM(0), rowE(0);
        for (int j = 0; j < n; j++) {
            rowM = ia_add(rowM, Interval<T>(std::abs(bm.M[i + n * j])));
            Interval<T> s((i == j) ? static_cast<T>(1) : static_cast<T>(0));
            for (int k = 0; k < n; k++) s = ia_sub(s, ia_mul(Interval<T>(bm.M[i + n * k]), bm.BcI[k + n * j]));
            rowE = ia_add(rowE, Interval<T>(ia_mag(s)));
        }
        bm.normM = std::max(bm.normM, rowM.hi);
        bm.normIMB = std::max(bm.normIMB, rowE.hi);
    }
    bm.ok = bm.normIMB < static_cast<T>(0.5);
    return bm;
}

// Bernstein form of one line face. c: point coefficients c~ [k + (p+1)*d];
// ci: enclosures of the exact coefficients; beta: radius bound; cmax: max |c~|.
template <class T>
struct BernsteinFace {
    std::vector<T> c;
    std::vector<Interval<T>> ci;
    T beta = 0, cmax = 0;
    bool ok = false;
};

template <class T, class I>
inline BernsteinFace<T> bernstein_face(const BernsteinMaster<T>& bm, const T* y, I npf)
{
    const int n = bm.p + 1;
    BernsteinFace<T> bf;
    bf.c.assign(static_cast<std::size_t>(2 * n), 0);
    bf.ci.assign(static_cast<std::size_t>(2 * n), Interval<T>());
    if (!bm.ok) return bf;
    T normMR = 0;
    for (int d = 0; d < 2; d++) {
        for (int k = 0; k < n; k++) {
            T s = 0;
            for (int j = 0; j < n; j++) s += bm.M[k + n * j] * y[j + npf * d];
            bf.c[k + n * d] = s;
            bf.cmax = std::max(bf.cmax, std::abs(s));
        }
        // R = Y - BcI c~ (interval), then M R (interval).
        std::vector<Interval<T>> R(static_cast<std::size_t>(n));
        for (int j = 0; j < n; j++) {
            Interval<T> s(y[j + npf * d]);
            for (int k = 0; k < n; k++) s = ia_sub(s, ia_mul(bm.BcI[j + n * k], Interval<T>(bf.c[k + n * d])));
            R[j] = s;
        }
        for (int k = 0; k < n; k++) {
            Interval<T> s(0);
            for (int j = 0; j < n; j++) s = ia_add(s, ia_mul(Interval<T>(bm.M[k + n * j]), R[j]));
            normMR = std::max(normMR, ia_mag(s));
        }
    }
    // beta = ||M R||_inf / (1 - q), q = normIMB, rounded up.
    const Interval<T> den = ia_sub(Interval<T>(static_cast<T>(1)), Interval<T>(bm.normIMB));
    bf.beta = ia_div_pos(Interval<T>(normMR), den).hi;
    for (int i = 0; i < 2 * n; i++)
        bf.ci[i] = Interval<T>(ia_down(bf.c[i] - bf.beta), ia_up(bf.c[i] + bf.beta));
    bf.ok = bf.beta >= 0;
    return bf;
}

// Point evaluation G(t) by de Casteljau on c~ (informational; not used in bounds).
template <class T>
inline void bernstein_eval(T* g, const BernsteinFace<T>& bf, int p, T t)
{
    const int n = p + 1;
    T w[2][5];
    for (int d = 0; d < 2; d++) for (int k = 0; k < n; k++) w[d][k] = bf.c[k + n * d];
    for (int r = 1; r <= p; r++)
        for (int d = 0; d < 2; d++)
            for (int k = 0; k + r <= p; k++) w[d][k] = (1 - t) * w[d][k] + t * w[d][k + 1];
    g[0] = w[0][0];
    g[1] = w[1][0];
}

// Enclosure of G(t): interval de Casteljau on the coefficient enclosures.
template <class T>
inline void bernstein_eval_interval(Interval<T>* g, const BernsteinFace<T>& bf, int p, T t)
{
    const int n = p + 1;
    const Interval<T> ti(t), omt = ia_sub(Interval<T>(static_cast<T>(1)), Interval<T>(t));
    Interval<T> w[2][5];
    for (int d = 0; d < 2; d++) for (int k = 0; k < n; k++) w[d][k] = bf.ci[k + n * d];
    for (int r = 1; r <= p; r++)
        for (int d = 0; d < 2; d++)
            for (int k = 0; k + r <= p; k++) w[d][k] = ia_add(ia_mul(omt, w[d][k]), ia_mul(ti, w[d][k + 1]));
    g[0] = w[0][0];
    g[1] = w[1][0];
}

// Point control points on [a, b] (informational: used only to choose directions).
template <class T>
inline void bernstein_control_points(T* P, const BernsteinFace<T>& bf, int p, T a, T b)
{
    const int n = p + 1;
    for (int k = 0; k <= p; k++) {
        T w[2][5];
        for (int d = 0; d < 2; d++) for (int i = 0; i < n; i++) w[d][i] = bf.c[i + n * d];
        for (int r = 1; r <= p; r++) {
            const T t = (r <= p - k) ? a : b;
            for (int d = 0; d < 2; d++)
                for (int i = 0; i + r <= p; i++) w[d][i] = (1 - t) * w[d][i] + t * w[d][i + 1];
        }
        P[2 * k] = w[0][0];
        P[2 * k + 1] = w[1][0];
    }
}

// Enclosures (boxes) of the exact control points on [a, b]: P[2k + d].
template <class T>
inline void bernstein_control_points_interval(Interval<T>* P, const BernsteinFace<T>& bf, int p, T a, T b)
{
    const int n = p + 1;
    const Interval<T> ai(a), bi(b);
    const Interval<T> oma = ia_sub(Interval<T>(static_cast<T>(1)), ai), omb = ia_sub(Interval<T>(static_cast<T>(1)), bi);
    for (int k = 0; k <= p; k++) {
        Interval<T> w[2][5];
        for (int d = 0; d < 2; d++) for (int i = 0; i < n; i++) w[d][i] = bf.ci[i + n * d];
        for (int r = 1; r <= p; r++) {
            const bool useA = (r <= p - k);
            const Interval<T>& t = useA ? ai : bi;
            const Interval<T>& omt = useA ? oma : omb;
            for (int d = 0; d < 2; d++)
                for (int i = 0; i + r <= p; i++) w[d][i] = ia_add(ia_mul(omt, w[d][i]), ia_mul(t, w[d][i + 1]));
        }
        P[2 * k] = w[0][0];
        P[2 * k + 1] = w[1][0];
    }
}

template <class T>
inline T segment_distance(const T* x, const T* A, const T* B)
{
    const T ux = B[0] - A[0], uy = B[1] - A[1];
    const T L2 = ux * ux + uy * uy;
    T t = 0;
    if (L2 > 0) t = std::max(static_cast<T>(0), std::min(static_cast<T>(1), ((x[0] - A[0]) * ux + (x[1] - A[1]) * uy) / L2));
    const T dx = A[0] + t * ux - x[0], dy = A[1] + t * uy - x[1];
    return std::sqrt(dx * dx + dy * dy);
}

// Approximate dist(x, hull(P)) in floating point (informational, Revision 3);
// the certified bound is hull_lower_bound.
template <class T>
inline T hull_distance(const T* x, const T* P, int n)
{
    auto cross = [](const T* o, const T* u, const T* v) {
        return (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0]);
    };
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            for (int k = j + 1; k < n; k++) {
                const T c1 = cross(&P[2 * i], &P[2 * j], x), c2 = cross(&P[2 * j], &P[2 * k], x),
                        c3 = cross(&P[2 * k], &P[2 * i], x);
                if ((c1 > 0 && c2 > 0 && c3 > 0) || (c1 < 0 && c2 < 0 && c3 < 0)) return 0;
            }
    T best = std::numeric_limits<T>::max();
    for (int i = 0; i < n; i++)
        for (int j = i; j < n; j++) best = std::min(best, segment_distance(x, &P[2 * i], &P[2 * j]));
    return best;
}

// Certified lower bound of dist(x, hull of the exact control points), whose
// enclosures are the boxes PI (step 4 above): max over directions u of the lower
// end of min_k u.(PI_k - x) / |u|_up, clipped at 0. Directions: from x to the
// nearest point of each segment between box centers (and to each center).
template <class T>
inline T hull_lower_bound(const T* x, const Interval<T>* PI, int n)
{
    T C[10];
    for (int k = 0; k < n; k++)
        for (int d = 0; d < 2; d++) C[2 * k + d] = static_cast<T>(0.5) * PI[2 * k + d].lo + static_cast<T>(0.5) * PI[2 * k + d].hi;
    const Interval<T> xi0(x[0]), xi1(x[1]);
    T best = 0;
    auto tryDirection = [&](T u0, T u1) {
        if (!(u0 != 0 || u1 != 0)) return;
        const Interval<T> U0(u0), U1(u1);
        T mn = std::numeric_limits<T>::max();
        for (int k = 0; k < n; k++) {
            const Interval<T> v = ia_add(ia_mul(U0, ia_sub(PI[2 * k], xi0)), ia_mul(U1, ia_sub(PI[2 * k + 1], xi1)));
            mn = std::min(mn, v.lo);
        }
        if (!(mn > 0)) return;
        const Interval<T> un = ia_sqrt(ia_add(ia_sqr(U0), ia_sqr(U1)));
        const T b = ia_div_pos(Interval<T>(mn), un).lo;
        if (b > best) best = b;
    };
    for (int i = 0; i < n; i++) {
        tryDirection(C[2 * i] - x[0], C[2 * i + 1] - x[1]);
        for (int j = i + 1; j < n; j++) {
            const T* A = &C[2 * i];
            const T* B = &C[2 * j];
            const T ux = B[0] - A[0], uy = B[1] - A[1];
            const T L2 = ux * ux + uy * uy;
            T t = 0;
            if (L2 > 0) t = std::max(static_cast<T>(0), std::min(static_cast<T>(1), ((x[0] - A[0]) * ux + (x[1] - A[1]) * uy) / L2));
            tryDirection(A[0] + t * ux - x[0], A[1] + t * uy - x[1]);
        }
    }
    return best;
}

template <class T>
struct CertifiedDistance;
template <class T>
inline CertifiedDistance<T> certified_distance_bounds(const BernsteinFace<T>& bf, int p, const T* x, T xi0,
        T gapTol, int maxSplits);

// Gap target of the branch and bound: a small multiple of the rounding scale of the
// bounds, so that the projection part of (upper - lower) is not larger than it.
template <class T>
inline T certified_gap_target(const BernsteinFace<T>& bf, int p, const T* x)
{
    return static_cast<T>(4 * (2 * p + 2)) * std::numeric_limits<T>::epsilon() *
           (bf.cmax + std::max(std::abs(x[0]), std::abs(x[1]))) + 4 * bf.beta;
}

// Certified bounds for a face given by its nodes y (npf x 2) and the projected xi0.
template <class T, class I>
inline CertifiedDistance<T> face_distance_bounds(const BernsteinMaster<T>& bm, const T* y, I npf, const T* x, T xi0)
{
    const BernsteinFace<T> bf = bernstein_face(bm, y, npf);
    return certified_distance_bounds(bf, bm.p, x, xi0, certified_gap_target(bf, bm.p, x), 4000);
}

template <class T>
struct CertifiedDistance {
    T lower = 0, upper = 0;   // lower <= d* <= upper (d* = distance from x to the face)
    T ub = 0;                 // = upper: certified upper bound at xibest
    T xibest = 0;             // parameter attaining ub
    bool ok = false;
};

// Certified bounds of the distance from x to a line face (see above). xi0 is the
// projected point; the branch and bound stops when upper - LB <= gapTol, when no
// leaf can be split, or after maxSplits splits (the bounds stay valid, only wider).
template <class T>
inline CertifiedDistance<T> certified_distance_bounds(const BernsteinFace<T>& bf, int p, const T* x, T xi0,
        T gapTol, int maxSplits)
{
    CertifiedDistance<T> cd;
    if (!bf.ok || p < 1 || p > 4) return cd;
    const T eps = std::numeric_limits<T>::epsilon();
    const Interval<T> X0(x[0]), X1(x[1]);
    auto dist_up = [&](T t) { // upper end of the enclosure of |G(t) - x|
        Interval<T> g[2];
        bernstein_eval_interval(g, bf, p, t);
        return ia_sqrt(ia_add(ia_sqr(ia_sub(g[0], X0)), ia_sqr(ia_sub(g[1], X1)))).hi;
    };
    cd.xibest = std::min(static_cast<T>(1), std::max(static_cast<T>(0), xi0));
    cd.ub = dist_up(cd.xibest);
    for (T t : {static_cast<T>(0), static_cast<T>(1)}) {
        const T v = dist_up(t);
        if (v < cd.ub) { cd.ub = v; cd.xibest = t; }
    }

    struct Leaf { T a, b, lb; };
    std::vector<Leaf> leaves;
    Interval<T> PI[10];
    bernstein_control_points_interval(PI, bf, p, static_cast<T>(0), static_cast<T>(1));
    leaves.push_back({0, 1, hull_lower_bound(x, PI, p + 1)});
    for (int it = 0; it < maxSplits; it++) {
        int imin = -1;
        T LBall = std::numeric_limits<T>::max();
        for (int i = 0; i < static_cast<int>(leaves.size()); i++) {
            LBall = std::min(LBall, leaves[i].lb);
            if ((leaves[i].b - leaves[i].a) > 4 * eps && (imin < 0 || leaves[i].lb < leaves[imin].lb)) imin = i;
        }
        if (imin < 0) break;
        if (!(cd.ub - LBall > gapTol)) break;
        if (leaves[imin].lb > LBall) break; // the minimum sits on a leaf that cannot be split
        const Leaf v = leaves[imin];
        const T m = static_cast<T>(0.5) * v.a + static_cast<T>(0.5) * v.b;
        const T dm = dist_up(m);
        if (dm < cd.ub) { cd.ub = dm; cd.xibest = m; }
        bernstein_control_points_interval(PI, bf, p, v.a, m);
        leaves[imin] = Leaf{v.a, m, hull_lower_bound(x, PI, p + 1)};
        bernstein_control_points_interval(PI, bf, p, m, v.b);
        leaves.push_back(Leaf{m, v.b, hull_lower_bound(x, PI, p + 1)});
        // A leaf whose certified lower bound exceeds the certified upper bound cannot
        // contain the minimizer.
        leaves.erase(std::remove_if(leaves.begin(), leaves.end(), [&](const Leaf& w) { return w.lb > cd.ub; }),
                     leaves.end());
    }
    T LB = std::numeric_limits<T>::max();
    for (const Leaf& v : leaves) LB = std::min(LB, v.lb);
    cd.lower = std::max(static_cast<T>(0), LB);
    cd.upper = cd.ub;
    cd.ok = !leaves.empty() && cd.lower <= cd.upper;
    return cd;
}

// Sample points of the reference face F_hat (line, triangle or square) and the
// face basis at them: psi[j + npf*s]. Used to start the safeguarded projection
// and to detect a local minimum of the Phase 1 projection.
template <class T, class I>
struct FaceSamples {
    I ns = 0;
    std::vector<T> xi;   // [ns x (nd-1)], xi[s*(nd-1) + k]
    std::vector<T> psi;  // [npf x ns]
};

template <class T, class I>
inline FaceSamples<T, I> make_face_samples(const T* xpf, I nd, I npf, I elemtype, I porder)
{
    FaceSamples<T, I> fs;
    const I ndf = nd - 1;
    const I N = 8 * std::max<I>(porder, 1);
    for (I i = 0; i <= N; i++) {
        if (ndf == 1) { fs.xi.push_back(T(i) / T(N)); continue; }
        for (I j = 0; j <= N; j++) {
            if (elemtype == 0 && i + j > N) continue; // triangle face
            fs.xi.push_back(T(i) / T(N));
            fs.xi.push_back(T(j) / T(N));
        }
    }
    fs.ns = static_cast<I>(fs.xi.size()) / ndf;
    fs.psi.assign(static_cast<std::size_t>(npf * fs.ns), static_cast<T>(0));
    for (I s = 0; s < fs.ns; s++)
        EvaluateFaceShape(&fs.psi[static_cast<std::size_t>(npf * s)], &fs.xi[static_cast<std::size_t>(ndf * s)],
                xpf, nd, npf, elemtype, porder);
    return fs;
}

// Stationarity test and refinement for a line face (nd = 2), used when the
// backtracking line search of safeguarded_closest_point stagnates at xi with a
// full projected step s > tol. f'(z) = T(z).(G(z) - x). Converged if
//   (a) xi is a vertex of F_hat and f' points out of F_hat (KKT), or
//   (b) f' changes sign from - to + on [xi - 2s, xi + 2s] (clamped to F_hat); the
//       bracket is then bisected to width <= tol and xi set to its midpoint, so a
//       stationary point (a local minimizer of f along the face) lies within tol/2.
// Otherwise returns false (not converged).
template <class T, class I>
inline bool polish_line_face_stationary(T* xi, const T* x, const T* y, const T* xpf,
        I nd, I npf, I elemtype, I porder, T s, T tol)
{
    std::vector<T> psi(static_cast<std::size_t>(npf)), dpsi(static_cast<std::size_t>(npf));
    T xb[2] = {0, 0}, tg[2] = {0, 0};
    auto fp = [&](T z) {
        const T zz[1] = {z};
        MapReferenceFaceToPhysical(xb, tg, psi.data(), dpsi.data(), zz, y, xpf, nd, npf, elemtype, porder);
        return tg[0] * (xb[0] - x[0]) + tg[1] * (xb[1] - x[1]);
    };
    const T f0 = fp(xi[0]);
    if ((xi[0] <= 0 && f0 >= 0) || (xi[0] >= 1 && f0 <= 0)) return true; // (a)
    T a = std::max(static_cast<T>(0), xi[0] - 2 * s), b = std::min(static_cast<T>(1), xi[0] + 2 * s);
    T fa = fp(a), fb = fp(b);
    if (!(fa < 0 && fb > 0)) {
        // try the half brackets around xi
        if (fa < 0 && f0 > 0) { b = xi[0]; fb = f0; }
        else if (f0 < 0 && fb > 0) { a = xi[0]; fa = f0; }
        else return false;
    }
    for (int it = 0; it < 200 && b - a > tol; it++) { // (b)
        const T c = static_cast<T>(0.5) * (a + b);
        const T fc = fp(c);
        if (fc < 0) a = c; else b = c;
    }
    if (b - a > tol) return false;
    xi[0] = static_cast<T>(0.5) * (a + b);
    return true;
}

// Safeguarded closest-point projection onto one face (Revision 1). Used only
// when the Phase 1 projected Gauss-Newton iteration (eqs. (21)-(22)) does not
// converge, or converges to a point farther than a sample of the face (a local
// minimum). It starts from the closest sample and applies projected
// Gauss-Newton steps with backtracking, accepting a step only if it decreases
// |G(xi) - x|^2. Converged when the full projected Gauss-Newton step is below
// prm.projectionTolerance (first-order stationarity of eq. (21)). This is
// needed for strongly curved faces, where full Gauss-Newton steps cycle when
// dist * curvature > 1. It finds the global minimum up to the sample spacing.
// If no step decreases the objective while the full step is still above the
// tolerance, convergence is decided by polish_line_face_stationary (a sign change
// of f' bracketed to the tolerance, or the KKT condition at a vertex); otherwise
// it reports failure (e.g. floating-point stagnation far from any stationary point).
template <class T, class I>
inline bool safeguarded_closest_point(T* xi, T* dist, const T* x, const T* y, const T* xpf,
        const FaceSamples<T, I>& fs, I nd, I npf, I elemtype, I porder, const DonorSearchParams& prm,
        const T* start = nullptr)
{
    const I ndf = nd - 1;
    std::vector<T> psi(static_cast<std::size_t>(npf)), dpsi(static_cast<std::size_t>(npf * ndf));
    T xb[3] = {0, 0, 0}, tg[6] = {0, 0, 0, 0, 0, 0};
    auto f2 = [&](const T* z) {
        MapReferenceFaceToPhysical(xb, tg, psi.data(), dpsi.data(), z, y, xpf, nd, npf, elemtype, porder);
        T s = 0;
        for (I d = 0; d < nd; d++) s += (xb[d] - x[d]) * (xb[d] - x[d]);
        return s;
    };

    // Closest sample, or the given start.
    T best = std::numeric_limits<T>::max();
    if (start) for (I k = 0; k < ndf; k++) xi[k] = start[k];
    for (I s = 0; s < fs.ns && !start; s++) {
        T v = 0;
        for (I d = 0; d < nd; d++) {
            T c = 0;
            for (I j = 0; j < npf; j++) c += fs.psi[j + npf * s] * y[j + npf * d];
            v += (c - x[d]) * (c - x[d]);
        }
        if (v < best) {
            best = v;
            for (I k = 0; k < ndf; k++) xi[k] = fs.xi[ndf * s + k];
        }
    }

    bool converged = false;
    const I maxit = 10 * prm.maxProjectionIterations;
    for (I it = 0; it < maxit && !converged; it++) {
        const T f0 = f2(xi); // also sets xb, tg at xi
        T metric[4] = {0, 0, 0, 0}, grad[2] = {0, 0}, delta[2] = {0, 0}, trial[2] = {0, 0}, proj[2] = {0, 0};
        for (I i = 0; i < ndf; i++) {
            for (I d = 0; d < nd; d++) {
                grad[i] -= tg[d * ndf + i] * (xb[d] - x[d]);
                for (I j = 0; j < ndf; j++) metric[i * ndf + j] += tg[d * ndf + i] * tg[d * ndf + j];
            }
        }
        if (!SolveScaleInvariantSmallSystem(delta, metric, grad, ndf)) break;
        for (I k = 0; k < ndf; k++) trial[k] = xi[k] + delta[k];
        if (!ProjectReferenceFaceMetric(proj, trial, metric, nd, elemtype)) break;
        T step2 = 0;
        for (I k = 0; k < ndf; k++) step2 += (proj[k] - xi[k]) * (proj[k] - xi[k]);
        if (std::sqrt(step2) <= static_cast<T>(prm.projectionTolerance)) { converged = true; break; }

        // Backtracking along the projected path xi -> P(xi + alpha*delta).
        T alpha = 1;
        bool accepted = false;
        for (int ls = 0; ls < 60 && !accepted; ls++, alpha *= 0.5) {
            for (I k = 0; k < ndf; k++) trial[k] = xi[k] + alpha * delta[k];
            if (!ProjectReferenceFaceMetric(proj, trial, metric, nd, elemtype)) break;
            if (f2(proj) < f0) {
                for (I k = 0; k < ndf; k++) xi[k] = proj[k];
                accepted = true;
            }
        }
        if (!accepted) {
            // No decrease of |G - x|^2 is resolvable in floating point (its rounding is
            // about eps*d^2, while the decrease near the minimizer is O(|xi - xi*|^2)).
            // Stationarity is then decided on f' = T.(G - x), which is computed
            // accurately: line faces only (nd = 2, the supported range).
            converged = (ndf == 1) && polish_line_face_stationary(xi, x, y, xpf, nd, npf, elemtype, porder,
                    std::sqrt(step2), static_cast<T>(prm.projectionTolerance));
            break;
        }
    }
    *dist = std::sqrt(f2(xi));
    return converged;
}

// Closest-face rule (3), certified form (Phase 2, Revisions 1-4). Phase 7 splits
// the search into one shared certified core and two drivers:
//   prepare_candidate_faces   per-face data (box diagonal, samples, Bernstein form);
//   evaluate_candidate_face   the per-(point, face) work, below;
//   select_certified_face     the tie rule of eq. (3);
//   donor_search              the exhaustive driver (all faces, one rank; the oracle);
//   nonmatchingsearch.hpp     the distributed driver (Steps D1-D5).
// Supported range: nd = 2, 1 <= porder <= prm.maxPorder (see DonorSearchParams).
// Per point x and candidate face m (evaluate_candidate_face):
// 1. x is projected onto the face (Phase 1 ClosestPointOnFace, eqs. (21)-(22)).
//    If that projection does not converge, or a sample point of the face is
//    closer than its result by more than the rounding scale (a local minimum),
//    safeguarded_closest_point recomputes it.
// 2. The face gets certified bounds lower_m <= d*_m <= upper_m
//    (certified_distance_bounds: Bernstein branch and bound, global over the
//    face). If the branch and bound finds a point closer than the projection
//    by more than the rounding scale, the projection was a local minimum and is
//    recomputed from that point (then the bounds are recomputed).
// 3. Tie rule (eq. (3), select_certified_face): with U = min_m upper_m, the donor
//    is the face of smallest global index among {m : lower_m <= U}, the faces that
//    cannot be shown to be farther than the closest one. The true closest face
//    always belongs to this set; a face with lower_m > U is resolvably farther
//    and is never selected.
// 4. Failure policy: a face fails if its projection fails, it has no valid
//    bound, or the bound gap upper - lower exceeds its target gapTol (Phase 7;
//    measured on every Phase 2-3 fixture before it was enforced: max ratio 0.97).
//    The exhaustive driver fails the point if ANY candidate fails; the
//    distributed driver if any face on a QUERIED processor fails.
// The reported dist and xi are those of the projection onto the donor. Each face
// is evaluated independently of the others, so the result depends only on the
// point and the set of candidates, not on their order or partition.

// Per-face data of a set of candidate faces, computed once per face set.
template <class T, class I>
struct PreparedFaces {
    const InterfaceFaceSet<T, I>* faces = nullptr;
    I nd = 0, npf = 0, nc = 0, ns = 0, elemtype = 0, porder = 0;
    const T* xpf = nullptr;
    std::vector<T> hf;                       // [nc] diagonal of the nodal face box (diagnostics)
    std::vector<T> sx;                       // [nd x ns x nc] physical sample points
    FaceSamples<T, I> fsamp;
    BernsteinMaster<T> bm;
    std::vector<BernsteinFace<T>> bfaces;    // [nc] Bernstein form (certified bounds, boxes)
};

// Phase 7: outward-rounded upper bound of the gap target 4 (2p+2) eps (cmax + xabs)
// + 4 beta (the expression of certified_gap_target), evaluated in interval
// arithmetic. For nonnegative arguments every exact operation is nondecreasing and
// the outward rounding (nextafter upward of the round-to-nearest result) is
// nondecreasing, so under standard arithmetic the result is nondecreasing in cmax
// and beta: the value for the processor maxima (cmax_q, beta_q) bounds the value of
// every face of the processor. Best-effort under -ffast-math (see above).
template <class T>
inline T certified_gap_target_upper(T cmax, T beta, T xabs, int porder)
{
    const T eps = std::numeric_limits<T>::epsilon();
    Interval<T> g = ia_mul(Interval<T>(static_cast<T>(4 * (2 * porder + 2))), Interval<T>(eps));
    g = ia_mul(g, ia_add(Interval<T>(cmax), Interval<T>(xabs)));
    g = ia_add(g, ia_mul(Interval<T>(static_cast<T>(4)), Interval<T>(beta)));
    return g.hi;
}

// Phase 7: outward-rounded upper bound of the real gap upper - lower.
template <class T>
inline T certified_gap_upper(T upper, T lower)
{
    return ia_sub(Interval<T>(upper), Interval<T>(lower)).hi;
}

// Phase 7: the acceptance predicate of the gap rule, used by both drivers and by
// the pruning margin: the outward-rounded upper bound of upper - lower must not
// exceed the outward-rounded upper bound of the face gap target. Numerical
// comparison only (inputs assumed finite).
template <class T>
inline bool certified_gap_accepted(T upper, T lower, T cmax, T beta, T xabs, int porder)
{
    return certified_gap_upper(upper, lower) <= certified_gap_target_upper(cmax, beta, xabs, porder);
}

// Result of evaluate_candidate_face for one (point, face).
template <class T>
struct FaceEvaluation {
    T dist = std::numeric_limits<T>::max(); // |G(xi) - x| of the projection
    T lower = 0, upper = 0;                 // certified bounds of the distance
    T gapUpper = 0;                         // outward-rounded upper bound of upper - lower
    T gapTarget = 0;                        // outward-rounded upper bound of the gap target
    T xi[2] = {0, 0};                       // projected reference coordinate
    bool ok = false;                        // projection converged, bounds valid, gap rule met
};

template <class T, class I>
inline PreparedFaces<T, I> prepare_candidate_faces(const InterfaceFaceSet<T, I>& cand,
        const T* xpf, I elemtype, I porder)
{
    PreparedFaces<T, I> pf;
    pf.faces = &cand;
    pf.nd = cand.nd;
    pf.npf = cand.npf;
    pf.nc = cand.size();
    pf.elemtype = elemtype;
    pf.porder = porder;
    pf.xpf = xpf;
    const I nd = pf.nd, npf = pf.npf, nc = pf.nc;

    // Diagonal of the box of the face nodes: local length scale for diagnostics only.
    pf.hf.assign(static_cast<std::size_t>(nc), static_cast<T>(0));
    for (I m = 0; m < nc; m++) {
        const T* y = &cand.nodes[static_cast<std::size_t>(npf) * nd * m];
        T diag2 = 0;
        for (I d = 0; d < nd; d++) {
            T lo = y[npf * d], hi = y[npf * d];
            for (I j = 1; j < npf; j++) {
                lo = std::min(lo, y[j + npf * d]);
                hi = std::max(hi, y[j + npf * d]);
            }
            diag2 += (hi - lo) * (hi - lo);
        }
        pf.hf[m] = std::sqrt(diag2);
    }

    // Sample points of every face (physical): sx[d + nd*(s + ns*m)].
    pf.fsamp = make_face_samples(xpf, nd, npf, elemtype, porder);
    const I ns = pf.fsamp.ns;
    pf.ns = ns;
    pf.sx.assign(static_cast<std::size_t>(nd) * ns * nc, static_cast<T>(0));
    for (I m = 0; m < nc; m++) {
        const T* y = &cand.nodes[static_cast<std::size_t>(npf) * nd * m];
        for (I s = 0; s < ns; s++)
            for (I d = 0; d < nd; d++) {
                T c = 0;
                for (I j = 0; j < npf; j++) c += pf.fsamp.psi[j + npf * s] * y[j + npf * d];
                pf.sx[d + nd * (s + static_cast<std::size_t>(ns) * m)] = c;
            }
    }

    // Bernstein form of every face (certified bounds).
    pf.bm = make_bernstein_master(xpf, porder);
    if (!pf.bm.ok)
        error("non-matching donor search: Bernstein collocation matrix of the face nodes is ill-conditioned");
    pf.bfaces.reserve(static_cast<std::size_t>(nc));
    for (I m = 0; m < nc; m++)
        pf.bfaces.push_back(bernstein_face(pf.bm, &cand.nodes[static_cast<std::size_t>(npf) * nd * m], npf));
    return pf;
}

// Steps 1-2 above for the point xp and candidate face m.
template <class T, class I>
inline FaceEvaluation<T> evaluate_candidate_face(const PreparedFaces<T, I>& pf, I m, const T* xp,
        const DonorSearchParams& prm)
{
    const I nd = pf.nd, npf = pf.npf, ns = pf.ns;
    const I elemtype = pf.elemtype, porder = pf.porder;
    const T* xpf = pf.xpf;
    const T* y = &pf.faces->nodes[static_cast<std::size_t>(npf) * nd * m];
    T xb[3] = {0, 0, 0};
    T dd = std::numeric_limits<T>::max();
    T xx[2] = {0, 0};
    bool ok = ClosestPointOnFace(xx, xb, &dd, xp, y, xpf, nd, npf, elemtype, porder,
            prm.maxProjectionIterations, static_cast<T>(prm.projectionTolerance)); // eqs. (21)-(22)
    T rs = rounding_scale(xx, xp, y, xpf, nd, npf, elemtype, porder);
    // Safeguard: no convergence, or a sample of the face is closer than the
    // result by more than the rounding scale (local minimum).
    T dsample = std::numeric_limits<T>::max();
    for (I s = 0; s < ns; s++) {
        T v = 0;
        for (I d = 0; d < nd; d++) {
            const T t = pf.sx[d + nd * (s + static_cast<std::size_t>(ns) * m)] - xp[d];
            v += t * t;
        }
        dsample = std::min(dsample, v);
    }
    dsample = std::sqrt(dsample);
    if (!ok || dsample < dd - 2 * rs) {
        T xs[2] = {0, 0};
        T ds = std::numeric_limits<T>::max();
        ok = safeguarded_closest_point(xs, &ds, xp, y, xpf, pf.fsamp, nd, npf, elemtype, porder, prm);
        xx[0] = xs[0]; xx[1] = xs[1];
        dd = ds;
        rs = rounding_scale(xx, xp, y, xpf, nd, npf, elemtype, porder);
    }
    // Certified bounds; gap target: the rounding scale of the bounds themselves.
    const BernsteinFace<T>& bface = pf.bfaces[m];
    const T gap = certified_gap_target(bface, static_cast<int>(porder), xp);
    CertifiedDistance<T> cdist = certified_distance_bounds(bface, static_cast<int>(porder), xp, xx[0], gap, 4000);
    if (ok && cdist.ok && cdist.ub < dd - 2 * rs - (cdist.upper - cdist.ub)) {
        // The projection stopped at a local minimum: restart from the better point.
        T xs[2] = {cdist.xibest, 0};
        T ds = std::numeric_limits<T>::max();
        ok = safeguarded_closest_point(xs, &ds, xp, y, xpf, pf.fsamp, nd, npf, elemtype, porder, prm, xs);
        xx[0] = xs[0]; xx[1] = xs[1];
        dd = ds;
        cdist = certified_distance_bounds(bface, static_cast<int>(porder), xp, xx[0], gap, 4000);
    }
    FaceEvaluation<T> fe;
    fe.dist = dd;
    fe.lower = cdist.lower;
    fe.upper = cdist.upper;
    fe.xi[0] = xx[0];
    fe.xi[1] = xx[1];
    // Phase 7: the branch and bound must also reach its gap target. Faces that stop
    // early (maxSplits, unsplittable leaf) keep valid but wider bounds; they fail.
    // The test uses outward-rounded bounds: under standard arithmetic upper - lower
    // <= gapUpper <= gapTarget holds in real arithmetic for every accepted face,
    // which the pruning margin of the distributed search (Step D4) relies on
    // (best-effort under -ffast-math). The branch and bound itself still uses the
    // floating-point target gap, so its iterations are unchanged.
    const T xabs = std::max(std::abs(xp[0]), std::abs(xp[1]));
    fe.gapUpper = certified_gap_upper(cdist.upper, cdist.lower);
    fe.gapTarget = certified_gap_target_upper(bface.cmax, bface.beta, xabs, static_cast<int>(porder));
    fe.ok = ok && cdist.ok &&
            certified_gap_accepted(cdist.upper, cdist.lower, bface.cmax, bface.beta, xabs, static_cast<int>(porder));
    return fe;
}

// Step 3 above: index (into lower/upper/global) of the donor, or -1 if n == 0.
template <class T, class I>
inline I select_certified_face(const T* lower, const T* upper, const I* global, I n)
{
    T U = std::numeric_limits<T>::max();
    for (I m = 0; m < n; m++) U = std::min(U, upper[m]);
    I best = -1;
    for (I m = 0; m < n; m++)
        if (lower[m] <= U && (best < 0 || global[m] < global[best]))
            best = m; // eq. (3): smallest global index among faces not resolvably farther
    return best;
}

// Exhaustive driver (the oracle): every point of res against every face of cand,
// on one rank, no MPI.
template <class T, class I>
inline void donor_search(DonorSearchResult<T, I>& res, const InterfaceFaceSet<T, I>& cand,
        const T* xpf, I elemtype, I porder, const DonorSearchParams& prm = DonorSearchParams(),
        SearchFailures<I>* failures = nullptr)
{
    const I nd = res.nd;
    const I ndf = nd - 1;
    const I npts = res.npts;
    const I nc = cand.size();

    res.r.assign(npts, -1);
    res.e.assign(npts, -1);
    res.f.assign(npts, -1);
    res.a.assign(npts, -1);
    res.dist.assign(npts, std::numeric_limits<T>::max());
    res.xi.assign(static_cast<std::size_t>(npts) * static_cast<std::size_t>(std::max<I>(ndf, 1)), static_cast<T>(0));
    res.hface.assign(npts, static_cast<T>(0));
    res.converged.assign(npts, 0);
    if (failures) failures->reset(npts);
    if (npts == 0)
        return;
    if (nd != 2 || porder < 1 || porder > prm.maxPorder) {
        std::ostringstream oss;
        oss << "non-matching donor search: supported for nd = 2 and 1 <= porder <= " << prm.maxPorder
            << " (got nd = " << nd << ", porder = " << porder << "); see DonorSearchParams";
        error(oss.str());
    }
    if (nc == 0)
        error("non-matching donor search: the other model has no interface faces");

    const PreparedFaces<T, I> pf = prepare_candidate_faces(cand, xpf, elemtype, porder);
    std::vector<T> dm(static_cast<std::size_t>(nc)), lowm(static_cast<std::size_t>(nc)), upm(static_cast<std::size_t>(nc));
    std::vector<T> xim(static_cast<std::size_t>(2) * nc);

    for (I p = 0; p < npts; p++) {
        T xp[3] = {0, 0, 0};
        for (I d = 0; d < nd; d++)
            xp[d] = res.x[static_cast<std::size_t>(p) + static_cast<std::size_t>(npts) * d];

        bool allok = true;
        for (I m = 0; m < nc; m++) {
            const FaceEvaluation<T> fe = evaluate_candidate_face(pf, m, xp, prm);
            dm[m] = fe.dist;
            lowm[m] = fe.lower;
            upm[m] = fe.upper;
            xim[2 * m] = fe.xi[0];
            xim[2 * m + 1] = fe.xi[1];
            if (!fe.ok) {
                allok = false;
                if (failures) failures->record(p, cand.rank[m], cand.global[m]);
            }
        }
        const I best = select_certified_face(lowm.data(), upm.data(), cand.global.data(), nc);

        res.r[p] = cand.rank[best];
        res.e[p] = cand.elem[best];
        res.f[p] = cand.global[best];
        res.a[p] = cand.lface[best];
        res.dist[p] = dm[best];
        for (I k = 0; k < ndf; k++)
            res.xi[static_cast<std::size_t>(p) + static_cast<std::size_t>(npts) * k] = xim[2 * best + k];
        res.hface[p] = pf.hf[best];
        res.converged[p] = allok ? 1 : 0;
    }
}

// Own coupled interface faces of this rank, in the order of mesh.intfaces
// (faces with bf == problem[30] over e < ne1, element then local face; see
// collect_local_interface_faces and discretization.cpp getinterfacefaces).
// sol.xdg is npe x ncx x ne.
template <class T, class I>
inline InterfaceFaceSet<T, I> collect_own_interface_faces(const appstructT<T, I>& app,
        const masterstructT<T, I>& master, const meshstructT<T, I>& mesh, const solstructT<T, I>& sol,
        I mpirank, I fileoffset)
{
    InterfaceFaceSet<T, I> faces;
    const I bc = app.problem[30];
    const I nd = master.ndims[0];
    const I npe = master.ndims[5];
    const I npf = master.ndims[6];
    const I ne = mesh.ndims[1];
    const I nfe = mesh.ndims[4];
    const I ncx = app.ndims[AppNdims::ncx];
    faces.nd = nd;
    faces.npf = npf;

    I ne1 = ne;
    if (mesh.elempartpts && exasim::interfacepartition::mesh_nsize(mesh, static_cast<I>(10)) >= 2)
        ne1 = std::min(ne, mesh.elempartpts[0] + mesh.elempartpts[1]);

    bool haveFace = false;
    for (I e = 0; e < ne1 && !haveFace; e++)
        for (I lf = 0; lf < nfe; lf++)
            if (mesh.bf[lf + nfe * e] == bc) haveFace = true;
    if (haveFace && (mesh.elempart == nullptr || mesh.szelempart < ne)) {
        std::ostringstream oss;
        oss << "non-matching donor search: mesh.elempart has " << mesh.szelempart
            << " entries on rank " << mpirank << ", need ne = " << ne;
        error(oss.str());
    }

    const I model = (fileoffset > 0) ? 1 : 0;
    for (I e = 0; e < ne1; e++) {
        for (I lf = 0; lf < nfe; lf++) {
            if (mesh.bf[lf + nfe * e] != bc)
                continue;
            faces.rank.push_back(mpirank);
            faces.model.push_back(model);
            faces.elem.push_back(e);
            faces.lface.push_back(lf);
            faces.global.push_back(nfe * mesh.elempart[e] + lf); // Section 3.1, global face numbering
            const std::size_t off = faces.nodes.size();
            faces.nodes.resize(off + static_cast<std::size_t>(npf * nd));
            for (I j = 0; j < npf; j++) {
                const I node = mesh.perm[j + npf * lf];
                for (I d = 0; d < nd; d++)
                    faces.nodes[off + j + npf * d] = sol.xdg[node + npe * d + npe * ncx * e];
            }
        }
    }
    return faces;
}

// Quadrature points x_gl = sum_i psi_i(g) y_i of the own faces, with
// master.shapfgt (ngf x npf, values first): x[p + npts*d], p = g + ngf*l.
template <class T, class I>
inline void own_quadrature_points(DonorSearchResult<T, I>& res, const InterfaceFaceSet<T, I>& own,
        const T* shapfgt, I ngf)
{
    const I nd = own.nd, npf = own.npf, nf = own.size();
    res.nd = nd;
    res.ngf = ngf;
    res.nfaces = nf;
    const long long nptsWide = static_cast<long long>(ngf) * static_cast<long long>(nf);
    if (ngf < 0 || nf < 0 || nptsWide > std::numeric_limits<I>::max())
        error("non-matching setup: owner point count exceeds integer limits");
    res.npts = static_cast<I>(nptsWide);
    res.ownglobal = own.global;
    res.x.assign(static_cast<std::size_t>(res.npts) * static_cast<std::size_t>(nd), static_cast<T>(0));
    for (I l = 0; l < nf; l++)
        for (I g = 0; g < ngf; g++)
            for (I d = 0; d < nd; d++) {
                T s = 0;
                for (I j = 0; j < npf; j++)
                    s += shapfgt[g + ngf * j] * own.nodes[static_cast<std::size_t>(j + npf * d + npf * nd * l)];
                res.x[static_cast<std::size_t>(g + ngf * l) +
                      static_cast<std::size_t>(res.npts) * d] = s;
            }
}

template <class I>
inline void copy_to_host_array(I*& ptr, const std::vector<I>& v)
{
    if (ptr) CPUFREE(ptr);
    ptr = nullptr;
    if (!v.empty()) {
        TemplateMalloc(&ptr, static_cast<I>(v.size()), 0);
        std::copy(v.begin(), v.end(), ptr);
    }
}

template <class T>
inline void copy_to_host_scalar_array(T*& ptr, const std::vector<T>& v)
{
    if (ptr) CPUFREE(ptr);
    ptr = nullptr;
    if (!v.empty()) {
        TemplateMalloc(&ptr, static_cast<Int>(v.size()), 0);
        std::copy(v.begin(), v.end(), ptr);
    }
}

// Owner side of the frozen struct from the donor tuples: K_l, R_l and G_{l,r}
// (eqs. (4), (6), (60)) and the receive lists (eq. (62)): recvnb in increasing
// donor rank; for each donor, the own faces l with that rank in R_l, ordered
// by increasing own global face index (list order L_pr). ownface[l] = l
// (index into mesh.intfaces). Host arrays.
template <class T, class I>
inline void build_owner_side(nonmatchinginterfaceT<T, I>& nmi, const DonorSearchResult<T, I>& res)
{
    const I nf = res.nfaces;
    nmi.nownfaces = nf;
    nmi.isowner = (nf > 0) ? 1 : 0;

    std::vector<I> ownface(static_cast<std::size_t>(nf));
    for (I l = 0; l < nf; l++) ownface[l] = l;
    copy_to_host_array(nmi.ownface, ownface);
    copy_to_host_array(nmi.ownfaceglobal, res.ownglobal);

    // R_l per face, eq. (60).
    std::vector<std::vector<I>> R(static_cast<std::size_t>(nf));
    std::vector<I> allranks;
    for (I l = 0; l < nf; l++) {
        I begin = 0, end = 0;
        face_point_range(res, l, begin, end);
        for (I p = begin; p < end; p++) R[l].push_back(res.r[p]);
        std::sort(R[l].begin(), R[l].end());
        R[l].erase(std::unique(R[l].begin(), R[l].end()), R[l].end());
        allranks.insert(allranks.end(), R[l].begin(), R[l].end());
    }
    std::sort(allranks.begin(), allranks.end());
    allranks.erase(std::unique(allranks.begin(), allranks.end()), allranks.end());

    // Own faces in increasing global index (list order of eq. (62)).
    std::vector<I> order(static_cast<std::size_t>(nf));
    for (I l = 0; l < nf; l++) order[l] = l;
    std::sort(order.begin(), order.end(),
            [&](I u, I v) { return res.ownglobal[u] < res.ownglobal[v]; });

    std::vector<I> recvcount, recvface;
    for (I rnk : allranks) {
        I count = 0;
        for (I l : order)
            if (std::binary_search(R[l].begin(), R[l].end(), rnk)) {
                recvface.push_back(l);
                count++;
            }
        recvcount.push_back(count);
    }
    nmi.nrecvnb = static_cast<I>(allranks.size());
    copy_to_host_array(nmi.recvnb, allranks);
    copy_to_host_array(nmi.recvcount, recvcount);
    copy_to_host_array(nmi.recvface, recvface);
}

// |K_l| (eqs. (4), (6)) and |R_l| (eq. (60)) of own face l.
template <class T, class I>
inline void face_set_sizes(I& nK, I& nR, const DonorSearchResult<T, I>& res, I l)
{
    std::vector<I> K, R;
    I begin = 0, end = 0;
    face_point_range(res, l, begin, end);
    for (I p = begin; p < end; p++) {
        K.push_back(res.f[p]);
        R.push_back(res.r[p]);
    }
    std::sort(K.begin(), K.end());
    std::sort(R.begin(), R.end());
    nK = static_cast<I>(std::unique(K.begin(), K.end()) - K.begin());
    nR = static_cast<I>(std::unique(R.begin(), R.end()) - R.begin());
}

template <class T, class I>
inline void free_nonmatching_interface(nonmatchinginterfaceT<T, I>& nmi)
{
    CPUFREE(nmi.ownface);
    CPUFREE(nmi.ownfaceglobal);
    CPUFREE(nmi.recvnb);
    CPUFREE(nmi.recvcount);
    CPUFREE(nmi.recvface);
    CPUFREE(nmi.pe);
    CPUFREE(nmi.pa);
    CPUFREE(nmi.pf);
    CPUFREE(nmi.pslot);
    CPUFREE(nmi.pg);
    CPUFREE(nmi.px);
    CPUFREE(nmi.pnl);
    CPUFREE(nmi.pwJ);
    CPUFREE(nmi.pxi);
    CPUFREE(nmi.pzeta);
    CPUFREE(nmi.pphi);
    CPUFREE(nmi.pphibar);
    CPUFREE(nmi.ppsi);
    CPUFREE(nmi.pd);
    CPUFREE(nmi.pdphibar);
    CPUFREE(nmi.slotrank);
    CPUFREE(nmi.slotface);
    CPUFREE(nmi.slotptr);
    CPUFREE(nmi.pairslot);
    CPUFREE(nmi.pairelem);
    CPUFREE(nmi.pairptr);
    CPUFREE(nmi.sendnb);
    CPUFREE(nmi.sendcount);
    CPUFREE(nmi.Ri);
    CPUFREE(nmi.Hi);
    CPUFREE(nmi.sendbuf);
    CPUFREE(nmi.recvbuf);
    nmi = nonmatchinginterfaceT<T, I>();
}

// Releases all host arrays allocated by the Phase 2-3 setup.
template <class T, class I>
inline void free_nonmatching_data(nonmatchingdataT<T, I>& nm)
{
    for (int k = 0; k < 2; k++)
        free_nonmatching_interface(nm.dir[k]);
    nm.interfacetype = 0; // [nonmatching] leave a fully inactive cache after release
}

#ifdef HAVE_MPI
// All coupled interface faces of all ranks of comm (Phase 7: comm_Gamma, the
// exhaustive oracle only), gathered with MPI_Allgatherv. Every rank of comm calls it.
template <class T, class I>
inline InterfaceFaceSet<T, I> gather_interface_faces(const InterfaceFaceSet<T, I>& own, MPI_Comm comm)
{
    const I nd = own.nd, npf = own.npf, nlocal = own.size();
    const I nmeta = 5, ncoord = npf * nd;
    int commSize = 0;
    MPI_Comm_size(comm, &commSize);
    const I mpiprocs = static_cast<I>(commSize);

    std::vector<I> counts(static_cast<std::size_t>(mpiprocs), 0);
    MPI_Allgather(&nlocal, 1, mpi_type<I>(), counts.data(), 1, mpi_type<I>(), comm);

    // Counts and displacements in a wide integer: every count, every displacement and
    // the final extent (displacement + count of the last rank) must fit in int, the
    // MPI_Allgatherv argument type (review F5-R1). Buffer sizes use std::size_t.
    const long long intmax = std::numeric_limits<int>::max();
    std::vector<int> mc(mpiprocs, 0), md(mpiprocs, 0), cc(mpiprocs, 0), cd(mpiprocs, 0);
    long long mext = 0, cext = 0, ltotal = 0; // running extents = next displacements
    for (I r = 0; r < mpiprocs; r++) {
        const long long n = static_cast<long long>(counts[r]);
        const long long nm = n * nmeta, ncr = n * static_cast<long long>(ncoord);
        if (n < 0 || nm > intmax || ncr > intmax || mext + nm > intmax || cext + ncr > intmax)
            error("too many non-matching interface faces for MPI_Allgatherv integer counts/displacements");
        md[r] = static_cast<int>(mext);
        cd[r] = static_cast<int>(cext);
        mc[r] = static_cast<int>(nm);
        cc[r] = static_cast<int>(ncr);
        mext += nm;
        cext += ncr;
        ltotal += n;
    }
    const I total = static_cast<I>(ltotal); // ltotal <= mext <= INT_MAX

    std::vector<I> meta(static_cast<std::size_t>(nmeta) * static_cast<std::size_t>(nlocal));
    for (I i = 0; i < nlocal; i++) {
        meta[nmeta * i + 0] = own.rank[i];
        meta[nmeta * i + 1] = own.model[i];
        meta[nmeta * i + 2] = own.elem[i];
        meta[nmeta * i + 3] = own.lface[i];
        meta[nmeta * i + 4] = own.global[i];
    }
    std::vector<I> allmeta(static_cast<std::size_t>(mext));
    InterfaceFaceSet<T, I> all;
    all.nd = nd;
    all.npf = npf;
    all.nodes.assign(static_cast<std::size_t>(cext), static_cast<T>(0));

    MPI_Allgatherv(meta.data(), static_cast<int>(meta.size()), mpi_type<I>(),
            allmeta.data(), mc.data(), md.data(), mpi_type<I>(), comm);
    MPI_Allgatherv(own.nodes.data(), static_cast<int>(own.nodes.size()), mpi_type<T>(),
            all.nodes.data(), cc.data(), cd.data(), mpi_type<T>(), comm);

    for (I j = 0; j < total; j++) {
        const std::size_t o = static_cast<std::size_t>(nmeta) * static_cast<std::size_t>(j);
        all.rank.push_back(allmeta[o + 0]);
        all.model.push_back(allmeta[o + 1]);
        all.elem.push_back(allmeta[o + 2]);
        all.lface.push_back(allmeta[o + 3]);
        all.global.push_back(allmeta[o + 4]);
    }
    return all;
}

// Faces of the given model only.
template <class T, class I>
inline InterfaceFaceSet<T, I> select_model_faces(const InterfaceFaceSet<T, I>& all, I model)
{
    InterfaceFaceSet<T, I> out;
    out.nd = all.nd;
    out.npf = all.npf;
    const std::size_t nc = static_cast<std::size_t>(all.npf * all.nd);
    for (I j = 0; j < all.size(); j++) {
        if (all.model[j] != model) continue;
        out.rank.push_back(all.rank[j]);
        out.model.push_back(all.model[j]);
        out.elem.push_back(all.elem[j]);
        out.lface.push_back(all.lface[j]);
        out.global.push_back(all.global[j]);
        out.nodes.insert(out.nodes.end(), all.nodes.begin() + nc * j, all.nodes.begin() + nc * (j + 1));
    }
    return out;
}

} // namespace nonmatching
} // namespace exasim

#include "nonmatchingsearch.hpp" // [nonmatching] Phase 7: distributed donor search (Steps D1-D5)

namespace exasim {
namespace nonmatching {

// Recompute the owner face geometry needed by Step D6.  This is the same
// node ordering and 2D formula as ElemFaceGeomBlock/FaceGeom2D, but it runs
// during cpuInit, before sol.elemfaceg is constructed in compGeometry.
template <class T, class I>
inline void owner_point_geometry(std::vector<T>& wJ, std::vector<T>& normal,
        const DonorSearchResult<T, I>& res, const InterfaceFaceSet<T, I>& own,
        const T* shapfgt, const T* gwf, const T* xpf = nullptr)
{
    if (res.nd != 2 || own.nd != 2 || shapfgt == nullptr || gwf == nullptr)
        error("non-matching setup: owner geometry currently supports nd = 2 only");
    const I npf = own.npf, ngf = res.ngf, npts = res.npts;
    wJ.assign(static_cast<std::size_t>(npts), static_cast<T>(0));
    normal.assign(static_cast<std::size_t>(npts) * 2, static_cast<T>(0));
    if (!res.facePointPtr.empty()) {
        // [nonmatching] Phase 9 A1: composite points. Uniform faces use the expressions
        // of the uniform branch below (bitwise); pieces use the Lagrange basis at s.
        if (!xpf) error("non-matching setup: composite owner geometry needs master.xpf");
        std::vector<T> psi(static_cast<std::size_t>(npf)), dpsiS(static_cast<std::size_t>(npf));
        for (I p = 0; p < npts; p++) {
            const I l = res.pointFace[p], g = res.pointGauss[p];
            const std::size_t off = static_cast<std::size_t>(npf * 2 * l);
            T tx = 0, ty = 0, jac = 0, weight = 0;
            if (res.pointUniform[p]) {
                for (I j = 0; j < npf; j++) {
                    const T dpsi = shapfgt[ngf * npf + g + ngf * j];
                    tx += dpsi * own.nodes[off + j];
                    ty += dpsi * own.nodes[off + j + npf];
                }
                jac = std::sqrt(tx * tx + ty * ty);
                weight = gwf[g] * jac;
            }
            else {
                face_lagrange_1d(psi.data(), dpsiS.data(), xpf, npf, res.pointS[p]);
                for (I j = 0; j < npf; j++) {
                    tx += dpsiS[j] * own.nodes[off + j];
                    ty += dpsiS[j] * own.nodes[off + j + npf];
                }
                jac = std::sqrt(tx * tx + ty * ty);
                weight = gwf[g] * res.pointLen[p] * jac;
            }
            if (!(jac > 0))
                error("non-matching setup: singular owner face geometry");
            wJ[p] = weight;
            normal[p] = ty / jac;
            normal[p + npts] = -tx / jac;
        }
        return;
    }
    for (I l = 0; l < res.nfaces; l++)
        for (I g = 0; g < ngf; g++) {
            T tx = 0, ty = 0;
            for (I j = 0; j < npf; j++) {
                const T dpsi = shapfgt[ngf * npf + g + ngf * j];
                const std::size_t off = static_cast<std::size_t>(npf * 2 * l);
                tx += dpsi * own.nodes[off + j];
                ty += dpsi * own.nodes[off + j + npf];
            }
            const T jac = std::sqrt(tx * tx + ty * ty);
            if (!(jac > 0))
                error("non-matching setup: singular owner face geometry");
            const I p = g + ngf * l;
            wJ[p] = gwf[g] * jac;
            normal[p] = ty / jac;
            normal[p + npts] = -tx / jac;
        }
}

template <class I>
inline I checked_setup_size(std::initializer_list<I> factors, const char* what)
{
    long long value = 1;
    for (I factor : factors) {
        if (factor < 0 || (factor != 0 && value > std::numeric_limits<int>::max() / factor)) {
            std::ostringstream oss;
            oss << "non-matching setup: " << what << " exceeds integer allocation limits";
            error(oss.str());
        }
        value *= static_cast<long long>(factor);
    }
    return static_cast<I>(value);
}

template <class T, class I>
inline void allocate_zeroed(T*& ptr, I& storedSize, I size)
{
    if (ptr) CPUFREE(ptr);
    ptr = nullptr;
    storedSize = size;
    if (size > 0) {
        TemplateMalloc(&ptr, size, 0);
        std::fill(ptr, ptr + size, static_cast<T>(0));
    }
}

// Physical scale for the inverse-map forward error.  |x| accounts for
// cancellation on translated meshes, while the donor-element extent keeps the
// tolerance relative on meshes centered near the origin.
template <class T, class I>
inline T inverse_map_scale(const T* x, const T* elementNodes, I nd, I npe)
{
    T scale = 0;
    for (I d = 0; d < nd; ++d) {
        T xmin = elementNodes[static_cast<std::size_t>(npe) * d];
        T xmax = xmin;
        scale = std::max(scale, std::abs(x[d]));
        for (I j = 1; j < npe; ++j) {
            const T value = elementNodes[j + static_cast<std::size_t>(npe) * d];
            xmin = std::min(xmin, value);
            xmax = std::max(xmax, value);
        }
        scale = std::max(scale, xmax - xmin);
    }
    return std::max(scale, std::numeric_limits<T>::min());
}

// Steps D6-D7, eqs. (61)-(62).  xi is sent with the Phase 2 donor tuple, so
// the donor starts at chi_a(xi) and does not repeat closest-face projection.
template <class T, class I>
inline void distribute_setup_data(nonmatchingdataT<T, I>& nm,
        DonorSearchResult<T, I>& res, const InterfaceFaceSet<T, I>& own,
        appstructT<T, I>& app, masterstructT<T, I>& master,
        meshstructT<T, I>& mesh, solstructT<T, I>& sol,
        I mpiprocs, I mpirank, I ownDirection, const DonorSearchParams& prm,
        const InterfaceComm<I>& ic, CompositeOwnerQuadrature<T, I>* quad = nullptr)
{
    const I nd = res.nd, ndf = nd - 1, npe = master.ndims[5], npf = master.ndims[6];
    const I nfe = mesh.ndims[4], elemtype = master.ndims[1], porder = master.ndims[3];
    const I ncx = app.ndims[AppNdims::ncx];
    const I donorDirection = 1 - ownDirection;
    SimplexNodalBasis2D simplexBasis;
    const SimplexNodalBasis2D* simplexBasisPtr = nullptr;
    if (UsesPolynomialSimplexBasis(nd, elemtype)) {
        try {
            simplexBasis.initialize(master.xpe, npe, porder);
        }
        catch (const std::runtime_error& exception) {
            std::ostringstream oss;
            oss << "non-matching setup: simplex nodal basis initialization failed on rank "
                << mpirank << " (direction " << donorDirection << ", elemtype "
                << elemtype << ", porder " << porder << "): " << exception.what();
            error(oss.str());
        }
        simplexBasisPtr = &simplexBasis;
    }

    std::vector<T> ownerWJ, ownerNormal;
    owner_point_geometry(ownerWJ, ownerNormal, res, own, master.shapfgt, master.gwf, master.xpf);
    // [nonmatching] Phase 10: mixed-family messages carry the owner's face
    // test values. Phase 9 composite messages additionally carry piece and s.
    // Same-family, A1-off messages retain their original byte layout.
    const bool compositeMetadata = quad && (quad->dir[0].applied || quad->dir[1].applied);
    const bool ownerTestsRequired = quad && (quad->dir[0].ownerTests || quad->dir[1].ownerTests);
    const bool ownComposite = !res.facePointPtr.empty();

    // Phase 7: comm_Gamma and its rank maps come from the caller (one split per
    // rebuild); nonmembers do not call this function.
    if (ic.comm == MPI_COMM_NULL)
        error("non-matching setup: D6-D7 called without the interface communicator");
    const MPI_Comm interfaceComm = ic.comm;
    const int interfaceSize = ic.size;
    const std::vector<I>& interfaceWorld = ic.world;
    const std::vector<int>& worldToInterface = ic.worldToComm;
    if (static_cast<I>(worldToInterface.size()) != mpiprocs)
        error("non-matching setup: interface rank map has the wrong size");

    std::vector<int> sendPoints(static_cast<std::size_t>(interfaceSize), 0);
    for (I p = 0; p < res.npts; p++) {
        if (res.r[p] < 0 || res.r[p] >= mpiprocs || worldToInterface[res.r[p]] < 0)
            error("non-matching setup: donor rank does not own interface faces");
        int& count = sendPoints[worldToInterface[res.r[p]]];
        if (count == std::numeric_limits<int>::max())
            error("non-matching setup: MPI point count exceeds integer limits");
        count++;
    }
    std::vector<int> recvPoints(static_cast<std::size_t>(interfaceSize), 0);
    MPI_Alltoall(sendPoints.data(), 1, MPI_INT, recvPoints.data(), 1, MPI_INT, interfaceComm);
    ic.collective();

    auto make_displacements = [](const std::vector<int>& counts, std::vector<int>& displs) {
        long long extent = 0;
        displs.resize(counts.size());
        for (std::size_t q = 0; q < counts.size(); q++) {
            if (counts[q] < 0 || extent > std::numeric_limits<int>::max() - counts[q])
                error("non-matching setup: MPI point counts exceed integer limits");
            displs[q] = static_cast<int>(extent);
            extent += counts[q];
        }
        return static_cast<int>(extent);
    };
    std::vector<int> sendDisp, recvDisp;
    const int nsend = make_displacements(sendPoints, sendDisp);
    const int nrecv = make_displacements(recvPoints, recvDisp);

    // owner rank, owner face, g, donor e, donor a, donor f [, piece]
    const int nmeta = compositeMetadata ? 7 : 6;
    // x, n, wJ, xi: nd + nd + 1 + (nd-1) [, s, psi_0..psi_{npf-1}]
    const int nreal = 3 * static_cast<int>(nd) + (compositeMetadata ? 1 : 0) +
                      (ownerTestsRequired ? static_cast<int>(npf) : 0);
    auto scaled_mpi_int = [](int value, int scale, const char* what) {
        const long long result = static_cast<long long>(value) * scale;
        if (value < 0 || scale < 0 || result > std::numeric_limits<int>::max()) {
            std::ostringstream oss;
            oss << "non-matching setup: " << what << " exceeds MPI integer limits";
            error(oss.str());
        }
        return static_cast<int>(result);
    };
    const int nsendMeta = scaled_mpi_int(nsend, nmeta, "metadata send extent");
    const int nsendReal = scaled_mpi_int(nsend, nreal, "coordinate send extent");
    const int nrecvMeta = scaled_mpi_int(nrecv, nmeta, "metadata receive extent");
    const int nrecvReal = scaled_mpi_int(nrecv, nreal, "coordinate receive extent");
    std::vector<I> sendMeta(static_cast<std::size_t>(nsendMeta));
    std::vector<T> sendReal(static_cast<std::size_t>(nsendReal));
    std::vector<int> cursor = sendDisp;
    for (I p = 0; p < res.npts; p++) {
        I l = 0, g = 0, piece = 0;
        point_face_gauss(res, p, l, g, piece);
        const int q = worldToInterface[res.r[p]];
        const int k = cursor[q]++;
        I* im = &sendMeta[static_cast<std::size_t>(nmeta * k)];
        im[0] = mpirank; im[1] = res.ownglobal[l]; im[2] = g;
        im[3] = res.e[p]; im[4] = res.a[p]; im[5] = res.f[p];
        T* rm = &sendReal[static_cast<std::size_t>(nreal * k)];
        for (I d = 0; d < nd; d++)
            rm[d] = res.x[static_cast<std::size_t>(p) + static_cast<std::size_t>(res.npts) * d];
        for (I d = 0; d < nd; d++)
            rm[nd + d] = ownerNormal[static_cast<std::size_t>(p) + static_cast<std::size_t>(res.npts) * d];
        rm[2 * nd] = ownerWJ[p];
        for (I d = 0; d < ndf; d++)
            rm[2 * nd + 1 + d] = res.xi[static_cast<std::size_t>(p) +
                                        static_cast<std::size_t>(res.npts) * d];
        if (compositeMetadata) {
            im[6] = piece;
            rm[3 * nd] = ownComposite ? res.pointS[p] :
                    (master.gpf ? master.gpf[g] : static_cast<T>(0));
        }
        if (ownerTestsRequired) {
            T* xm = &rm[3 * nd + (compositeMetadata ? 1 : 0)];
            if (ownComposite) {
                for (I i = 0; i < npf; i++)
                    xm[i] = res.pointTest[static_cast<std::size_t>(i) + static_cast<std::size_t>(npf) * p];
            }
            else {
                for (I i = 0; i < npf; i++)
                    xm[i] = master.shapfgt[g + static_cast<std::size_t>(res.ngf) * i];
            }
        }
    }

    std::vector<int> smc(interfaceSize), smd(interfaceSize), rmc(interfaceSize), rmd(interfaceSize);
    std::vector<int> src(interfaceSize), srd(interfaceSize), rrc(interfaceSize), rrd(interfaceSize);
    for (int q = 0; q < interfaceSize; q++) {
        smc[q] = scaled_mpi_int(sendPoints[q], nmeta, "metadata send count");
        smd[q] = scaled_mpi_int(sendDisp[q], nmeta, "metadata send displacement");
        rmc[q] = scaled_mpi_int(recvPoints[q], nmeta, "metadata receive count");
        rmd[q] = scaled_mpi_int(recvDisp[q], nmeta, "metadata receive displacement");
        src[q] = scaled_mpi_int(sendPoints[q], nreal, "coordinate send count");
        srd[q] = scaled_mpi_int(sendDisp[q], nreal, "coordinate send displacement");
        rrc[q] = scaled_mpi_int(recvPoints[q], nreal, "coordinate receive count");
        rrd[q] = scaled_mpi_int(recvDisp[q], nreal, "coordinate receive displacement");
    }
    std::vector<I> recvMeta(static_cast<std::size_t>(nrecvMeta));
    std::vector<T> recvReal(static_cast<std::size_t>(nrecvReal));
    MPI_Alltoallv(sendMeta.data(), smc.data(), smd.data(), mpi_type<I>(),
            recvMeta.data(), rmc.data(), rmd.data(), mpi_type<I>(), interfaceComm);
    ic.collective();
    MPI_Alltoallv(sendReal.data(), src.data(), srd.data(), mpi_type<T>(),
            recvReal.data(), rrc.data(), rrd.data(), mpi_type<T>(), interfaceComm);
    ic.collective();

    std::vector<I> order(static_cast<std::size_t>(nrecv));
    for (int p = 0; p < nrecv; p++) order[p] = static_cast<I>(p);
    std::sort(order.begin(), order.end(), [&](I u, I v) {
        const I* a = &recvMeta[static_cast<std::size_t>(nmeta * u)];
        const I* b = &recvMeta[static_cast<std::size_t>(nmeta * v)];
        if (compositeMetadata) // [nonmatching] Phase 9 A1: piece before the Gauss index
            return std::tie(a[0], a[1], a[3], a[6], a[2], a[5]) <
                   std::tie(b[0], b[1], b[3], b[6], b[2], b[5]);
        return std::tie(a[0], a[1], a[3], a[2], a[5]) <
               std::tie(b[0], b[1], b[3], b[2], b[5]);
    });

    nonmatchinginterfaceT<T, I>& donor = nm.dir[donorDirection];
    donor.ncu = app.ndims[AppNdims::ncu];
    donor.ncq = app.ndims[AppNdims::ncq];
    donor.npts = static_cast<I>(nrecv);
    donor.isdonor = nrecv > 0 ? 1 : 0;

    const I nr = static_cast<I>(nrecv);
    const I nrNd = checked_setup_size<I>({nr, nd}, "point coordinates");
    const I nrNdf = checked_setup_size<I>({nr, ndf}, "face coordinates");
    const I nrNpe = checked_setup_size<I>({nr, npe}, "volume basis data");
    const I nrNpf = checked_setup_size<I>({nr, npf}, "face basis data");
    const I nrNdNpe = checked_setup_size<I>({nr, nd, npe}, "volume-basis gradients");
    std::vector<I> pe(static_cast<std::size_t>(nr)), pa(static_cast<std::size_t>(nr));
    std::vector<I> pf(static_cast<std::size_t>(nr)), pslot(static_cast<std::size_t>(nr));
    std::vector<I> pg(static_cast<std::size_t>(nr));
    std::vector<T> px(static_cast<std::size_t>(nrNd));
    std::vector<T> pnl(static_cast<std::size_t>(nrNd));
    std::vector<T> pwJ(static_cast<std::size_t>(nr));
    std::vector<T> pxi(static_cast<std::size_t>(nrNdf));
    std::vector<T> pzeta(static_cast<std::size_t>(nrNd));
    std::vector<T> pphi(static_cast<std::size_t>(nrNpe));
    std::vector<T> pphibar(static_cast<std::size_t>(nrNpe));
    std::vector<T> ppsi(static_cast<std::size_t>(nrNpf));
    std::vector<T> pd(static_cast<std::size_t>(nrNd));
    std::vector<T> pdphibar;
    if (donor.ncq == 0) pdphibar.resize(static_cast<std::size_t>(nrNdNpe));
    std::vector<I> slotrank, slotface, slotptr(1, 0), pairslot, pairelem, pairptr(1, 0);

    // [nonmatching] Phase 9 A1: donor-side sidecar data of an applied direction.
    const bool donorComposite = compositeMetadata && quad->dir[donorDirection].applied;
    const bool donorOwnerTests = ownerTestsRequired && quad->dir[donorDirection].ownerTests;
    std::vector<T> sideS, sideTest;
    std::vector<I> sidePiece;
    if (donorComposite) {
        sideS.assign(static_cast<std::size_t>(nr), static_cast<T>(0));
        sidePiece.assign(static_cast<std::size_t>(nr), 0);
    }
    if (donorOwnerTests) {
        sideTest.assign(static_cast<std::size_t>(nrNpf), static_cast<T>(0));
    }

    I currentSlot = -1, lastRank = -1, lastFace = -1, lastElem = -1;
    const I ne = mesh.ndims[1];
    for (I p = 0; p < static_cast<I>(nrecv); p++) {
        const I raw = order[p];
        const I* im = &recvMeta[static_cast<std::size_t>(nmeta * raw)];
        const T* rm = &recvReal[static_cast<std::size_t>(nreal * raw)];
        if (im[3] < 0 || im[3] >= ne || im[4] < 0 || im[4] >= nfe ||
            im[2] < 0 || im[2] >= donor.ngf)
            error("non-matching setup: received donor tuple is out of range");
        if (p == 0 || im[0] != lastRank || im[1] != lastFace) {
            if (p > 0) slotptr.push_back(p);
            slotrank.push_back(im[0]);
            slotface.push_back(im[1]);
            currentSlot++;
            lastRank = im[0]; lastFace = im[1]; lastElem = -1;
        }
        if (p == 0 || im[3] != lastElem || pairslot.empty() || pairslot.back() != currentSlot) {
            if (!pairslot.empty()) pairptr.push_back(p);
            pairslot.push_back(currentSlot);
            pairelem.push_back(im[3]);
            lastElem = im[3];
        }
        pe[p] = im[3]; pa[p] = im[4]; pf[p] = im[5]; pslot[p] = currentSlot; pg[p] = im[2];
        for (I d = 0; d < nd; d++) {
            const std::size_t index = static_cast<std::size_t>(p) + static_cast<std::size_t>(nrecv) * d;
            px[index] = rm[d];
            pnl[index] = rm[nd + d];
        }
        pwJ[p] = rm[2 * nd];
        for (I d = 0; d < ndf; d++)
            pxi[static_cast<std::size_t>(p) + static_cast<std::size_t>(nrecv) * d] = rm[2 * nd + 1 + d];
        if (donorComposite) {
            const T* xm = &rm[3 * nd];
            sideS[p] = xm[0];
            sidePiece[p] = im[6];
        }
        if (donorOwnerTests) {
            const T* xm = &rm[3 * nd + (compositeMetadata ? 1 : 0)];
            for (I i = 0; i < npf; i++)
                sideTest[static_cast<std::size_t>(i) + static_cast<std::size_t>(npf) * p] = xm[i];
        }

        std::vector<T> elementNodes(static_cast<std::size_t>(npe * nd));
        for (I d = 0; d < nd; d++)
            for (I j = 0; j < npe; j++)
                elementNodes[j + npe * d] = sol.xdg[j + npe * d + npe * ncx * pe[p]];
        std::vector<T> psi(static_cast<std::size_t>(npf)), zetaBar(static_cast<std::size_t>(nd));
        FaceToElementReference(zetaBar.data(), psi.data(), &pxi[p], master.xpe, master.xpf,
                mesh.perm, pa[p], nd, npe, npf, elemtype, porder);
        std::vector<T> faceNodes(static_cast<std::size_t>(npf * nd));
        for (I j = 0; j < npf; j++)
            for (I d = 0; d < nd; d++)
                faceNodes[j + npf * d] = elementNodes[mesh.perm[j + npf * pa[p]] + npe * d];
        T xbar[3] = {0, 0, 0};
        for (I d = 0; d < nd; d++)
            for (I j = 0; j < npf; j++) xbar[d] += psi[j] * faceNodes[j + npf * d];
        T xpoint[3] = {0, 0, 0};
        for (I d = 0; d < nd; d++)
            xpoint[d] = px[static_cast<std::size_t>(p) + static_cast<std::size_t>(nrecv) * d];
        const T inverseScale = inverse_map_scale(xpoint, elementNodes.data(), nd, npe);
        const T inverseTolerance = static_cast<T>(prm.inverseToleranceFactor) *
                std::numeric_limits<T>::epsilon() * inverseScale;
        if (!(inverseTolerance > 0))
            error("non-matching setup: invalid scale-invariant inverse tolerance");
        T inverseResidual = std::numeric_limits<T>::max();
        std::vector<T> zeta(static_cast<std::size_t>(nd));
        InverseMapStatus inverse = InverseElementMap(zeta.data(), &inverseResidual, xpoint,
                elementNodes.data(), master.xpe, zetaBar.data(), nd, npe, elemtype, porder,
                prm.maxInverseIterations, prm.maxProjectedInverseIterations,
                inverseTolerance, prm.inverseProjectionMargin, simplexBasisPtr);
        if (inverse == InverseMapStatus::Failed) {
            T alternate[3] = {0, 0, 0};
            DefaultReferenceGuess(alternate, nd, elemtype);
            inverse = InverseElementMap(zeta.data(), &inverseResidual, xpoint,
                    elementNodes.data(), master.xpe, alternate, nd, npe, elemtype, porder,
                    prm.maxInverseIterations, prm.maxProjectedInverseIterations,
                    inverseTolerance, prm.inverseProjectionMargin, simplexBasisPtr);
        }
        if (inverse == InverseMapStatus::Failed) {
            std::ostringstream oss;
            oss << "non-matching setup: inverse donor element map failed on rank " << mpirank
                << ", owner rank " << im[0] << ", owner face " << im[1]
                << ", quadrature point " << im[2] << ", donor element " << pe[p]
                << ", donor face " << pf[p] << ", x = (" << xpoint[0] << ", " << xpoint[1]
                << "), zetabar = (" << zetaBar[0] << ", " << zetaBar[1]
                << "), zeta = (" << zeta[0] << ", " << zeta[1]
                << "), final residual = " << inverseResidual;
            error(oss.str());
        }
        res.inversePoints[donorDirection]++;
        if (inverse == InverseMapStatus::ConvergedProjected)
            res.projectedExactInversePoints[donorDirection]++;
        else if (inverse == InverseMapStatus::ProjectedInexact)
            res.projectedInexactInversePoints[donorDirection]++;
        res.maxInverseResidual[donorDirection] =
                std::max(res.maxInverseResidual[donorDirection], inverseResidual);
        res.maxRelativeInverseResidual[donorDirection] =
                std::max(res.maxRelativeInverseResidual[donorDirection], inverseResidual / inverseScale);
        T projectedReference[3] = {0, 0, 0};
        T projectionWork[3] = {0, 0, 0};
        const T referenceDistance = ReferenceElementDistance(
                zeta.data(), projectedReference, projectionWork, nd, elemtype);
        res.maxReferenceExtrapolation[donorDirection] =
                std::max(res.maxReferenceExtrapolation[donorDirection], referenceDistance);
        EvaluateNonmatchingVolumeShape(
                &pphi[static_cast<std::size_t>(npe) * p], zeta.data(), master.xpe,
                nd, npe, elemtype, porder, simplexBasisPtr);
        EvaluateNonmatchingVolumeShape(
                &pphibar[static_cast<std::size_t>(npe) * p], zetaBar.data(), master.xpe,
                nd, npe, elemtype, porder, simplexBasisPtr);
        for (I j = 0; j < npf; j++) ppsi[j + static_cast<std::size_t>(npf) * p] = psi[j];
        for (I d = 0; d < nd; d++) {
            const std::size_t index = static_cast<std::size_t>(p) + static_cast<std::size_t>(nrecv) * d;
            pzeta[index] = zeta[d];
            pd[index] = px[index] - xbar[d];
        }

        if (donor.ncq == 0) {
            std::vector<T> shape(static_cast<std::size_t>(npe)), dshape(static_cast<std::size_t>(npe * nd));
            T mapped[3] = {0, 0, 0}, jacobian[9] = {0};
            MapNonmatchingReferenceToPhysical(
                    mapped, jacobian, shape.data(), dshape.data(), zetaBar.data(),
                    elementNodes.data(), master.xpe, nd, npe, elemtype, porder,
                    simplexBasisPtr);
            T transposed[9] = {0};
            for (I k = 0; k < nd; k++)
                for (I d = 0; d < nd; d++) transposed[k * nd + d] = jacobian[d * nd + k];
            for (I j = 0; j < npe; j++) {
                T rhs[3] = {0, 0, 0}, grad[3] = {0, 0, 0};
                for (I k = 0; k < nd; k++) rhs[k] = dshape[k * npe + j];
                if (!SolveScaleInvariantSmallSystem(grad, transposed, rhs, nd))
                    error("non-matching setup: singular donor element map at closest point");
                for (I d = 0; d < nd; d++)
                    pdphibar[j + static_cast<std::size_t>(npe) *
                                   (d + static_cast<std::size_t>(nd) * p)] = grad[d];
            }
        }
    }
    if (nrecv > 0) { slotptr.push_back(static_cast<I>(nrecv)); pairptr.push_back(static_cast<I>(nrecv)); }

    donor.nslots = static_cast<I>(slotrank.size());
    donor.npairs = static_cast<I>(pairslot.size());
    std::vector<I> sendnb, sendcount;
    for (I s = 0; s < donor.nslots; s++) {
        if (s == 0 || slotrank[s] != slotrank[s - 1]) {
            sendnb.push_back(slotrank[s]);
            sendcount.push_back(0);
        }
        sendcount.back()++;
    }
    donor.nsendnb = static_cast<I>(sendnb.size());

    copy_to_host_array(donor.pe, pe); copy_to_host_array(donor.pa, pa);
    copy_to_host_array(donor.pf, pf); copy_to_host_array(donor.pslot, pslot);
    copy_to_host_array(donor.pg, pg); copy_to_host_scalar_array(donor.px, px);
    copy_to_host_scalar_array(donor.pnl, pnl); copy_to_host_scalar_array(donor.pwJ, pwJ);
    copy_to_host_scalar_array(donor.pxi, pxi); copy_to_host_scalar_array(donor.pzeta, pzeta);
    copy_to_host_scalar_array(donor.pphi, pphi); copy_to_host_scalar_array(donor.pphibar, pphibar);
    copy_to_host_scalar_array(donor.ppsi, ppsi); copy_to_host_scalar_array(donor.pd, pd);
    copy_to_host_scalar_array(donor.pdphibar, pdphibar);
    copy_to_host_array(donor.slotrank, slotrank); copy_to_host_array(donor.slotface, slotface);
    copy_to_host_array(donor.slotptr, slotptr); copy_to_host_array(donor.pairslot, pairslot);
    copy_to_host_array(donor.pairelem, pairelem); copy_to_host_array(donor.pairptr, pairptr);
    copy_to_host_array(donor.sendnb, sendnb); copy_to_host_array(donor.sendcount, sendcount);
    if (donorComposite) {
        CompositeQuadratureDirection<T, I>& side = quad->dir[donorDirection];
        side.ps = std::move(sideS);
        side.ppiece = std::move(sidePiece);
    }
    if (donorOwnerTests) {
        CompositeQuadratureDirection<T, I>& side = quad->dir[donorDirection];
        side.ptest = std::move(sideTest);
    }

    const I szRi = checked_setup_size<I>({donor.ncu12, donor.npf, donor.nslots}, "Ri");
    const I szHi = checked_setup_size<I>({donor.ncu12, donor.npf, donor.ncu, donor.npf,
                                          donor.nfe, donor.npairs}, "Hi");
    allocate_zeroed(donor.Ri, donor.szRi, szRi);
    allocate_zeroed(donor.Hi, donor.szHi, szHi);
    allocate_zeroed(donor.sendbuf, donor.szsendbuf, szRi);

    nonmatchinginterfaceT<T, I>& owner = nm.dir[ownDirection];
    long long nrecvslotsWide = 0;
    for (I q = 0; q < owner.nrecvnb; q++) {
        if (owner.recvcount[q] < 0 ||
            nrecvslotsWide > std::numeric_limits<int>::max() - owner.recvcount[q])
            error("non-matching setup: receive-slot count exceeds integer limits");
        nrecvslotsWide += owner.recvcount[q];
    }
    const I nrecvslots = static_cast<I>(nrecvslotsWide);
    const I szrecv = checked_setup_size<I>({owner.ncu12, owner.npf, nrecvslots}, "recvbuf");
    allocate_zeroed(owner.recvbuf, owner.szrecvbuf, szrecv);

    // Verify the Phase 2 owner lists against the independently formed donor lists.
    std::vector<int> slotSends(static_cast<std::size_t>(interfaceSize), 0),
                     slotReceives(static_cast<std::size_t>(interfaceSize), 0);
    for (I q = 0; q < donor.nsendnb; q++)
        slotSends[worldToInterface[donor.sendnb[q]]] = static_cast<int>(donor.sendcount[q]);
    MPI_Alltoall(slotSends.data(), 1, MPI_INT, slotReceives.data(), 1, MPI_INT, interfaceComm);
    ic.collective();
    for (int q = 0; q < interfaceSize; q++) {
        I expected = 0;
        for (I j = 0; j < owner.nrecvnb; j++)
            if (owner.recvnb[j] == interfaceWorld[q]) expected = owner.recvcount[j];
        if (slotReceives[q] != expected)
            error("non-matching setup: donor sendcount does not match owner recvcount");
    }

    // Counts alone are insufficient for positional assembly in Phase 4. Send
    // each donor's slotface list in L_pr order and verify it equals the
    // owner's ownfaceglobal[recvface] list for that donor (eq. (62)).
    std::vector<int> slotSendDisp, slotRecvDisp;
    const int nslotSend = make_displacements(slotSends, slotSendDisp);
    const int nslotRecv = make_displacements(slotReceives, slotRecvDisp);
    std::vector<I> slotFaceSend(static_cast<std::size_t>(nslotSend));
    std::vector<I> slotFaceRecv(static_cast<std::size_t>(nslotRecv));
    std::vector<int> slotCursor = slotSendDisp;
    for (I s = 0; s < donor.nslots; ++s) {
        const int q = worldToInterface[slotrank[s]];
        slotFaceSend[static_cast<std::size_t>(slotCursor[q]++)] = slotface[s];
    }
    MPI_Alltoallv(slotFaceSend.data(), slotSends.data(), slotSendDisp.data(), mpi_type<I>(),
            slotFaceRecv.data(), slotReceives.data(), slotRecvDisp.data(), mpi_type<I>(), interfaceComm);
    ic.collective();
    for (int q = 0; q < interfaceSize; ++q) {
        I ownerOffset = 0;
        bool found = false;
        for (I j = 0; j < owner.nrecvnb; ++j) {
            if (owner.recvnb[j] == interfaceWorld[q]) {
                found = true;
                break;
            }
            ownerOffset += owner.recvcount[j];
        }
        if (!found && slotReceives[q] != 0)
            error("non-matching setup: received an unexpected donor slot list");
        for (int j = 0; j < slotReceives[q]; ++j) {
            const I ownFace = owner.recvface[ownerOffset + static_cast<I>(j)];
            if (ownFace < 0 || ownFace >= owner.nownfaces ||
                slotFaceRecv[slotRecvDisp[q] + j] != owner.ownfaceglobal[ownFace])
                error("non-matching setup: donor slotface list does not match owner recvface order");
        }
    }
}

// Preserve the existing request workspace while adding capacity for the
// asymmetric non-matching donor sends and owner receives.  The workspace is
// still owned and released by commonstructT.
template <class T, class I>
inline void ensure_nonmatching_request_capacity(commonstructT<T, I>& common,
        const nonmatchingdataT<T, I>& nm)
{
    long long nsend = 0, nrecv = 0;
    for (int k = 0; k < 2; k++) {
        if (nm.dir[k].nsendnb < 0 || nm.dir[k].nrecvnb < 0)
            error("non-matching setup: negative MPI neighbor count");
        nsend += nm.dir[k].nsendnb;
        nrecv += nm.dir[k].nrecvnb;
    }
    const long long oldCapacityWide = 2LL * static_cast<long long>(common.nnbsd) +
                                      2LL * static_cast<long long>(common.nnbintf);
    const long long neededWide = 2LL * static_cast<long long>(common.nnbsd) + nsend + nrecv;
    const long long capacityWide = std::max(oldCapacityWide, neededWide);
    if (nsend < 0 || nrecv < 0 || oldCapacityWide < 0 || neededWide < 0 ||
        capacityWide > std::numeric_limits<int>::max())
        error("non-matching setup: MPI request count exceeds integer limits");
    const I capacity = static_cast<I>(capacityWide);
    if (common.requests) CPUFREE(common.requests);
    if (common.statuses) CPUFREE(common.statuses);
    common.requests = nullptr;
    common.statuses = nullptr;
    if (capacity > 0) {
        common.requests = static_cast<MPI_Request*>(std::malloc(static_cast<std::size_t>(capacity) * sizeof(MPI_Request)));
        common.statuses = static_cast<MPI_Status*>(std::malloc(static_cast<std::size_t>(capacity) * sizeof(MPI_Status)));
        if (!common.requests || !common.statuses) error("non-matching setup: MPI request allocation failed");
    }
}

// Diagnostics of Section 3.2, printed on comm_Gamma rank 0 for both directions:
// faces, points, fraction of faces with |R_l| > 1, average |R_l| and |K_l|,
// max point-to-face distance, points off their donor face. Collective.
template <class T, class I>
inline void print_donor_diagnostics(const DonorSearchResult<T, I>& res, const InterfaceComm<I>& ic,
        const DonorSearchParams& prm = DonorSearchParams())
{
    // sums[k + 6*dir]: faces, points, faces with |R_l|>1, sum |R_l|, sum |K_l|, off-face points
    double sums[12] = {0}, gsums[12] = {0};
    double maxd[2] = {0, 0}, gmaxd[2] = {0, 0};
    // Per direction: total, projected-exact, projected-inexact.
    double inverseCounts[6] = {0, 0, 0, 0, 0, 0};
    double globalInverseCounts[6] = {0, 0, 0, 0, 0, 0};
    // Per direction: maximum physical residual, relative residual, and distance from E_hat.
    double inverseMax[6] = {0, 0, 0, 0, 0, 0};
    double globalInverseMax[6] = {0, 0, 0, 0, 0, 0};
    if (res.direction == 0 || res.direction == 1) {
        double* s = &sums[6 * res.direction];
        s[0] = res.nfaces;
        s[1] = res.npts;
        for (I l = 0; l < res.nfaces; l++) {
            I nK = 0, nR = 0;
            face_set_sizes(nK, nR, res, l);
            if (nR > 1) s[2] += 1;
            s[3] += nR;
            s[4] += nK;
        }
        for (I p = 0; p < res.npts; p++) {
            maxd[res.direction] = std::max(maxd[res.direction], static_cast<double>(res.dist[p]));
            if (res.dist[p] > static_cast<T>(prm.onFaceRelTol) * res.hface[p]) s[5] += 1;
        }
    }
    for (int dir = 0; dir < 2; ++dir) {
        inverseCounts[3 * dir] = static_cast<double>(res.inversePoints[dir]);
        inverseCounts[3 * dir + 1] = static_cast<double>(res.projectedExactInversePoints[dir]);
        inverseCounts[3 * dir + 2] = static_cast<double>(res.projectedInexactInversePoints[dir]);
        inverseMax[3 * dir] = static_cast<double>(res.maxInverseResidual[dir]);
        inverseMax[3 * dir + 1] = static_cast<double>(res.maxRelativeInverseResidual[dir]);
        inverseMax[3 * dir + 2] = static_cast<double>(res.maxReferenceExtrapolation[dir]);
    }
    MPI_Allreduce(sums, gsums, 12, MPI_DOUBLE, MPI_SUM, ic.comm);
    MPI_Allreduce(maxd, gmaxd, 2, MPI_DOUBLE, MPI_MAX, ic.comm);
    MPI_Allreduce(inverseCounts, globalInverseCounts, 6, MPI_DOUBLE, MPI_SUM, ic.comm);
    MPI_Allreduce(inverseMax, globalInverseMax, 6, MPI_DOUBLE, MPI_MAX, ic.comm);
    ic.collective(); ic.collective(); ic.collective(); ic.collective();
    if (ic.rank == 0) {
        for (int dir = 0; dir < 2; dir++) {
            const double* s = &gsums[6 * dir];
            const double nf = std::max(1.0, s[0]);
            printf("Non-matching donor search, Gamma^%d (owners in model %d): %.0f faces, %.0f points; "
                   "fraction |R_l|>1 = %.6f, average |R_l| = %.6f, average |K_l| = %.6f, "
                   "max distance = %.6e, points off donor face = %.0f\n",
                   dir + 1, dir + 1, s[0], s[1], s[2] / nf, s[3] / nf, s[4] / nf, gmaxd[dir], s[5]);
            if (globalInverseCounts[3 * dir] > 0)
                printf("  inverse maps: %.0f points, %.0f projected exact, %.0f projected inexact; "
                       "max residual = %.6e, max relative residual = %.6e, "
                       "max distance from E_hat = %.6e\n",
                       globalInverseCounts[3 * dir], globalInverseCounts[3 * dir + 1],
                       globalInverseCounts[3 * dir + 2], globalInverseMax[3 * dir],
                       globalInverseMax[3 * dir + 1], globalInverseMax[3 * dir + 2]);
        }
    }
}

// [nonmatching] Phase 9 A1 (plan amendment r1, revision 3): composite owner-face quadrature
// on coincident 2D interfaces. Collective on comm_Gamma; called by every member when the
// rank-uniform option is on, after the forward donor search. Sets quad.dir[*].coincident,
// applied and the global diagnostics, and on an applied own direction replaces res by the
// composite points (searched again) and fills the owner side of the sidecar. A direction
// that is not applied keeps res bitwise unchanged.
//
// Coincidence (scale-aware tolerance onFaceRelTol * h_F, as the "points off donor face"
// diagnostic): (a) every forward-search point of the direction lies on its donor face,
// and (b) every donor-face endpoint whose nearest owner face it projects into (reverse
// search: endpoints of the opposite model's faces queried against the owner faces, with
// reference coordinate s strictly inside (tau, 1 - tau), tau = 1e-12) lies on that face.
// Endpoint projection is a common refinement only on coincident interfaces: with a gap,
// nearest-donor transitions need not occur at projected endpoints, so non-coincident
// directions keep the uniform rule.
template <class T, class I>
inline void apply_composite_owner_quadrature(DonorSearchResult<T, I>& res,
        const InterfaceFaceSet<T, I>& own, const masterstructT<T, I>& master,
        const FaceMasterCompatibility<T, I>& compatibility,
        const InterfaceComm<I>& ic, I mpirank, I mpiprocs, I ownDirection, I elemtype, I porder,
        I nfe, const DonorSearchParams& prm, SearchStatistics* stats,
        CompositeOwnerQuadrature<T, I>& quad)
{
    const double start = MPI_Wtime();
    const I nd = res.nd, npf = own.npf, nf = own.size(), ngf = res.ngf;
    const T tauS = static_cast<T>(1.0e-12);
    const T tol = static_cast<T>(prm.onFaceRelTol);
    if (nd != 2) error("non-matching setup: composite owner quadrature supports nd = 2 only");
    if (!master.xpf) error("non-matching setup: composite owner quadrature needs master.xpf");
    const std::vector<int>& worldToInterface = ic.worldToComm;
    if (static_cast<I>(worldToInterface.size()) != mpiprocs)
        error("non-matching setup: interface rank map has the wrong size");

    int nonCoincident[2] = {0, 0};
    // (a) Forward-search points of the own direction.
    for (I p = 0; p < res.npts; ++p)
        if (res.dist[p] > tol * res.hface[p]) nonCoincident[ownDirection] = 1;

    // Reverse search: the two endpoints (reference 0 and 1) of every own face, queried
    // against the opposite model's faces (the owner faces of the other direction).
    DonorSearchResult<T, I> ep;
    ep.nd = nd; ep.ngf = 2; ep.nfaces = nf; ep.npts = checked_setup_size<I>({nf, 2}, "endpoints");
    ep.ownglobal = own.global;
    ep.x.assign(static_cast<std::size_t>(ep.npts) * nd, static_cast<T>(0));
    {
        std::vector<T> psi0(static_cast<std::size_t>(npf)), psi1(static_cast<std::size_t>(npf));
        face_lagrange_1d(psi0.data(), static_cast<T*>(nullptr), master.xpf, npf, static_cast<T>(0));
        face_lagrange_1d(psi1.data(), static_cast<T*>(nullptr), master.xpf, npf, static_cast<T>(1));
        for (I l = 0; l < nf; ++l)
            for (I d = 0; d < nd; ++d) {
                T x0 = 0, x1 = 0;
                for (I j = 0; j < npf; ++j) {
                    const T y = own.nodes[static_cast<std::size_t>(j + npf * d + npf * nd * l)];
                    x0 += psi0[j] * y;
                    x1 += psi1[j] * y;
                }
                ep.x[static_cast<std::size_t>(2 * l) + static_cast<std::size_t>(ep.npts) * d] = x0;
                ep.x[static_cast<std::size_t>(2 * l + 1) + static_cast<std::size_t>(ep.npts) * d] = x1;
            }
    }
    SearchFailures<I> epFailures;
    int epProtocol = SearchProtocolOk;
    DonorSearchParams prmEndpoint = prm;
    if (prm.endpointMaxProjectionIterations >= 0)
        prmEndpoint.maxProjectionIterations = prm.endpointMaxProjectionIterations;
    if (prm.searchMode == SearchMode::Exhaustive)
        exhaustive_donor_search(ep, ic, own, compatibility.model[1 - ownDirection], ownDirection,
                prmEndpoint, stats,
                &epFailures);
    else
        epProtocol = distributed_donor_search(ep, ic, own, master.xpf, elemtype, porder, mpirank,
                ownDirection, prmEndpoint, stats, &epFailures, nfe);
    {
        int globalProtocol = SearchProtocolOk;
        MPI_Allreduce(&epProtocol, &globalProtocol, 1, MPI_INT, MPI_MAX, ic.comm);
        ic.collective();
        if (globalProtocol != SearchProtocolOk)
            error("non-matching setup: composite owner quadrature endpoint search protocol error");
    }

    // Route each endpoint record to the rank owning its nearest opposite face.
    const int P = ic.size;
    // Revision 2: an endpoint the reverse search could not certify is a search failure (Phase 2
    // strict policy), never geometric noncoincidence. Failures and structural errors are kept
    // as data and reported uniformly after the record exchange (no rank-local exit between
    // collectives).
    const I noneI = std::numeric_limits<I>::max();
    I localFailPoint = noneI;      // first unconverged endpoint of this rank
    int localStructural = 0;       // 1: converged endpoint without a valid owner rank; 2: record
                                   // for an unowned face; 3: record count beyond MPI limits
    std::vector<int> sendCount(static_cast<std::size_t>(P), 0), recvCount(static_cast<std::size_t>(P), 0);
    auto record_valid = [&](I p) {
        return ep.converged[p] && ep.r[p] >= 0 && ep.r[p] < mpiprocs && worldToInterface[ep.r[p]] >= 0;
    };
    for (I p = 0; p < ep.npts; ++p) {
        if (!ep.converged[p]) {
            if (localFailPoint == noneI) localFailPoint = p;
            continue;
        }
        if (!record_valid(p)) { localStructural = std::max(localStructural, 1); continue; }
        sendCount[worldToInterface[ep.r[p]]]++;
    }
    MPI_Alltoall(sendCount.data(), 1, MPI_INT, recvCount.data(), 1, MPI_INT, ic.comm);
    ic.collective();
    constexpr int nInt = 3;  // matched owner global face, endpoint's face global index, endpoint flag
    constexpr int nReal = 3; // s, distance, h_F of the matched owner face
    std::vector<int> sendDisp(static_cast<std::size_t>(P), 0), recvDisp(static_cast<std::size_t>(P), 0);
    long long nsendWide = 0, nrecvWide = 0;
    for (int q = 0; q < P; ++q) {
        sendDisp[q] = static_cast<int>(std::min<long long>(nsendWide, std::numeric_limits<int>::max()));
        nsendWide += sendCount[q];
        recvDisp[q] = static_cast<int>(std::min<long long>(nrecvWide, std::numeric_limits<int>::max()));
        nrecvWide += recvCount[q];
    }
    {
        int localOverflow = (nsendWide * nReal > std::numeric_limits<int>::max() ||
                             nrecvWide * nReal > std::numeric_limits<int>::max()) ? 1 : 0;
        int globalOverflow = 0;
        MPI_Allreduce(&localOverflow, &globalOverflow, 1, MPI_INT, MPI_MAX, ic.comm);
        ic.collective();
        if (globalOverflow)
            error("non-matching setup: composite owner quadrature endpoint record count exceeds MPI integer limits");
    }
    const int nsend = static_cast<int>(nsendWide), nrecv = static_cast<int>(nrecvWide);
    std::vector<I> sendInt(static_cast<std::size_t>(nsend) * nInt), recvInt(static_cast<std::size_t>(nrecv) * nInt);
    std::vector<T> sendReal(static_cast<std::size_t>(nsend) * nReal), recvReal(static_cast<std::size_t>(nrecv) * nReal);
    {
        std::vector<int> cursor = sendDisp;
        for (I p = 0; p < ep.npts; ++p) {
            if (!record_valid(p)) continue;
            const int k = cursor[worldToInterface[ep.r[p]]]++;
            sendInt[static_cast<std::size_t>(nInt) * k] = ep.f[p];
            sendInt[static_cast<std::size_t>(nInt) * k + 1] = own.global[p / 2];
            sendInt[static_cast<std::size_t>(nInt) * k + 2] = p % 2;
            sendReal[static_cast<std::size_t>(nReal) * k] = ep.xi[p];
            sendReal[static_cast<std::size_t>(nReal) * k + 1] = ep.dist[p];
            sendReal[static_cast<std::size_t>(nReal) * k + 2] = ep.hface[p];
        }
    }
    auto scaled = [](const std::vector<int>& v, int factor) {
        std::vector<int> out(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) out[i] = v[i] * factor;
        return out;
    };
    {
        std::vector<int> sc = scaled(sendCount, nInt), sd = scaled(sendDisp, nInt);
        std::vector<int> rc = scaled(recvCount, nInt), rd = scaled(recvDisp, nInt);
        MPI_Alltoallv(sendInt.data(), sc.data(), sd.data(), mpi_type<I>(),
                recvInt.data(), rc.data(), rd.data(), mpi_type<I>(), ic.comm);
        ic.collective();
        sc = scaled(sendCount, nReal); sd = scaled(sendDisp, nReal);
        rc = scaled(recvCount, nReal); rd = scaled(recvDisp, nReal);
        MPI_Alltoallv(sendReal.data(), sc.data(), sd.data(), mpi_type<T>(),
                recvReal.data(), rc.data(), rd.data(), mpi_type<T>(), ic.comm);
        ic.collective();
    }

    // (b) Interior endpoint records of the own faces; candidate breakpoints per face.
    std::map<I, I> localFace;
    for (I l = 0; l < nf; ++l) localFace[own.global[l]] = l;
    struct Record { T s; I face; I flag; };
    std::vector<std::vector<Record>> records(static_cast<std::size_t>(nf));
    for (int k = 0; k < nrecv; ++k) {
        const I* ri = &recvInt[static_cast<std::size_t>(nInt) * k];
        const T* rr = &recvReal[static_cast<std::size_t>(nReal) * k];
        auto it = localFace.find(ri[0]);
        if (it == localFace.end()) { localStructural = std::max(localStructural, 2); continue; }
        const T s = rr[0];
        if (s <= tauS || s >= 1 - tauS) continue; // owner-face end vertices never break
        if (rr[1] > tol * rr[2]) { nonCoincident[ownDirection] = 1; continue; }
        records[it->second].push_back({s, ri[1], ri[2]});
    }

    // Uniform reports after all record communication: endpoint search failures first (the
    // deterministic tuple of the forward search: smallest failing querying rank, its first
    // failing endpoint, that endpoint's first failing donor rank and face), then structural
    // errors of the record exchange.
    {
        const I localRank = (localFailPoint == noneI) ? noneI : mpirank;
        I failRank = noneI;
        MPI_Allreduce(&localRank, &failRank, 1, mpi_type<I>(), MPI_MIN, ic.comm);
        ic.collective();
        if (failRank != noneI) {
            I mine[3] = {noneI, noneI, noneI}, tuple[3] = {noneI, noneI, noneI};
            if (failRank == mpirank) {
                mine[0] = localFailPoint;
                mine[1] = epFailures.donorRank.empty() ? -1 : epFailures.donorRank[localFailPoint];
                mine[2] = epFailures.face.empty() ? -1 : epFailures.face[localFailPoint];
            }
            MPI_Allreduce(mine, tuple, 3, mpi_type<I>(), MPI_MIN, ic.comm);
            ic.collective();
            const int failComm = ic.worldToComm[failRank];
            const I querying = (failComm >= 0) ? ic.model[failComm] : -1;
            std::ostringstream oss;
            oss << "non-matching setup: composite owner quadrature endpoint search failed (projection, "
                << "certified bound, or certified gap) for endpoint owner rank " << failRank
                << ", direction " << (querying >= 0 ? 1 - querying : -1) << " (Gamma^"
                << (querying >= 0 ? 2 - querying : 0) << "), endpoint " << tuple[0]
                << " (face " << tuple[0] / 2 << ", end " << tuple[0] % 2 << "); first failing donor rank "
                << tuple[1] << ", global face " << tuple[2];
            error(oss.str());
        }
        int globalStructural = 0;
        MPI_Allreduce(&localStructural, &globalStructural, 1, MPI_INT, MPI_MAX, ic.comm);
        ic.collective();
        if (globalStructural == 1)
            error("non-matching setup: composite owner quadrature endpoint has no valid owner rank");
        if (globalStructural == 2)
            error("non-matching setup: composite owner quadrature endpoint record for a face its rank does not own");
    }

    int globalNonCoincident[2] = {0, 0};
    MPI_Allreduce(nonCoincident, globalNonCoincident, 2, MPI_INT, MPI_MAX, ic.comm);
    ic.collective();

    // Breakpoints of the own faces (only meaningful if the own direction is coincident).
    std::vector<I> facebreakptr(static_cast<std::size_t>(nf) + 1, 0);
    std::vector<T> breakpoints;
    I localMaxPieces = nf > 0 ? 1 : 0;
    long long localBreaks = 0;
    if (!globalNonCoincident[ownDirection]) {
        for (I l = 0; l < nf; ++l) {
            auto& rec = records[l];
            std::sort(rec.begin(), rec.end(), [](const Record& a, const Record& b) {
                return std::tie(a.s, a.face, a.flag) < std::tie(b.s, b.face, b.flag);
            });
            std::vector<T> merged;
            for (const Record& r : rec)
                if (merged.empty() || r.s - merged.back() > tauS) merged.push_back(r.s);
            breakpoints.insert(breakpoints.end(), merged.begin(), merged.end());
            facebreakptr[l + 1] = static_cast<I>(breakpoints.size());
            localMaxPieces = std::max<I>(localMaxPieces, static_cast<I>(merged.size()) + 1);
            localBreaks += static_cast<long long>(merged.size());
        }
    }
    int localFlags[3] = {0, 0, 0}, globalFlags[3] = {0, 0, 0};
    localFlags[ownDirection] = static_cast<int>(localMaxPieces);
    localFlags[2] = localMaxPieces > prm.maxCompositePiecesPerFace ? 1 : 0;
    MPI_Allreduce(localFlags, globalFlags, 3, MPI_INT, MPI_MAX, ic.comm);
    ic.collective();
    long long localCounts[2] = {0, 0}, globalCounts[2] = {0, 0};
    localCounts[ownDirection] = localBreaks;
    MPI_Allreduce(localCounts, globalCounts, 2, MPI_LONG_LONG, MPI_SUM, ic.comm);
    ic.collective();
    if (globalFlags[2]) {
        std::ostringstream oss;
        oss << "non-matching setup: composite owner quadrature needs more than "
            << prm.maxCompositePiecesPerFace << " pieces on one owner face (maxCompositePiecesPerFace)";
        error(oss.str());
    }
    for (int k = 0; k < 2; ++k) {
        quad.dir[k].coincident = globalNonCoincident[k] ? 0 : 1;
        quad.dir[k].applied = (!globalNonCoincident[k] && globalCounts[k] > 0) ? 1 : 0;
        quad.dir[k].breakpointsTotal = static_cast<I>(globalCounts[k]);
        quad.dir[k].maxPiecesPerFace = static_cast<I>(globalFlags[k]);
    }
    if (!quad.dir[0].applied && !quad.dir[1].applied) {
        if (stats) stats->endpointSeconds += MPI_Wtime() - start;
        return;
    }

    // Composite points of the own faces (applied own direction only).
    const bool ownApplied = quad.dir[ownDirection].applied != 0;
    DonorSearchResult<T, I> cres;
    cres.nd = nd; cres.ngf = ngf; cres.direction = res.direction;
    if (ownApplied) {
        if (!master.gpf || !master.shapfgt || !master.gwf)
            error("non-matching setup: composite owner quadrature needs master.gpf, shapfgt and gwf");
        cres.nfaces = nf;
        cres.ownglobal = own.global;
        cres.facePointPtr.assign(static_cast<std::size_t>(nf) + 1, 0);
        for (I l = 0; l < nf; ++l) {
            const I nb = facebreakptr[l + 1] - facebreakptr[l];
            cres.facePointPtr[l + 1] = cres.facePointPtr[l] + checked_setup_size<I>({nb + 1, ngf}, "composite points");
        }
        const I npts = cres.facePointPtr[nf];
        cres.npts = npts;
        cres.pointFace.resize(npts); cres.pointGauss.resize(npts); cres.pointPiece.resize(npts);
        cres.pointS.resize(npts); cres.pointLen.resize(npts); cres.pointUniform.resize(npts);
        cres.pointTest.assign(static_cast<std::size_t>(checked_setup_size<I>({npf, npts}, "composite tests")), 0);
        cres.x.assign(static_cast<std::size_t>(npts) * nd, static_cast<T>(0));
        std::vector<T> psi(static_cast<std::size_t>(npf));
        for (I l = 0; l < nf; ++l) {
            const I b0 = facebreakptr[l], b1 = facebreakptr[l + 1];
            const bool uniform = b1 == b0;
            I p = cres.facePointPtr[l];
            for (I piece = 0; piece <= b1 - b0; ++piece) {
                const T a = piece == 0 ? static_cast<T>(0) : breakpoints[b0 + piece - 1];
                const T b = piece == b1 - b0 ? static_cast<T>(1) : breakpoints[b0 + piece];
                for (I g = 0; g < ngf; ++g, ++p) {
                    cres.pointFace[p] = l; cres.pointGauss[p] = g; cres.pointPiece[p] = piece;
                    cres.pointUniform[p] = uniform ? 1 : 0;
                    if (uniform) {
                        // The uniform expressions of own_quadrature_points (bitwise).
                        cres.pointS[p] = master.gpf[g];
                        cres.pointLen[p] = 1;
                        for (I d = 0; d < nd; d++) {
                            T x = 0;
                            for (I j = 0; j < npf; j++)
                                x += master.shapfgt[g + ngf * j] *
                                     own.nodes[static_cast<std::size_t>(j + npf * d + npf * nd * l)];
                            cres.x[static_cast<std::size_t>(p) + static_cast<std::size_t>(npts) * d] = x;
                        }
                        for (I i = 0; i < npf; ++i)
                            cres.pointTest[static_cast<std::size_t>(i) + static_cast<std::size_t>(npf) * p] =
                                master.shapfgt[g + ngf * i];
                    }
                    else {
                        const T sp = a + (b - a) * master.gpf[g];
                        cres.pointS[p] = sp;
                        cres.pointLen[p] = b - a;
                        face_lagrange_1d(psi.data(), static_cast<T*>(nullptr), master.xpf, npf, sp);
                        for (I d = 0; d < nd; d++) {
                            T x = 0;
                            for (I j = 0; j < npf; j++)
                                x += psi[j] * own.nodes[static_cast<std::size_t>(j + npf * d + npf * nd * l)];
                            cres.x[static_cast<std::size_t>(p) + static_cast<std::size_t>(npts) * d] = x;
                        }
                        for (I i = 0; i < npf; ++i)
                            cres.pointTest[static_cast<std::size_t>(i) + static_cast<std::size_t>(npf) * p] = psi[i];
                    }
                }
            }
        }
    }

    // Forward search of the composite points (every member takes part).
    SearchFailures<I> cFailures;
    int cProtocol = SearchProtocolOk;
    if (prm.searchMode == SearchMode::Exhaustive)
        exhaustive_donor_search(cres, ic, own, compatibility.model[1 - ownDirection], ownDirection,
                prm, stats, &cFailures);
    else
        cProtocol = distributed_donor_search(cres, ic, own, master.xpf, elemtype, porder, mpirank,
                ownDirection, prm, stats, &cFailures, nfe);
    int localFail = cProtocol != SearchProtocolOk ? 1 : 0;
    for (I p = 0; p < cres.npts; ++p)
        if (!cres.converged[p]) localFail = 1;

    // Tiling on the faces with breakpoints: one donor face per piece, pieces of one
    // donor face contiguous, every point on its donor face.
    I badFace = std::numeric_limits<I>::max();
    if (ownApplied) {
        for (I l = 0; l < nf; ++l) {
            const I nb = facebreakptr[l + 1] - facebreakptr[l];
            if (nb == 0) continue;
            std::vector<I> pieceFace;
            bool bad = false;
            for (I piece = 0; piece <= nb && !bad; ++piece) {
                const I p0 = cres.facePointPtr[l] + piece * ngf;
                for (I g = 0; g < ngf; ++g) {
                    const I p = p0 + g;
                    if (cres.f[p] != cres.f[p0] || cres.dist[p] > tol * cres.hface[p]) bad = true;
                }
                pieceFace.push_back(cres.f[p0]);
            }
            for (std::size_t k = 1; k < pieceFace.size() && !bad; ++k)
                if (pieceFace[k] != pieceFace[k - 1] &&
                    std::find(pieceFace.begin(), pieceFace.begin() + k - 1, pieceFace[k]) != pieceFace.begin() + k - 1)
                    bad = true;
            if (bad) badFace = std::min(badFace, own.global[l]);
        }
    }
    int globalFail = 0;
    MPI_Allreduce(&localFail, &globalFail, 1, MPI_INT, MPI_MAX, ic.comm);
    ic.collective();
    if (globalFail)
        error("non-matching setup: composite owner quadrature donor search failed");
    I globalBadFace = std::numeric_limits<I>::max();
    MPI_Allreduce(&badFace, &globalBadFace, 1, mpi_type<I>(), MPI_MIN, ic.comm);
    ic.collective();
    if (globalBadFace != std::numeric_limits<I>::max()) {
        std::ostringstream oss;
        oss << "non-matching setup: composite pieces do not tile owner face " << globalBadFace;
        error(oss.str());
    }

    if (ownApplied) {
        res = std::move(cres);
        quad.dir[ownDirection].facebreakptr = std::move(facebreakptr);
        quad.dir[ownDirection].breakpoints = std::move(breakpoints);
    }
    if (stats) stats->endpointSeconds += MPI_Wtime() - start;
}

// Low-level setup entry point. Production construction and mesh-motion rebuilds
// use rebuild_nonmatching_interface below so request capacity cannot lag the cache.
// Returns false (and does nothing) unless interfacetype == 1 and the coupled
// guards of build_runtime_interface_partition hold. Otherwise every rank of both
// models takes part (collective on EXASIM_COMM_WORLD). If result is non-null the
// per-point tuples are returned in it (phase 3 consumes them).
template <class T, class I>
inline bool setup_nonmatching_interface(appstructT<T, I>& app, masterstructT<T, I>& master,
        meshstructT<T, I>& mesh, solstructT<T, I>& sol, I mpiprocs, I mpirank, I fileoffset,
        nonmatchingdataT<T, I>& nm, DonorSearchResult<T, I>* result = nullptr,
        const DonorSearchParams& prm = DonorSearchParams(), bool performDistribution = true,
        SearchStatistics* stats = nullptr, CompositeOwnerQuadrature<T, I>* quad = nullptr)
{
    // [nonmatching] Setup owns a complete cache generation.  Discard the old
    // directional data first so mesh-motion rebuilds and active-to-inactive
    // transitions cannot retain stale point locations or work arrays.
    free_nonmatching_data(nm);
    if (quad) quad->clear(); // [nonmatching] Phase 9 A1 sidecar of this generation

    // Same guards as build_runtime_interface_partition.  Since every world
    // rank enters collectives below, first make pointer/configuration selection
    // rank-uniform: a rank-local early return would deadlock the other ranks.
    if (mpiprocs <= 1)
        return false;
    const int localReady = app.problem && app.nsize && master.ndims && mesh.ndims &&
                           mesh.bf && mesh.perm && sol.xdg;
    int allReady = 0, anyReady = 0;
    MPI_Allreduce(&localReady, &allReady, 1, MPI_INT, MPI_MIN, EXASIM_COMM_WORLD);
    MPI_Allreduce(&localReady, &anyReady, 1, MPI_INT, MPI_MAX, EXASIM_COMM_WORLD);
    if (allReady != anyReady)
        error("non-matching setup: required input arrays are not available on every coupled rank");
    if (!allReady)
        return false;
    const int localActive = app.nsize[2] > 34 && app.problem[28] > 0 &&
                            app.problem[29] > 0 && app.problem[30] > 0 &&
                            app.problem[34] == 1;
    int allActive = 0, anyActive = 0;
    MPI_Allreduce(&localActive, &allActive, 1, MPI_INT, MPI_MIN, EXASIM_COMM_WORLD);
    MPI_Allreduce(&localActive, &anyActive, 1, MPI_INT, MPI_MAX, EXASIM_COMM_WORLD);
    if (allActive != anyActive)
        error("non-matching setup: coupling selection is inconsistent across coupled ranks");
    if (!allActive)
        return false;
    // [nonmatching] Phase 9 A1: the composite owner quadrature option must be rank-uniform
    // before any direction-specific collective (amendment r1, A.7).
    {
        const int localRequest = prm.compositeOwnerQuadrature ? 1 : 0;
        int minRequest = 0, maxRequest = 0;
        MPI_Allreduce(&localRequest, &minRequest, 1, MPI_INT, MPI_MIN, EXASIM_COMM_WORLD);
        MPI_Allreduce(&localRequest, &maxRequest, 1, MPI_INT, MPI_MAX, EXASIM_COMM_WORLD);
        if (minRequest != maxRequest)
            error("non-matching setup: composite owner quadrature option differs across ranks");
        if (maxRequest && !quad)
            error("non-matching setup: composite owner quadrature requires the quadrature sidecar");
        if (quad) quad->requested = maxRequest;
    }
    nm.interfacetype = 1;

    const I nd = master.ndims[0], elemtype = master.ndims[1], porder = master.ndims[3];
    const I pgauss = master.ndims[4];
    const I npe = master.ndims[5], npf = master.ndims[6], ngf = master.ndims[8];
    const I nfe = mesh.ndims[4];
    const I own = (fileoffset > 0) ? 1 : 0;

    const InterfaceFaceSet<T, I> ownFaces = collect_own_interface_faces(app, master, mesh, sol, mpirank, fileoffset);
    for (int k = 0; k < 2; k++) {
        nonmatchinginterfaceT<T, I>& d = nm.dir[k];
        d.direction = k;
        d.nd = nd; d.npe = npe; d.npf = npf; d.nfe = nfe; d.ngf = ngf;
        d.ncu12 = (app.nsize[14] > 0) ? app.nsize[14] : 0; // szinterfacefluxmap
    }
    DonorSearchResult<T, I> res;
    res.direction = own;
    own_quadrature_points(res, ownFaces, master.shapfgt, ngf);

    // Phase 7: one setup-only interface communicator per rebuild (members: ranks
    // owning a coupled interface face). Nonmembers keep a valid empty type-1
    // cache and enter no further interface collective.
    InterfaceComm<I> ic;
    create_interface_comm(ic, ownFaces.size() > 0, mpirank, mpiprocs, own, stats);
    if (ic.comm == MPI_COMM_NULL) {
        build_owner_side(nm.dir[own], res);
        if (result) *result = std::move(res);
        return true;
    }

    // [nonmatching] Phase 10: the models share the 1D face space and rule,
    // while their triangle/quad volume masters remain role-local.
    const FaceMasterCompatibility<T, I> compatibility =
            check_face_master_compatibility<T, I>(master, nfe, own, ic);
    if (compatibility.mixedFamily && !quad)
        error("non-matching interface: mixed element families require the owner-test sidecar");

    SearchFailures<I> failures;
    int protocolError = SearchProtocolOk;
    const double searchStart = MPI_Wtime();
    if (prm.searchMode == SearchMode::Exhaustive)
        exhaustive_donor_search(res, ic, ownFaces, compatibility.model[1 - own], own,
                prm, stats, &failures);
    else
        protocolError = distributed_donor_search(res, ic, ownFaces, master.xpf, elemtype, porder, mpirank, own,
                prm, stats, &failures, nfe);
    if (stats) stats->searchSeconds += MPI_Wtime() - searchStart;

    // Protocol errors and point failures are reduced on comm_Gamma after all search
    // rounds and reported uniformly by every member.
    {
        int globalProtocolError = SearchProtocolOk;
        MPI_Allreduce(&protocolError, &globalProtocolError, 1, MPI_INT, MPI_MAX, ic.comm);
        ic.collective();
        if (globalProtocolError != SearchProtocolOk) {
            ic.release();
            std::ostringstream oss;
            oss << "non-matching donor search: protocol error (" << search_protocol_error_text(globalProtocolError)
                << ")";
            error(oss.str());
        }
        // Deterministic failure tuple: smallest failing owner world rank, then its
        // smallest failing point, with that point's first failing donor rank and face.
        const I none = std::numeric_limits<I>::max();
        I localPoint = none;
        for (I p = 0; p < res.npts && localPoint == none; p++)
            if (!res.converged[p]) localPoint = p;
        const I localRank = (localPoint == none) ? none : mpirank;
        I failRank = none;
        MPI_Allreduce(&localRank, &failRank, 1, mpi_type<I>(), MPI_MIN, ic.comm);
        ic.collective();
        if (failRank != none) {
            I mine[3] = {none, none, none}, tuple[3] = {none, none, none};
            if (failRank == mpirank) {
                mine[0] = localPoint;
                mine[1] = failures.donorRank.empty() ? -1 : failures.donorRank[localPoint];
                mine[2] = failures.face.empty() ? -1 : failures.face[localPoint];
            }
            MPI_Allreduce(mine, tuple, 3, mpi_type<I>(), MPI_MIN, ic.comm);
            ic.collective();
            const int failComm = ic.worldToComm[failRank];
            const I direction = (failComm >= 0) ? ic.model[failComm] : -1;
            ic.release();
            std::ostringstream oss;
            oss << "non-matching donor search: a candidate face failed (projection, certified bound, or "
                << "certified gap) for owner rank " << failRank << ", direction " << direction
                << " (Gamma^" << direction + 1 << "), point " << tuple[0] << "; first failing donor rank "
                << tuple[1] << ", global face " << tuple[2];
            error(oss.str());
        }
    }

    // [nonmatching] Phase 9 A1: opt-in composite owner-face quadrature (coincident 2D).
    if (quad && quad->requested)
        apply_composite_owner_quadrature(res, ownFaces, master, compatibility, ic, mpirank, mpiprocs, own,
                elemtype, porder, nfe, prm, stats, *quad);

    // [nonmatching] Phase 10: mixed-family D6 always transports the owner
    // test basis. Composite quadrature reuses the same payload.
    if (quad)
        for (int k = 0; k < 2; ++k)
            quad->dir[k].ownerTests = compatibility.mixedFamily || quad->dir[k].applied;

    build_owner_side(nm.dir[own], res);
    if (performDistribution)
        distribute_setup_data(nm, res, ownFaces, app, master, mesh, sol,
                mpiprocs, mpirank, own, prm, ic, quad);
    print_donor_diagnostics(res, ic, prm);
    if (quad && quad->requested && ic.rank == 0)
        for (int k = 0; k < 2; ++k)
            printf("Non-matching composite owner quadrature, Gamma^%d: %s, %d interior breakpoints, "
                   "max pieces per owner face %d\n", k + 1,
                   quad->dir[k].applied ? "applied" :
                   (quad->dir[k].coincident ? "not applied (no interior breakpoint)"
                                            : "not applied (interface not coincident)"),
                   static_cast<int>(quad->dir[k].breakpointsTotal),
                   static_cast<int>(quad->dir[k].maxPiecesPerFace));

    if (result) *result = std::move(res);
    return true;
}

// [nonmatching] Atomic cache rebuild used by production and mesh-motion paths.
// Resizing requests here makes it impossible for changed donor/owner neighbor
// counts to leave the next compact exchange with stale capacity.
template <class T, class I>
inline bool rebuild_nonmatching_interface(appstructT<T, I>& app, masterstructT<T, I>& master,
        meshstructT<T, I>& mesh, solstructT<T, I>& sol, commonstructT<T, I>& common,
        I mpiprocs, I mpirank, I fileoffset, nonmatchingdataT<T, I>& nm,
        DonorSearchResult<T, I>* result = nullptr,
        const DonorSearchParams& prm = DonorSearchParams(), bool performDistribution = true)
{
    const bool hadTypeOne = nm.interfacetype == 1;
    const bool ran = setup_nonmatching_interface(app, master, mesh, sol, mpiprocs, mpirank,
            fileoffset, nm, result, prm, performDistribution);
    if (ran || hadTypeOne)
        ensure_nonmatching_request_capacity(common, nm);
    return ran;
}

// [nonmatching] Phase 8 transaction: construct host setup data, backend
// mirrors/workspaces and replacement MPI request storage before publishing any
// of them.  This is the production rebuild API used for construction and mesh
// motion; the lower-level overload above remains for setup-only tests.
template <class T, class I>
inline void validate_nonmatching_runtime_owner_maps(const nonmatchingdataT<T, I>& nm,
        const meshstructT<T, I>& mesh, const appstructT<T, I>& app,
        const commonstructT<T, I>& common)
{
    if (nm.interfacetype != 1) return;
    std::vector<I> pendingInterfaceFaces;
    const I* interfaceFaces = mesh.intfaces;
    I interfaceFaceCount = mesh.szintfaces;
    if (!interfaceFaces) {
        const I boundary = common.couplingparams.coupledboundarycondition;
        if (!mesh.bf || boundary <= 0)
            error("non-matching runtime: coupled boundary-face data are incomplete");
        for (I element = 0; element < common.meshsizes.ne1; ++element)
            for (I localFace = 0; localFace < common.meshsizes.nfe; ++localFace)
                if (mesh.bf[localFace + common.meshsizes.nfe * element] == boundary)
                    pendingInterfaceFaces.push_back(localFace + common.meshsizes.nfe * element);
        interfaceFaces = pendingInterfaceFaces.data();
        interfaceFaceCount = static_cast<I>(pendingInterfaceFaces.size());
    }
    for (int direction = 0; direction < 2; ++direction) {
        const auto& owner = nm.dir[direction];
        if (!owner.isowner || owner.nownfaces == 0) continue;
        if (owner.nfe <= 0 || !owner.ownface || !app.interfacefluxmap)
            error("non-matching runtime: owner mapping data are incomplete");
        for (I own = 0; own < owner.nownfaces; ++own) {
            const I interfaceFace = owner.ownface[own];
            if (interfaceFace < 0 || interfaceFace >= interfaceFaceCount)
                error("non-matching runtime: owner interface-face index is out of range");
            const I encoded = interfaceFaces[interfaceFace];
            const I element = encoded / owner.nfe;
            const I localFace = encoded - owner.nfe * element;
            if (encoded < 0 || element < 0 || element >= common.meshsizes.ne ||
                localFace < 0 || localFace >= owner.nfe)
                error("non-matching runtime: encoded owner element face is out of range");
        }
        for (I component = 0; component < owner.ncu12; ++component)
            if (app.interfacefluxmap[component] < 0 ||
                app.interfacefluxmap[component] >= common.components.ncu)
                error("non-matching runtime: interfacefluxmap is out of range");
    }
}

template <class T, class I>
inline bool rebuild_nonmatching_interface(appstructT<T, I>& app,
        masterstructT<T, I>& master, meshstructT<T, I>& mesh,
        solstructT<T, I>& sol, commonstructT<T, I>& common,
        I mpiprocs, I mpirank, I fileoffset, nonmatchingdataT<T, I>& nm,
        std::unique_ptr<NonmatchingRuntime<T, I>>& runtime, I backend,
        DonorSearchResult<T, I>* result = nullptr,
        const DonorSearchParams& prm = DonorSearchParams(),
        bool performDistribution = true,
        std::unique_ptr<CompositeOwnerQuadrature<T, I>>* quad = nullptr)
{
    nonmatchingdataT<T, I> candidate;
    std::unique_ptr<NonmatchingRuntime<T, I>> candidateRuntime;
    // [nonmatching] Phase 9 A1: the composite quadrature sidecar is part of the same
    // transaction (built into a candidate, published with nm, the runtime and requests).
    std::unique_ptr<CompositeOwnerQuadrature<T, I>> candidateQuad;
    if (quad) candidateQuad = std::make_unique<CompositeOwnerQuadrature<T, I>>();
    MPI_Request* candidateRequests = nullptr;
    MPI_Status* candidateStatuses = nullptr;
    bool ran = false;
    try {
        ran = setup_nonmatching_interface(app, master, mesh, sol, mpiprocs, mpirank,
                fileoffset, candidate, result, prm, performDistribution, nullptr, candidateQuad.get());
        if (candidate.interfacetype == 1) {
            validate_nonmatching_runtime_owner_maps(candidate, mesh, app, common);
            candidateRuntime = std::make_unique<NonmatchingRuntime<T, I>>();
            refresh_nonmatching_runtime(*candidateRuntime, candidate, app, mesh, common, backend,
                                        candidateQuad.get());
        }

        long long nsend = 0, nrecv = 0;
        for (int direction = 0; direction < 2; ++direction) {
            nsend += candidate.dir[direction].nsendnb;
            nrecv += candidate.dir[direction].nrecvnb;
        }
        const long long conformal = 2LL * common.nnbsd + 2LL * common.nnbintf;
        const long long nonmatching = 2LL * common.nnbsd + nsend + nrecv;
        const long long requestCapacity = std::max(conformal, nonmatching);
        if (requestCapacity < 0 || requestCapacity > std::numeric_limits<int>::max())
            error("non-matching runtime: MPI request capacity exceeds integer range");
        if (requestCapacity > 0) {
            candidateRequests = static_cast<MPI_Request*>(std::malloc(
                static_cast<std::size_t>(requestCapacity) * sizeof(MPI_Request)));
            candidateStatuses = static_cast<MPI_Status*>(std::malloc(
                static_cast<std::size_t>(requestCapacity) * sizeof(MPI_Status)));
            if (!candidateRequests || !candidateStatuses)
                error("non-matching runtime: MPI request allocation failed");
        }
    }
    catch (...) {
        free_nonmatching_data(candidate);
        CPUFREE(candidateRequests);
        CPUFREE(candidateStatuses);
        throw;
    }

    std::swap(nm, candidate);
    runtime.swap(candidateRuntime);
    if (quad) quad->swap(candidateQuad);
    std::swap(common.requests, candidateRequests);
    std::swap(common.statuses, candidateStatuses);
    free_nonmatching_data(candidate);
    CPUFREE(candidateRequests);
    CPUFREE(candidateStatuses);
    return ran;
}
#else
template <class T, class I>
inline bool setup_nonmatching_interface(appstructT<T, I>&, masterstructT<T, I>&, meshstructT<T, I>&,
        solstructT<T, I>&, I, I, I, nonmatchingdataT<T, I>&, DonorSearchResult<T, I>* = nullptr,
        const DonorSearchParams& = DonorSearchParams(), bool = true, SearchStatistics* = nullptr,
        CompositeOwnerQuadrature<T, I>* = nullptr)
{
    return false; // no coupling without MPI (phase 0 report, D7)
}
template <class T, class I>
inline bool rebuild_nonmatching_interface(appstructT<T, I>&, masterstructT<T, I>&,
        meshstructT<T, I>&, solstructT<T, I>&, commonstructT<T, I>&,
        I, I, I, nonmatchingdataT<T, I>&, DonorSearchResult<T, I>* = nullptr,
        const DonorSearchParams& = DonorSearchParams(), bool = true)
{
    return false;
}
#endif

} // namespace nonmatching
} // namespace exasim

#endif
