// Phase 5 of the non-matching interface solver: coupling in the HDG
// matrix-vector product (docs/nonmatching/nonmatching_multiphysics.tex,
// eqs. (64), (67), (68)).
//
// MPI test, run with 2^j ranks. For every n = 2, 4, ..., worldSize the first n
// world ranks form EXASIM_COMM_WORLD (n/2 ranks per model, row strips); the
// other ranks idle. Each rank builds a full local HDG mesh of its strip (faces
// matched by coordinates, so f2e/elemcon carry the reversed node order of the
// second element: a nontrivial permutation), runs the production setup, and
// calls the production hdgMatVec. The original strip fixtures use independent
// pieces of the trace system (faces on strip boundaries are duplicated); the
// dedicated halo case below adds real same-model element-halo traffic.
//
// Cases:
//   product reference (non-matching curved quads, p = 2): the type-1 hdgMatVec
//      equals an independent assembly (direct pair products, all-gather, owner
//      scatter by global face) to round-off; |K_l| > 1 and, for 2 or more ranks
//      per model, |R_l| > 1 are required.
//   finite differences (same mesh, d != 0): the coupling part of hdgMatVec(v)
//      equals central differences of the uncondensed coupling residual along
//      the local-solver tangent (delta u = DinvF v, delta q = C delta u - E v),
//      computed with the production residual, coupling-slot exchange and owner
//      assembly. Nonlinear ncu = 2 q-dependent model, and a separate ncw = 2
//      model whose w(uh, q) chain goes through wEquation.
//   conformal equivalence (matching quads, p = 2, several elements per rank):
//      the conformal hdgMatVec path (production uEquationElemFaceBlock and
//      uEquationSchurBlock res.Hi, faceperm, facesend/facerecv) and the type-1
//      path give the same w to round-off.
//   GMRES (matching quads): right-preconditioned GMRES with the
//      elemental-block preconditioner applied through hdgMatVec (decision D1)
//      takes identical iteration counts for the conformal and non-matching
//      operators.
//   halo (2 or more ranks per model): strips with their ghost rows, real
//      same-model element-halo traffic (nnbsd, elemsend, elemrecv) next to the
//      coupling-slot exchange in one hdgMatVec with the exact request count;
//      donor elements on the partition boundary; compared with a reference
//      that adds every owned and ghost element contribution and the
//      independent pair products.
//   disabled coupling: with type-1 data present and coupledinterface = 0 or
//      coupledcondition = 0, hdgMatVec applies no coupling.
//   coupling selection: consistent selections pass; inconsistent or
//      unsupported ones (missing data, setup not run, type mismatch, GPU
//      backend, no MPI, also through hdgMatVec) fail in a child process forked
//      before MPI_Init with the expected message.

#include <functional>
#include <cstring>
#include <map>
#include <memory>
#include <sys/wait.h>
#include <unistd.h>

#define main nonmatching_donorsearch_embedded_main
#include "nonmatching_donorsearch.cpp"
#undef main

using namespace std;

#include "../../backend/Common/common.h"
#include "../../backend/Common/cpuimpl.h"
#include "../../backend/Common/kokkosimpl.h"
#include "../../backend/Common/pblas.h"
#include "../../backend/Discretization/ioutilities.cpp"
#include "../../backend/Discretization/residual.hpp"
#include "../../backend/Discretization/nonmatchingcoupling.hpp"
#include "../../backend/Discretization/matvec.hpp"

namespace {

dstype maxRoundoff = 0;   // largest scaled round-off discrepancy of the exact checks
dstype maxFdError = 0;    // smallest-step FD error of the FD cases (relative)

// Nonlinear q-dependent two-component coupling (no auxiliary fields).
//   f0 = u0*uh0 + 0.5*q(0,0)^2 + 2*uh0 + q(1,1)*uh1
//   f1 = sin(uh1) + u1*q(0,1) - uh1*uh0
// uq = (u0, u1, q(0,0), q(1,0), q(0,1), q(1,1)), q(c,d) = uq[2 + c + 2d].
struct NLCoupled : exasim::ModelDefaults<NLCoupled> {
    static constexpr int nd = 2, ncu = 2, ncq = ncu * nd, ncw = 0, nparam = 1, nfint = 2;
    static constexpr auto disc = exasim::Discretization::HDG;
    static constexpr bool has_external_coupling = true;

    KOKKOS_INLINE_FUNCTION static void flux(double f[], const double[],
            const double[], const double[], const double[], const double[],
            const double[], double)
    {
        for (int k = 0; k < ncu * nd; ++k) f[k] = 0;
    }

    KOKKOS_INLINE_FUNCTION static void fint(double f[], int, const double[],
            const double uq[], const double[], const double[], const double uh[],
            const double[], const double[], const double[], const double[], double)
    {
        f[0] = uq[0] * uh[0] + 0.5 * uq[2] * uq[2] + 2.0 * uh[0] + uq[5] * uh[1];
        f[1] = sin(uh[1]) + uq[1] * uq[4] - uh[1] * uh[0];
    }

    KOKKOS_INLINE_FUNCTION static void fint_jac_uq(double j[], int, const double[],
            const double uq[], const double[], const double[], const double uh[],
            const double[], const double[], const double[], const double[], double)
    {
        for (int k = 0; k < 12; ++k) j[k] = 0;
        j[0 + 2 * 0] = uh[0];
        j[0 + 2 * 2] = uq[2];
        j[0 + 2 * 5] = uh[1];
        j[1 + 2 * 1] = uq[4];
        j[1 + 2 * 4] = uq[1];
    }

    KOKKOS_INLINE_FUNCTION static void fint_jac_uh(double j[], int, const double[],
            const double uq[], const double[], const double[], const double uh[],
            const double[], const double[], const double[], const double[], double)
    {
        j[0 + 2 * 0] = uq[0] + 2.0;
        j[1 + 2 * 0] = -uh[1];
        j[0 + 2 * 1] = uq[5];
        j[1 + 2 * 1] = cos(uh[1]) - uh[0];
    }
};

// [nonmatching] Phase 9 B1: scalar Poisson interface pair of the coupled example
// (examples/CoupledProblems/Poisson/nonmatching). param[0] selects the side:
//   param[0] = 0 (model 1): fint = -uh                      (Dirichlet data for model 2)
//   param[0] = 1 (model 2): fint = q.n + tau (u - uh)       (flux for model 1)
// uq = (u, q0, q1).
struct PoissonCoupling : exasim::ModelDefaults<PoissonCoupling> {
    static constexpr int nd = 2, ncu = 1, ncq = ncu * nd, ncw = 0, nparam = 1, nfint = 1;
    static constexpr auto disc = exasim::Discretization::HDG;
    static constexpr bool has_external_coupling = true;

    KOKKOS_INLINE_FUNCTION static void flux(double f[], const double[],
            const double uq[], const double[], const double[], const double[],
            const double[], double)
    {
        f[0] = uq[1];
        f[1] = uq[2];
    }

    KOKKOS_INLINE_FUNCTION static void fint(double f[], int, const double[],
            const double uq[], const double[], const double[], const double uh[],
            const double n[], const double tau[], const double param[], const double[], double)
    {
        f[0] = param[0] > 0.5 ? uq[1] * n[0] + uq[2] * n[1] + tau[0] * (uq[0] - uh[0]) : -uh[0];
    }

    KOKKOS_INLINE_FUNCTION static void fint_jac_uq(double j[], int, const double[],
            const double[], const double[], const double[], const double[],
            const double n[], const double tau[], const double param[], const double[], double)
    {
        const bool fluxSide = param[0] > 0.5;
        j[0] = fluxSide ? tau[0] : 0.0;
        j[1] = fluxSide ? n[0] : 0.0;
        j[2] = fluxSide ? n[1] : 0.0;
    }

    KOKKOS_INLINE_FUNCTION static void fint_jac_uh(double j[], int, const double[],
            const double[], const double[], const double[], const double[],
            const double[], const double tau[], const double param[], const double[], double)
    {
        j[0] = param[0] > 0.5 ? -tau[0] : -1.0;
    }
};

// Two auxiliary fields, linear in w (the w Newton solve is exact after one
// step) and nonlinear in (uh, q). With dae_alpha = dae_beta = 0 the w equation
// is s(uq, w) = 0 and its Newton matrix is exactly -ds/dw:
//   s0 = sin(uq0) + uq2*uq3 - w0 + 0.3*w1,  s1 = uq1^2 + 0.5*uq4 + 0.5 - 0.2*w0 - w1
//   f0 = uh0*w0 + uq2 + 0.5*w1^2,  f1 = w0*w1 + sin(uh1) + uq1*uq4
// (wEquation forms its Newton matrix as (alpha*dtfactor + beta) - ds/dw entry
// by entry, which equals the Jacobian of the w residual only on the diagonal
// when ncw > 1 and alpha*dtfactor + beta != 0; the zero shift avoids that.)
struct WCoupled : exasim::ModelDefaults<WCoupled> {
    static constexpr int nd = 2, ncu = 2, ncq = ncu * nd, ncw = 2, nparam = 1, nfint = 2, nco = 0;
    static constexpr auto disc = exasim::Discretization::HDG;
    static constexpr bool has_external_coupling = true;

    KOKKOS_INLINE_FUNCTION static void flux(double f[], const double[],
            const double[], const double[], const double[], const double[],
            const double[], double)
    {
        for (int k = 0; k < ncu * nd; ++k) f[k] = 0;
    }

    KOKKOS_INLINE_FUNCTION static void sourcew(double s[], const double[],
            const double uq[], const double[], const double w[], const double[],
            const double[], double)
    {
        s[0] = sin(uq[0]) + uq[2] * uq[3] - w[0] + 0.3 * w[1];
        s[1] = uq[1] * uq[1] + 0.5 * uq[4] + 0.5 - 0.2 * w[0] - w[1];
    }

    // Layout [j*ncw + o] = d s[o] / d uq[j].
    KOKKOS_INLINE_FUNCTION static void sourcew_jac_uq(double s[], const double[],
            const double uq[], const double[], const double[], const double[],
            const double[], double)
    {
        for (int k = 0; k < 12; ++k) s[k] = 0;
        s[0 * 2 + 0] = cos(uq[0]);
        s[2 * 2 + 0] = uq[3];
        s[3 * 2 + 0] = uq[2];
        s[1 * 2 + 1] = 2.0 * uq[1];
        s[4 * 2 + 1] = 0.5;
    }

    KOKKOS_INLINE_FUNCTION static void sourcew_jac_w(double s[], const double[],
            const double[], const double[], const double[], const double[],
            const double[], double)
    {
        s[0 * 2 + 0] = -1.0;
        s[0 * 2 + 1] = -0.2;
        s[1 * 2 + 0] = 0.3;
        s[1 * 2 + 1] = -1.0;
    }

    KOKKOS_INLINE_FUNCTION static void fint(double f[], int, const double[],
            const double uq[], const double[], const double w[], const double uh[],
            const double[], const double[], const double[], const double[], double)
    {
        f[0] = uh[0] * w[0] + uq[2] + 0.5 * w[1] * w[1];
        f[1] = w[0] * w[1] + sin(uh[1]) + uq[1] * uq[4];
    }

    KOKKOS_INLINE_FUNCTION static void fint_jac_uq(double j[], int, const double[],
            const double uq[], const double[], const double[], const double[],
            const double[], const double[], const double[], const double[], double)
    {
        for (int k = 0; k < 12; ++k) j[k] = 0;
        j[0 + 2 * 2] = 1.0;
        j[1 + 2 * 1] = uq[4];
        j[1 + 2 * 4] = uq[1];
    }

    KOKKOS_INLINE_FUNCTION static void fint_jac_uh(double j[], int, const double[],
            const double[], const double[], const double w[], const double uh[],
            const double[], const double[], const double[], const double[], double)
    {
        j[0 + 2 * 0] = w[0];
        j[1 + 2 * 0] = 0;
        j[0 + 2 * 1] = 0;
        j[1 + 2 * 1] = cos(uh[1]);
    }

    // Layout [w*nfint + o] = d f[o] / d w.
    KOKKOS_INLINE_FUNCTION static void fint_jac_w(double j[], int, const double[],
            const double[], const double[], const double w[], const double uh[],
            const double[], const double[], const double[], const double[], double)
    {
        j[0 + 2 * 0] = uh[0];
        j[1 + 2 * 0] = w[1];
        j[0 + 2 * 1] = w[1];
        j[1 + 2 * 1] = w[0];
    }
};

// Nonlinear two-component coupling without q. This exercises the ncq == 0
// trace-correction, residual, condensation and matvec path on mixed families.
struct QFreeCoupled : exasim::ModelDefaults<QFreeCoupled> {
    static constexpr int nd = 2, ncu = 2, ncq = 0, ncw = 0, nparam = 1, nfint = 2;
    static constexpr auto disc = exasim::Discretization::HDG;
    static constexpr bool has_external_coupling = true;

    KOKKOS_INLINE_FUNCTION static void flux(double f[], const double[],
            const double[], const double[], const double[], const double[],
            const double[], double)
    {
        for (int k = 0; k < ncu * nd; ++k) f[k] = 0;
    }

    KOKKOS_INLINE_FUNCTION static void fint(double f[], int, const double[],
            const double uq[], const double[], const double[], const double uh[],
            const double[], const double[], const double[], const double[], double)
    {
        f[0] = uq[0] * uh[0] + 0.25 * uq[1] * uq[1] + sin(uh[1]);
        f[1] = uq[1] * uh[1] + 0.5 * uq[0] - uh[0] * uh[1];
    }

    KOKKOS_INLINE_FUNCTION static void fint_jac_uq(double j[], int, const double[],
            const double uq[], const double[], const double[], const double uh[],
            const double[], const double[], const double[], const double[], double)
    {
        j[0 + 2 * 0] = uh[0];
        j[1 + 2 * 0] = 0.5;
        j[0 + 2 * 1] = 0.5 * uq[1];
        j[1 + 2 * 1] = uh[1];
    }

    KOKKOS_INLINE_FUNCTION static void fint_jac_uh(double j[], int, const double[],
            const double uq[], const double[], const double[], const double uh[],
            const double[], const double[], const double[], const double[], double)
    {
        j[0 + 2 * 0] = uq[0];
        j[1 + 2 * 0] = -uh[1];
        j[0 + 2 * 1] = cos(uh[1]);
        j[1 + 2 * 1] = uq[1] - uh[0];
    }
};

// Deterministic value in [0, 1).
dstype hv(long i, long seed)
{
    const double x = std::sin(0.7311 * static_cast<double>(i) + 1.37 * static_cast<double>(seed) + 0.123) *
                     43758.5453;
    return x - std::floor(x);
}

// ---------------------------------------------------------------------------
// Local HDG mesh of one strip: elements without an interface face first, then
// the interface column (one interface face each, in row order), so that
// [ne0, ne) is the conformal interface block and intfaces[k] lies on element
// ne0 + k. Faces are matched by coordinates; elemcon is identity for the first
// element of a face and the matched node order for the second.
struct LocalMesh {
    int ne = 0, ne0 = 0, ne1 = 0, nf = 0, neI = 0; // ne1 < ne when ghost elements follow
    std::vector<Int> nbsd, elemsend, elemrecv, elemsendpts, elemrecvpts; // element halo (same model)
    std::vector<Int> elempart, bf, f2e, elemcon, intfaces, boufaces;
    std::vector<dstype> xdg;
};

// Faces of all local elements, matched by coordinates. Elements are visited in
// local order, so a face touching an owned element has an owned first element.
void build_faces(LocalMesh& lm, const MasterData& m, const std::string& tag)
{
    const int npe = m.npe, npf = m.npf, nfe = m.nfe, nd = m.nd;
    std::map<std::pair<long long, long long>, int> faceOf;
    std::vector<dstype> faceNodes;
    lm.f2e.clear();
    lm.elemcon.assign(static_cast<std::size_t>(npf) * nfe * lm.ne, -1);
    lm.nf = 0;
    for (int e = 0; e < lm.ne; e++)
        for (int lf = 0; lf < nfe; lf++) {
            std::vector<dstype> xf(static_cast<std::size_t>(npf) * nd);
            dstype cx = 0, cy = 0;
            for (int j = 0; j < npf; j++) {
                const int node = m.perm[j + npf * lf];
                xf[j] = lm.xdg[node + npe * 0 + npe * nd * e];
                xf[j + npf] = lm.xdg[node + npe * 1 + npe * nd * e];
                cx += xf[j] / npf;
                cy += xf[j + npf] / npf;
            }
            const auto key = std::make_pair(std::llround(cx * 1e9), std::llround(cy * 1e9));
            auto it = faceOf.find(key);
            if (it == faceOf.end()) {
                const int f = lm.nf++;
                faceOf[key] = f;
                lm.f2e.insert(lm.f2e.end(), {e, lf, -1, -1});
                faceNodes.insert(faceNodes.end(), xf.begin(), xf.end());
                for (int j = 0; j < npf; j++) lm.elemcon[j + npf * lf + npf * nfe * e] = npf * f + j;
            }
            else {
                const int f = it->second;
                check(lm.f2e[4 * f + 2] == -1, tag + ": a face has at most two elements");
                lm.f2e[4 * f + 2] = e;
                lm.f2e[4 * f + 3] = lf;
                for (int j = 0; j < npf; j++) {
                    int match = -1;
                    for (int k = 0; k < npf; k++)
                        if (std::abs(xf[j] - faceNodes[k + 2 * npf * f]) < 1e-10 &&
                            std::abs(xf[j + npf] - faceNodes[k + npf + 2 * npf * f]) < 1e-10)
                            match = k;
                    check(match >= 0, tag + ": shared face nodes match");
                    lm.elemcon[j + npf * lf + npf * nfe * e] = npf * f + std::max(match, 0);
                }
            }
        }
}

void build_local_mesh(LocalMesh& lm, const ModelMesh& mm, const MasterData& m, int part, int nparts,
                      const std::string& tag)
{
    const int nfe = m.nfe;
    std::vector<int> inner, column;
    std::vector<dstype> xe;
    std::vector<int> bfe;
    for (int ge = 0; ge < mm.ne(); ge++) {
        if ((mm.row(ge) * nparts) / mm.ny() != part) continue;
        element_data(xe, bfe, mm, m, ge);
        bool onInterface = false;
        for (int lf = 0; lf < nfe; lf++) onInterface = onInterface || bfe[lf] == mm.bcInterface;
        (onInterface ? column : inner).push_back(ge);
    }
    std::vector<int> order(inner);
    order.insert(order.end(), column.begin(), column.end());
    lm.ne = static_cast<int>(order.size());
    lm.ne0 = static_cast<int>(inner.size());
    lm.elempart.clear(); lm.bf.clear(); lm.xdg.clear();
    for (int ge : order) {
        element_data(xe, bfe, mm, m, ge);
        lm.elempart.push_back(ge);
        lm.xdg.insert(lm.xdg.end(), xe.begin(), xe.end());
        lm.bf.insert(lm.bf.end(), bfe.begin(), bfe.end());
    }

    lm.ne1 = lm.ne;
    build_faces(lm, m, tag);

    lm.intfaces.clear();
    lm.boufaces.clear();
    for (int e = 0; e < lm.ne; e++)
        for (int lf = 0; lf < nfe; lf++)
            if (lm.bf[lf + nfe * e] == mm.bcInterface) lm.intfaces.push_back(nfe * e + lf);
    lm.neI = static_cast<int>(lm.intfaces.size());
    for (int k = 0; k < lm.neI; k++) {
        const int e = lm.intfaces[k] / nfe, lf = lm.intfaces[k] % nfe;
        check(e == lm.ne0 + k, tag + ": interface face k lies on element ne0 + k");
        lm.boufaces.push_back(lf + nfe * k);
    }
}

// Strip with its element halo. Owned elements are the rows of this part; the
// ghost elements are the adjacent rows of the neighboring parts. Local order:
// owned elements without a ghost neighbor [0, ne0), owned elements next to a
// ghost row [ne0, ne1), ghosts [ne1, ne); each group by global element. The
// send list for a neighbor holds the owned row next to it, the receive list
// the ghost row it owns, both by global element, so the lists match pairwise.
void build_halo_mesh(LocalMesh& lm, const ModelMesh& mm, const MasterData& m, int part, int nparts,
                     const std::string& tag)
{
    const int nfe = m.nfe;
    auto partOf = [&](int row) { return (row * nparts) / mm.ny(); };
    int rowLo = mm.ny(), rowHi = -1;
    for (int row = 0; row < mm.ny(); row++)
        if (partOf(row) == part) { rowLo = std::min(rowLo, row); rowHi = std::max(rowHi, row); }
    const bool below = rowLo > 0, above = rowHi < mm.ny() - 1;
    std::vector<int> inner, edge, ghostBelow, ghostAbove;
    for (int ge = 0; ge < mm.ne(); ge++) {
        const int row = mm.row(ge);
        if (row >= rowLo && row <= rowHi)
            ((below && row == rowLo) || (above && row == rowHi) ? edge : inner).push_back(ge);
        else if (below && row == rowLo - 1) ghostBelow.push_back(ge);
        else if (above && row == rowHi + 1) ghostAbove.push_back(ge);
    }
    std::vector<int> order(inner);
    order.insert(order.end(), edge.begin(), edge.end());
    order.insert(order.end(), ghostBelow.begin(), ghostBelow.end());
    order.insert(order.end(), ghostAbove.begin(), ghostAbove.end());
    lm.ne = static_cast<int>(order.size());
    lm.ne0 = static_cast<int>(inner.size());
    lm.ne1 = lm.ne0 + static_cast<int>(edge.size());
    std::map<int, int> localOf;
    std::vector<dstype> xe;
    std::vector<int> bfe;
    lm.elempart.clear(); lm.bf.clear(); lm.xdg.clear();
    for (int e = 0; e < lm.ne; e++) {
        const int ge = order[e];
        localOf[ge] = e;
        element_data(xe, bfe, mm, m, ge);
        lm.elempart.push_back(ge);
        lm.xdg.insert(lm.xdg.end(), xe.begin(), xe.end());
        lm.bf.insert(lm.bf.end(), bfe.begin(), bfe.end());
    }
    lm.nbsd.clear(); lm.elemsend.clear(); lm.elemrecv.clear(); lm.elemsendpts.clear(); lm.elemrecvpts.clear();
    auto addNeighbor = [&](int neighborPart, int ownRow, const std::vector<int>& ghosts) {
        lm.nbsd.push_back(neighborPart);
        int nsend = 0;
        for (int ge : edge)
            if (mm.row(ge) == ownRow) { lm.elemsend.push_back(localOf[ge]); nsend++; }
        for (int ge : ghosts) lm.elemrecv.push_back(localOf[ge]);
        lm.elemsendpts.push_back(nsend);
        lm.elemrecvpts.push_back(static_cast<int>(ghosts.size()));
        check(partOf(mm.row(ghosts.front())) == neighborPart, tag + ": ghost row owned by the neighbor");
    };
    if (below) addNeighbor(part - 1, rowLo, ghostBelow);
    if (above) addNeighbor(part + 1, rowHi, ghostAbove);

    build_faces(lm, m, tag);
    lm.intfaces.clear();
    lm.boufaces.clear();
    for (int e = 0; e < lm.ne1; e++)
        for (int lf = 0; lf < nfe; lf++)
            if (lm.bf[lf + nfe * e] == mm.bcInterface) lm.intfaces.push_back(nfe * e + lf);
    lm.neI = static_cast<int>(lm.intfaces.size());
}

// Owned elements only (the input of the non-matching setup).
LocalMesh owned_part(const LocalMesh& lm, const MasterData& m)
{
    LocalMesh o = lm;
    o.ne = lm.ne1;
    o.elempart.resize(lm.ne1);
    o.bf.resize(static_cast<std::size_t>(m.nfe) * lm.ne1);
    o.xdg.resize(static_cast<std::size_t>(m.npe) * m.nd * lm.ne1);
    return o;
}

void fill_rank_structs(RankStructs& rs, const LocalMesh& lm, MasterData& m, int ncu, int ncu12,
                       int interfacetype, int ncq = -1)
{
    rs.problem.assign(35, 0);
    rs.problem[28] = 1; rs.problem[29] = 1; rs.problem[30] = 2;
    rs.problem[34] = interfacetype;
    rs.appnsize.assign(30, 0);
    rs.appnsize[2] = 35;
    rs.appnsize[14] = ncu12;
    rs.appndims.assign(30, 0);
    rs.appndims[AppNdims::nd] = m.nd;
    const int localNcq = ncq >= 0 ? ncq : ncu * m.nd;
    rs.appndims[AppNdims::nc] = ncu + localNcq;
    rs.appndims[AppNdims::ncu] = ncu;
    rs.appndims[AppNdims::ncq] = localNcq;
    rs.appndims[AppNdims::ncx] = m.nd;
    rs.elempart = lm.elempart;
    rs.bf = lm.bf;
    rs.xdg = lm.xdg;
    rs.meshndims.assign(20, 0);
    rs.meshndims[0] = m.nd; rs.meshndims[1] = lm.ne; rs.meshndims[4] = m.nfe;
    rs.perm.assign(m.perm.begin(), m.perm.end());
    rs.app = appstructT<dstype, Int>();
    rs.master = masterstructT<dstype, Int>();
    rs.mesh = meshstructT<dstype, Int>();
    rs.sol = solstructT<dstype, Int>();
    rs.app.problem = rs.problem.data();
    rs.app.nsize = rs.appnsize.data();
    rs.app.ndims = rs.appndims.data();
    rs.master.ndims = m.ndims.data();
    rs.master.shapfgt = m.shapfgt.data();
    rs.master.gpf = m.gpf.data();
    rs.master.gwf = m.gwf.data();
    rs.master.xpf = m.xpf.data();
    rs.master.xpe = m.xpe.data();
    rs.mesh.ndims = rs.meshndims.data();
    rs.mesh.bf = rs.bf.data();
    rs.mesh.perm = rs.perm.data();
    rs.mesh.elempart = rs.elempart.data();
    rs.mesh.szelempart = lm.ne;
    rs.sol.xdg = rs.xdg.data();
}

// ---------------------------------------------------------------------------
// Solver state of one rank. The structs hold pointers into the vectors, so an
// object is set up in place and never copied.
struct RankOperator {
    Int nd = 0, npe = 0, npf = 0, nfe = 0, ngf = 0, ncu = 0, ncq = 0, nc = 0, ncw = 0, ncu12 = 0;
    Int ne = 0, ne0 = 0, nf = 0, neI = 0, m = 0, n = 0, m12 = 0, N = 0;
    std::vector<dstype> udg0, uh0, udg, uh, wdg, wsrc, C, E, F, Ru, AE, PE, elemfaceg;
    std::vector<dstype> shapfgw, shapfgwdotshapfg, tau, param, uinf;
    std::vector<Int> f2e, elemcon, perm, intfaces, boufaces, map, eblks, nboufaces, ipiv, tempi;
    std::vector<dstype> tempn, tempg, ve, we, buffsend, buffrecv, faceSend, faceRecv;
    std::vector<dstype> confRi, confHi, confKi, confGi;
    std::vector<MPI_Request> requests;
    std::vector<MPI_Status> statuses;
    std::vector<Int> nbintf, facesendpts, facerecvpts;
    commonstruct common;
    appstruct app;
    masterstruct master;
    meshstruct mesh;
    solstruct sol;
    resstruct res;
    tempstruct tmp;

    RankOperator() = default;
    RankOperator(const RankOperator&) = delete;
    RankOperator& operator=(const RankOperator&) = delete;
};

template <class Model>
void init_operator(RankOperator& op, const LocalMesh& lm, const MasterData& mst, int model)
{
    op.nd = mst.nd; op.npe = mst.npe; op.npf = mst.npf; op.nfe = mst.nfe; op.ngf = mst.ngf;
    op.ncu = Model::ncu; op.ncq = Model::ncq; op.nc = op.ncu + op.ncq;
    op.ncw = Model::ncw; op.ncu12 = Model::nfint;
    op.ne = lm.ne; op.ne0 = lm.ne0; op.nf = lm.nf; op.neI = lm.neI;
    op.n = op.npe * op.ncu; op.m = op.npf * op.nfe * op.ncu; op.m12 = op.npf * op.ncu12;
    op.N = op.ncu * op.npf * op.nf;
    const Int npe = op.npe, npf = op.npf, nfe = op.nfe, ngf = op.ngf, nd = op.nd;
    const Int ne = op.ne, nc = op.nc, ncu = op.ncu, n = op.n, m = op.m;

    // Master: shapfgw = npf x ngf (weighted), shapfgwdotshapfg = npf*npf x ngf.
    op.shapfgw.assign(static_cast<std::size_t>(npf) * ngf, 0);
    op.shapfgwdotshapfg.assign(static_cast<std::size_t>(npf) * npf * ngf, 0);
    for (Int g = 0; g < ngf; g++)
        for (Int j = 0; j < npf; j++)
            op.shapfgw[j + npf * g] = mst.shapfgt[g + ngf * j] * mst.gwf[g];
    for (Int g = 0; g < ngf; g++)
        for (Int i = 0; i < npf; i++)
            for (Int j = 0; j < npf; j++)
                op.shapfgwdotshapfg[j + npf * i + npf * npf * g] =
                    op.shapfgw[j + npf * g] * mst.shapfgt[g + ngf * i];
    op.master = masterstruct();
    op.master.shapfgt = const_cast<dstype*>(mst.shapfgt.data());
    op.master.shapfgw = op.shapfgw.data();
    op.master.shapfgwdotshapfg = op.shapfgwdotshapfg.data();
    op.master.gwf = const_cast<dstype*>(mst.gwf.data());

    op.f2e = lm.f2e; op.elemcon = lm.elemcon; op.intfaces = lm.intfaces; op.boufaces = lm.boufaces;
    op.perm.assign(mst.perm.begin(), mst.perm.end());
    op.mesh = meshstruct();
    op.mesh.f2e = op.f2e.data();
    op.mesh.elemcon = op.elemcon.data();
    op.mesh.perm = op.perm.data();
    op.mesh.intfaces = op.intfaces.data();
    op.mesh.szintfaces = op.neI;
    op.mesh.boufaces = op.boufaces.data();

    op.map.resize(op.ncu12);
    for (Int c = 0; c < op.ncu12; c++) op.map[c] = c;
    op.tau.assign(ncu, 1.0); op.param.assign(1, 1.0); op.uinf.assign(1, 0.0);
    op.app = appstruct();
    op.app.interfacefluxmap = op.map.data();
    op.app.tau = op.tau.data();
    op.app.physicsparam = op.param.data();
    op.app.uinf = op.uinf.data();
    op.app.materialdb_nprop = 0;

    commonstruct& c = op.common;
    c.backend = 0;
    c.outputparams.debugMode = 0;
    c.components.nc = nc; c.components.ncu = ncu; c.components.ncq = op.ncq;
    c.components.nco = 0; c.components.ncw = op.ncw; c.components.ncx = nd;
    c.grid.nd = nd; c.grid.npe = npe; c.grid.npf = npf; c.grid.ngf = ngf;
    c.meshsizes.ne = ne; c.meshsizes.ne0 = op.ne0; c.meshsizes.ne1 = lm.ne1;
    c.meshsizes.nf = op.nf; c.meshsizes.nf0 = op.nf; c.meshsizes.nfe = nfe;
    c.meshsizes.maxnbc = 2; c.meshsizes.nbe = 2;
    c.spatialScheme = 1;
    c.timeparams.wave = 0; c.timeparams.dae_alpha = 0.0; c.timeparams.dae_beta = 0.0;
    c.timestate.dtfactor = 1.0; c.timestate.time = 0.0;
    c.couplingparams.coupledcondition = 1;
    c.couplingparams.coupledboundarycondition = 2;
    c.couplingparams.coupledinterface = 1;
    c.couplingparams.nintfaces = op.neI;
    c.szinterfacefluxmap = op.ncu12;
    c.nnbsd = 0; c.nelemsend = 0; c.nelemrecv = 0;
    op.requests.resize(256);
    op.statuses.resize(256);
    c.requests = op.requests.data();
    c.statuses = op.statuses.data();
    op.eblks = {1, op.ne0, 0, op.ne0 + 1, ne, -1};
    op.nboufaces = {0, 0, 0, 0, op.neI};
    c.eblks = op.eblks.data();
    c.nboufaces = op.nboufaces.data();

    // Nodal state, seeded by the global element number and the model.
    op.udg0.assign(static_cast<std::size_t>(npe) * nc * ne, 0);
    for (Int e = 0; e < ne; e++)
        for (Int k = 0; k < nc; k++)
            for (Int j = 0; j < npe; j++)
                op.udg0[j + npe * (k + nc * e)] =
                    0.2 + 0.6 * hv(j + 31 * k + 977 * lm.elempart[e], 11 + model);
    op.uh0.assign(op.N, 0);
    for (Int i = 0; i < op.N; i++) op.uh0[i] = 0.1 + 0.7 * hv(i, 23 + model);
    op.udg = op.udg0;
    op.uh = op.uh0;
    op.wdg.assign(static_cast<std::size_t>(npe) * std::max<Int>(op.ncw, 1) * ne, 0);
    op.wsrc.assign(op.wdg.size(), 0);

    // Local-solver linearization: q_d = C_d u - E_d uh, u = DinvF uh.
    op.C.assign(static_cast<std::size_t>(npe) * npe * ne * nd, 0);
    op.E.assign(static_cast<std::size_t>(npe) * npf * nfe * ne * nd, 0);
    op.F.assign(static_cast<std::size_t>(n) * m * ne, 0);
    op.Ru.assign(static_cast<std::size_t>(n) * ne, 0);
    for (std::size_t i = 0; i < op.C.size(); i++) op.C[i] = 0.2 * (hv(i, 31 + model) - 0.5);
    for (std::size_t i = 0; i < op.E.size(); i++) op.E[i] = 0.2 * (hv(i, 37 + model) - 0.5);
    for (std::size_t i = 0; i < op.F.size(); i++) op.F[i] = 0.1 * (hv(i, 41 + model) - 0.5);
    for (std::size_t i = 0; i < op.Ru.size(); i++) op.Ru[i] = 0.05 * (hv(i, 43 + model) - 0.5);

    // Element trace matrices (diagonally dominant) and the elemental-block
    // preconditioner used through hdgMatVec.
    op.AE.assign(static_cast<std::size_t>(m) * m * ne, 0);
    op.PE.assign(static_cast<std::size_t>(m) * m * ne, 0);
    for (Int e = 0; e < ne; e++)
        for (Int j = 0; j < m; j++)
            for (Int i = 0; i < m; i++) {
                const std::size_t k = i + static_cast<std::size_t>(m) * (j + static_cast<std::size_t>(m) * e);
                op.AE[k] = (i == j) ? 4.0 + hv(k, 47 + model) : 0.1 * (hv(k, 53 + model) - 0.5);
                if (i == j) op.PE[k] = 0.5 / op.AE[k];
            }

    op.sol = solstruct();
    op.sol.udg = op.udg.data();
    op.sol.uh = op.uh.data();
    op.sol.wdg = op.ncw > 0 ? op.wdg.data() : nullptr;
    op.sol.wsrc = op.ncw > 0 ? op.wsrc.data() : nullptr;
    op.sol.odg = nullptr;

    op.res = resstruct();
    op.res.C = op.C.data();
    op.res.E = op.E.data();
    op.res.F = op.F.data();
    op.res.Ru = op.Ru.data();

    op.tempn.assign(1 << 21, 0);
    op.tempg.assign(1 << 21, 0);
    op.tempi.assign(1 << 16, 0);
    op.ipiv.assign(static_cast<std::size_t>(n) * std::max<Int>(op.neI, 1), 0);
    op.tmp = tempstruct();
    op.tmp.tempn = op.tempn.data();
    op.tmp.tempg = op.tempg.data();
    op.tmp.tempi = op.tempi.data();
    op.ve.assign(static_cast<std::size_t>(m) * ne, 0);
    op.we.assign(static_cast<std::size_t>(m) * ne, 0);
    op.buffsend.assign(1, 0);
    op.buffrecv.assign(1, 0);
    op.tmp.buffsend = op.buffsend.data();
    op.tmp.buffrecv = op.buffrecv.data();
}

// Select the non-matching (type 1) coupling.
void use_nonmatching(RankOperator& op)
{
    op.common.couplingparams.interfacetype = 1;
    op.common.nnbintf = 0;
    op.common.nfacesend = 0;
    op.common.nfacerecv = 0;
}

// Select the conformal (type 0) coupling with the arrays of
// build_runtime_interface_partition.
void use_conformal(RankOperator& op, const meshstruct& part)
{
    op.common.couplingparams.interfacetype = 0;
    op.nbintf.assign(part.nbintf, part.nbintf + part.sznbintf);
    op.facesendpts.assign(part.facesendpts, part.facesendpts + part.szfacesendpts);
    op.facerecvpts.assign(part.facerecvpts, part.facerecvpts + part.szfacerecvpts);
    op.common.nnbintf = part.sznbintf;
    op.common.nfacesend = part.szfacesend;
    op.common.nfacerecv = part.szfacerecv;
    op.common.nbintf = op.nbintf.data();
    op.common.facesendpts = op.facesendpts.data();
    op.common.facerecvpts = op.facerecvpts.data();
    op.mesh.facesend = part.facesend;
    op.mesh.facerecv = part.facerecv;
    op.mesh.faceperm = part.faceperm;
    op.faceSend.assign(static_cast<std::size_t>(op.m12) * std::max<Int>(part.szfacesend, 1), 0);
    op.faceRecv.assign(static_cast<std::size_t>(op.m12) * std::max<Int>(part.szfacerecv, 1), 0);
    op.tmp.bufffacesend = op.faceSend.data();
    op.tmp.bufffacerecv = op.faceRecv.data();
    op.res.Hi = op.confHi.data();
    op.res.Ri = op.confRi.data();
}

void matvec(RankOperator& op, std::vector<dstype>& w, std::vector<dstype>& AE, const std::vector<dstype>& v,
            nonmatchingdata* nm,
            const CompositeOwnerQuadrature<dstype, Int>* quad = nullptr)
{
    w.assign(op.N, 0);
    std::vector<dstype> vin(v);
    exasim::nonmatching::NonmatchingRuntime<dstype, Int> runtime;
    exasim::nonmatching::NonmatchingRuntime<dstype, Int>* runtimePointer = nullptr;
    if (nm) {
        exasim::nonmatching::refresh_nonmatching_runtime(
                runtime, *nm, op.app, op.mesh, op.common, Int(0), quad);
        runtimePointer = &runtime;
    }
    hdgMatVec<dstype, Int>(w.data(), AE.data(), vin.data(), op.ve.data(), op.we.data(), op.res, op.app,
                           op.mesh, op.common, op.tmp, 0, 0, runtimePointer);
}

// Element trace vector of v (ncu x npf x nfe per element), gathered directly.
void gather_trace(const RankOperator& op, std::vector<dstype>& ve, const std::vector<dstype>& v)
{
    const Int ndf = op.npf * op.nfe;
    ve.assign(static_cast<std::size_t>(op.m) * op.ne, 0);
    for (Int e = 0; e < op.ne; e++)
        for (Int i = 0; i < ndf; i++)
            for (Int c = 0; c < op.ncu; c++)
                ve[c + op.ncu * (i + ndf * e)] = v[c + op.ncu * op.elemcon[i + ndf * e]];
}

// we = AE * ve per element (plain loops).
void element_products(const RankOperator& op, std::vector<dstype>& we, const std::vector<dstype>& AE,
                      const std::vector<dstype>& ve)
{
    const Int m = op.m;
    we.assign(static_cast<std::size_t>(m) * op.ne, 0);
    for (Int e = 0; e < op.ne; e++)
        for (Int j = 0; j < m; j++)
            for (Int i = 0; i < m; i++)
                we[i + m * e] += AE[i + static_cast<std::size_t>(m) * (j + static_cast<std::size_t>(m) * e)] *
                                 ve[j + m * e];
}

// Scatter of the element vector to the faces through elemcon.
void scatter_faces(const RankOperator& op, std::vector<dstype>& w, const std::vector<dstype>& we)
{
    const Int ndf = op.npf * op.nfe;
    w.assign(op.N, 0);
    for (Int e = 0; e < op.ne; e++)
        for (Int i = 0; i < ndf; i++)
            for (Int c = 0; c < op.ncu; c++)
                w[c + op.ncu * op.elemcon[i + ndf * e]] += we[c + op.ncu * (i + ndf * e)];
}

dstype global_dot(const std::vector<dstype>& a, const std::vector<dstype>& b, MPI_Comm comm)
{
    dstype local = 0, global = 0;
    for (std::size_t i = 0; i < a.size(); i++) local += a[i] * b[i];
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
    return global;
}

dstype global_max_abs_diff(const std::vector<dstype>& a, const std::vector<dstype>& b, MPI_Comm comm)
{
    dstype local = 0, global = 0;
    for (std::size_t i = 0; i < a.size(); i++) local = std::max(local, std::abs(a[i] - b[i]));
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, comm);
    return global;
}

dstype global_max_abs(const std::vector<dstype>& a, MPI_Comm comm)
{
    std::vector<dstype> zero(a.size(), 0);
    return global_max_abs_diff(a, zero, comm);
}

// Independent owner-side coupling contribution: direct pair products summed
// per slot on the donor, all-gathered with (direction, owner rank, owner
// global face), and added by the owner at its face with that global index.
// Also counts faces with |K_l| > 1 and |R_l| > 1.
void reference_coupling(const RankOperator& op, const nonmatchingdata& nm, const std::vector<dstype>& ve,
                        std::vector<dstype>& weRef, int myRank, MPI_Comm comm, long& multiK, long& multiR)
{
    const Int m = op.m, m12 = op.m12;
    const int rec = 4 + m12;
    std::vector<double> mine;
    multiK = 0;
    for (int direction = 0; direction < 2; direction++) {
        const nonmatchinginterface& d = nm.dir[direction];
        if (!d.isdonor) continue;
        for (Int s = 0; s < d.nslots; s++) {
            // Accumulate the independent dense reference above production
            // precision so its own summation error does not consume the fixed
            // round-off acceptance budget for multi-pair slots.
            std::vector<long double> vals(m12, 0.0L);
            int npairs = 0;
            for (Int q = 0; q < d.npairs; q++) {
                if (d.pairslot[q] != s) continue;
                npairs++;
                for (Int col = 0; col < m; col++)
                    for (Int r = 0; r < m12; r++)
                        vals[r] += d.Hi[r + static_cast<std::size_t>(m12) * (col + static_cast<std::size_t>(m) * q)] *
                                   ve[col + static_cast<std::size_t>(m) * d.pairelem[q]];
            }
            if (npairs > 1) multiK++;
            mine.push_back(direction);
            mine.push_back(d.slotrank[s]);
            mine.push_back(d.slotface[s]);
            mine.push_back(myRank);
            for (long double value : vals) mine.push_back(static_cast<double>(value));
        }
    }
    int count = static_cast<int>(mine.size()), size = 0;
    MPI_Comm_size(comm, &size);
    std::vector<int> counts(size), displs(size, 0);
    MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    int total = 0;
    for (int r = 0; r < size; r++) { displs[r] = total; total += counts[r]; }
    std::vector<double> all(static_cast<std::size_t>(std::max(total, 1)));
    MPI_Allgatherv(mine.data(), count, MPI_DOUBLE, all.data(), counts.data(), displs.data(), MPI_DOUBLE, comm);

    std::map<std::pair<int, long>, std::vector<int>> donorsOf;
    for (int i = 0; i + rec <= total; i += rec) {
        const int direction = static_cast<int>(all[i]);
        if (static_cast<int>(all[i + 1]) != myRank) continue;
        const nonmatchinginterface& owner = nm.dir[direction];
        const long face = static_cast<long>(all[i + 2]);
        int l = -1;
        for (Int k = 0; k < owner.nownfaces; k++)
            if (owner.ownfaceglobal[k] == face) l = k;
        check(owner.isowner && l >= 0, "reference: owner face of a slot found");
        if (l < 0) continue;
        donorsOf[std::make_pair(direction, face)].push_back(static_cast<int>(all[i + 3]));
        const Int encoded = op.intfaces[owner.ownface[l]];
        const Int e = encoded / op.nfe, lf = encoded % op.nfe;
        for (Int i2 = 0; i2 < op.npf; i2++)
            for (Int c = 0; c < op.ncu12; c++)
                weRef[op.map[c] + op.ncu * (i2 + op.npf * (lf + op.nfe * e))] +=
                    all[i + 4 + c + op.ncu12 * i2];
    }
    multiR = 0;
    for (auto& kv : donorsOf) {
        std::sort(kv.second.begin(), kv.second.end());
        if (std::unique(kv.second.begin(), kv.second.end()) - kv.second.begin() > 1) multiR++;
    }
}

// AE * ve(sol.uh) minus the uncondensed coupling residual of the current sol
// state (production residual, exchange and owner assembly); scattered to faces.
template <class Model>
void coupled_residual(RankOperator& op, nonmatchingdata& nm, std::vector<dstype>& R,
        const CompositeOwnerQuadrature<dstype, Int>* quad = nullptr)
{
    exasim::nonmatching::compute_nonmatching_coupling<Model>(nm, op.sol, op.res, op.app, op.master, op.mesh,
            op.tmp, op.common, 0, false, 0, quad);
    Int count = 0;
    exasim::nonmatching::begin_coupling_slot_exchange(nm, op.common, count);
    MPI_Waitall(count, op.common.requests, op.common.statuses);
    // Exasim stores residuals with the sign of the right-hand side
    // (donor.Ri = -sum W f, like res.Rh and the conformal res.Ri) and the
    // matrices as +dR/du, so the operator is the derivative of AE*ve - Ri.
    std::vector<dstype> ve, we, wc(static_cast<std::size_t>(op.m) * op.ne, 0);
    gather_trace(op, ve, op.uh);
    element_products(op, we, op.AE, ve);
    exasim::nonmatching::assemble_coupling_slot_owner(wc.data(), nm, op.mesh, op.app, op.common);
    for (std::size_t i = 0; i < we.size(); i++) we[i] -= wc[i];
    scatter_faces(op, R, we);
}

// Local-solver tangent of the nodal state for the trace direction ve:
// delta u = DinvF ve, delta q_d = C_d delta u - E_d ve.
void state_tangent(const RankOperator& op, const std::vector<dstype>& ve, std::vector<dstype>& dudg)
{
    const Int npe = op.npe, ncu = op.ncu, nc = op.nc, n = op.n, m = op.m, ne = op.ne;
    const Int ndf = op.npf * op.nfe;
    dudg.assign(static_cast<std::size_t>(npe) * nc * ne, 0);
    for (Int e = 0; e < ne; e++) {
        std::vector<dstype> du(n, 0);
        for (Int col = 0; col < m; col++)
            for (Int r = 0; r < n; r++)
                du[r] += op.F[r + static_cast<std::size_t>(n) * (col + static_cast<std::size_t>(m) * e)] *
                         ve[col + m * e];
        for (Int c = 0; c < ncu; c++)
            for (Int j = 0; j < npe; j++) dudg[j + npe * (c + nc * e)] = du[j + npe * c];
        if (op.ncq > 0)
            for (Int d = 0; d < op.nd; d++)
                for (Int c = 0; c < ncu; c++)
                    for (Int k = 0; k < npe; k++) {
                        dstype dq = 0;
                        for (Int j = 0; j < npe; j++)
                            dq += op.C[k + npe * j + static_cast<std::size_t>(npe) * npe * (e + ne * d)] * du[j + npe * c];
                        for (Int j = 0; j < ndf; j++)
                            dq -= op.E[k + npe * j + static_cast<std::size_t>(npe) * ndf * (e + ne * d)] *
                                  ve[c + ncu * (j + ndf * e)];
                        dudg[k + npe * (exasim::nonmatching::q_component(c, d, ncu) + nc * e)] = dq;
                    }
    }
}

struct SubComm {
    MPI_Comm comm = MPI_COMM_NULL;
    int k = 0, model = 0, part = 0;
    explicit SubComm(int n)
    {
        MPI_Comm_split(MPI_COMM_WORLD, worldRank < n ? 0 : MPI_UNDEFINED, worldRank, &comm);
        if (comm != MPI_COMM_NULL) {
            EXASIM_COMM_WORLD = comm;
            EXASIM_COMM_LOCAL = MPI_COMM_SELF;
            k = n / 2;
            model = worldRank < k ? 0 : 1;
            part = worldRank - model * k;
        }
    }
    ~SubComm()
    {
        if (comm != MPI_COMM_NULL) {
            MPI_Comm_free(&comm);
            EXASIM_COMM_WORLD = MPI_COMM_WORLD;
            EXASIM_COMM_LOCAL = MPI_COMM_WORLD;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
};

// Product reference and finite differences on a non-matching mesh.
template <class Model>
void nonmatching_case(const std::string& name, const ModelMesh& m1, const ModelMesh& m2, MasterData& mst,
                      int n, bool productReference, bool mixedFamilies = false,
                      bool model0Triangle = false)
{
    SubComm sc(n);
    if (sc.comm == MPI_COMM_NULL) return;
    const std::string tag = name + " (" + std::to_string(sc.k) + " per model)";
    MasterData triangle, quad;
    if (mixedFamilies) {
        triangle = make_master(0, mst.porder);
        quad = make_master(1, mst.porder);
    }
    MasterData& localMaster = mixedFamilies ?
            (((sc.model == 0) == model0Triangle) ? triangle : quad) : mst;
    LocalMesh lm;
    build_local_mesh(lm, sc.model == 0 ? m1 : m2, localMaster, sc.part, sc.k, tag);
    RankStructs rs;
    fill_rank_structs(rs, lm, localMaster, Model::ncu, Model::nfint, 1, Model::ncq);
    nonmatchingdata nm;
    CompositeOwnerQuadrature<dstype, Int> sidecar;
    const bool ran = exasim::nonmatching::setup_nonmatching_interface(rs.app, rs.master, rs.mesh, rs.sol,
            static_cast<Int>(n), static_cast<Int>(worldRank), static_cast<Int>(sc.model * sc.k), nm,
            static_cast<DonorSearchResult<dstype, Int>*>(nullptr), DonorSearchParams(), true,
            nullptr, mixedFamilies ? &sidecar : nullptr);
    check(ran && nm.interfacetype == 1, tag + ": setup ran");

    RankOperator op;
    init_operator<Model>(op, lm, localMaster, sc.model);
    use_nonmatching(op);
    check(exasim::nonmatching::validate_coupling_selection(op.common, &nm, Int(0), "test"),
          tag + ": type-1 coupling is active");
    exasim::nonmatching::compute_nonmatching_coupling<Model>(nm, op.sol, op.res, op.app, op.master, op.mesh,
            op.tmp, op.common, 0, true, 0, mixedFamilies ? &sidecar : nullptr);
    exasim::nonmatching::NonmatchingRuntime<dstype, Int> exactRuntime;
    exasim::nonmatching::refresh_nonmatching_runtime(exactRuntime, nm, op.app, op.mesh, op.common, Int(0),
            mixedFamilies ? &sidecar : nullptr);
    exasim::nonmatching::compute_nonmatching_coupling<Model>(exactRuntime, op.sol, op.res, op.app,
            op.master, op.mesh, op.tmp, op.common, 0, true, 0);
    for (int direction = 0; direction < 2; ++direction) {
        const auto& hostDirection = nm.dir[direction];
        const auto& runtimeDirection = exactRuntime.dir[direction];
        Int residualMismatch = 0, matrixMismatch = 0;
        dstype residualDifference = 0, matrixDifference = 0;
        Int firstMatrixMismatch = -1;
        for (Int i = 0; i < hostDirection.szRi; ++i) {
            residualMismatch += hostDirection.Ri[i] != runtimeDirection.Ri.data[i];
            residualDifference = std::max(residualDifference,
                std::abs(hostDirection.Ri[i] - runtimeDirection.Ri.data[i]));
        }
        for (Int i = 0; i < hostDirection.szHi; ++i) {
            const bool differs = hostDirection.Hi[i] != runtimeDirection.Hi.data[i];
            if (differs && firstMatrixMismatch < 0) firstMatrixMismatch = i;
            matrixMismatch += differs;
            matrixDifference = std::max(matrixDifference,
                std::abs(hostDirection.Hi[i] - runtimeDirection.Hi.data[i]));
        }
        if (residualMismatch || matrixMismatch)
            std::fprintf(stderr, "rank %d %s direction %d host/runtime mismatch: Ri %d (%.3e), Hi %d (%.3e), first %d %.17e %.17e\n",
                worldRank, tag.c_str(), direction, residualMismatch, residualDifference,
                matrixMismatch, matrixDifference, firstMatrixMismatch,
                firstMatrixMismatch >= 0 ? hostDirection.Hi[firstMatrixMismatch] : 0,
                firstMatrixMismatch >= 0 ? runtimeDirection.Hi.data[firstMatrixMismatch] : 0);
        check(residualMismatch == 0, tag + ": runtime residual is bitwise accepted CPU");
        check(matrixMismatch == 0, tag + ": runtime matrix is bitwise accepted CPU");
    }

    std::vector<dstype> v(op.N), w, ve, weAE, wAE;
    for (Int i = 0; i < op.N; i++)
        v[i] = Model::ncq == 0 ? hv(i, 61 + sc.model) : hv(i, 61 + sc.model) - 0.5;
    matvec(op, w, op.AE, v, &nm, mixedFamilies ? &sidecar : nullptr);
    gather_trace(op, ve, v);
    element_products(op, weAE, op.AE, ve);
    scatter_faces(op, wAE, weAE);
    const dstype couplingScale = global_max_abs_diff(w, wAE, sc.comm);
    check(couplingScale > 1e-3 * global_max_abs(w, sc.comm), tag + ": coupling contribution is significant");

    // Coupling disabled with type-1 setup data present (and Hi formed): the
    // product is the element part only, through the same production call.
    for (int off = 0; off < 2; off++) {
        Int& flag = off == 0 ? op.common.couplingparams.coupledinterface : op.common.couplingparams.coupledcondition;
        flag = 0;
        check(!exasim::nonmatching::nonmatching_coupling_active(op.common), tag + ": coupling inactive");
        std::vector<dstype> wOff;
        matvec(op, wOff, op.AE, v, &nm, mixedFamilies ? &sidecar : nullptr);
        const dstype errOff = global_max_abs_diff(wOff, wAE, sc.comm) / global_max_abs(wAE, sc.comm);
        maxRoundoff = std::max(maxRoundoff, errOff);
        check(errOff < 1e-14, tag + (off == 0 ? ": coupledinterface = 0 applies no type-1 data"
                                              : ": coupledcondition = 0 applies no type-1 data"));
        flag = 1;
    }
    check(exasim::nonmatching::nonmatching_coupling_active(op.common), tag + ": coupling active again");

    if (productReference) {
        std::vector<dstype> weRef(weAE), wRef;
        long multiK = 0, multiR = 0;
        reference_coupling(op, nm, ve, weRef, worldRank, sc.comm, multiK, multiR);
        scatter_faces(op, wRef, weRef);
        const dstype err = global_max_abs_diff(w, wRef, sc.comm) / couplingScale;
        maxRoundoff = std::max(maxRoundoff, err);
        check(err < 1e-13, tag + ": hdgMatVec equals the independent pair-product assembly");
        long counts[2] = {multiK, multiR}, totals[2] = {0, 0};
        MPI_Allreduce(counts, totals, 2, MPI_LONG, MPI_SUM, sc.comm);
        check(totals[0] > 0, tag + ": slots with |K_l| > 1");
        if (sc.k > 1) check(totals[1] > 0, tag + ": owner faces with |R_l| > 1");
        if (worldRank == 0)
            std::printf("%s: product reference error %.3e, %ld slots |K_l|>1, %ld faces |R_l|>1\n",
                        tag.c_str(), err, totals[0], totals[1]);
    }

    // Central differences along the local-solver tangent.
    std::vector<dstype> dudg;
    state_tangent(op, ve, dudg);
    const dstype eps[3] = {1e-2, 1e-3, 1e-4};
    dstype errs[3];
    for (int s = 0; s < 3; s++) {
        std::vector<dstype> Rp, Rm;
        for (std::size_t i = 0; i < op.uh.size(); i++) op.uh[i] = op.uh0[i] + eps[s] * v[i];
        for (std::size_t i = 0; i < op.udg.size(); i++) op.udg[i] = op.udg0[i] + eps[s] * dudg[i];
        coupled_residual<Model>(op, nm, Rp, mixedFamilies ? &sidecar : nullptr);
        for (std::size_t i = 0; i < op.uh.size(); i++) op.uh[i] = op.uh0[i] - eps[s] * v[i];
        for (std::size_t i = 0; i < op.udg.size(); i++) op.udg[i] = op.udg0[i] - eps[s] * dudg[i];
        coupled_residual<Model>(op, nm, Rm, mixedFamilies ? &sidecar : nullptr);
        std::vector<dstype> fd(op.N);
        for (Int i = 0; i < op.N; i++) fd[i] = (Rp[i] - Rm[i]) / (2 * eps[s]);
        errs[s] = global_max_abs_diff(fd, w, sc.comm) / couplingScale;
    }
    op.uh = op.uh0;
    op.udg = op.udg0;
    maxFdError = std::max(maxFdError, errs[2]);
    check(errs[2] < 1e-6, tag + ": finite differences match hdgMatVec");
    check(errs[0] < 1e-9 || (errs[1] < errs[0] / 30 && errs[2] < errs[1] / 30),
          tag + ": finite-difference error decreases as eps^2");
    if (worldRank == 0)
        std::printf("%s: FD error %.3e %.3e %.3e (eps 1e-2 1e-3 1e-4)\n", tag.c_str(), errs[0], errs[1], errs[2]);
    exasim::nonmatching::free_nonmatching_data(nm);
}

// Element halo of the same model (nnbsd > 0, elemsend/elemrecv) together with
// the coupling-slot exchange in one hdgMatVec, with the exact request count.
// Reference: every element contribution (owned and ghost, same AE by global
// element) and the independent pair-product coupling, compared on the faces
// that touch an owned element.
template <class Model>
void halo_case(const std::string& name, const ModelMesh& m1, const ModelMesh& m2, MasterData& mst, int n)
{
    if (n < 4) return; // needs at least two ranks per model
    SubComm sc(n);
    if (sc.comm == MPI_COMM_NULL) return;
    MPI_Comm local = MPI_COMM_NULL;
    MPI_Comm_split(sc.comm, sc.model, sc.part, &local);
    EXASIM_COMM_LOCAL = local;
    const std::string tag = name + " (" + std::to_string(sc.k) + " per model)";
    const ModelMesh& mm = sc.model == 0 ? m1 : m2;
    LocalMesh lm;
    build_halo_mesh(lm, mm, mst, sc.part, sc.k, tag);
    check(!lm.nbsd.empty() && lm.ne > lm.ne1, tag + ": rank has halo neighbors and ghosts");

    LocalMesh owned = owned_part(lm, mst);
    RankStructs rs;
    fill_rank_structs(rs, owned, mst, Model::ncu, Model::nfint, 1, Model::ncq);
    nonmatchingdata nm;
    const bool ran = exasim::nonmatching::setup_nonmatching_interface(rs.app, rs.master, rs.mesh, rs.sol,
            static_cast<Int>(n), static_cast<Int>(worldRank), static_cast<Int>(sc.model * sc.k), nm);
    check(ran && nm.interfacetype == 1, tag + ": setup ran");

    RankOperator op;
    init_operator<Model>(op, lm, mst, sc.model);
    use_nonmatching(op);
    const Int m = op.m, nfe = op.nfe, npf = op.npf;

    // Same element matrix on the owner and on every rank holding the element as a ghost.
    for (Int e = 0; e < op.ne; e++)
        for (Int j = 0; j < m; j++)
            for (Int i = 0; i < m; i++) {
                const std::size_t k = i + static_cast<std::size_t>(m) * (j + static_cast<std::size_t>(m) * e);
                const long g = i + static_cast<long>(m) * (j + static_cast<long>(m) * lm.elempart[e]);
                op.AE[k] = (i == j) ? 4.0 + hv(g, 147 + sc.model) : 0.1 * (hv(g, 153 + sc.model) - 0.5);
            }

    // Halo lists and buffers.
    commonstruct& c = op.common;
    c.nnbsd = static_cast<Int>(lm.nbsd.size());
    c.nbsd = lm.nbsd.data();
    c.elemsendpts = lm.elemsendpts.data();
    c.elemrecvpts = lm.elemrecvpts.data();
    c.nelemsend = static_cast<Int>(lm.elemsend.size());
    c.nelemrecv = static_cast<Int>(lm.elemrecv.size());
    op.mesh.elemsend = lm.elemsend.data();
    op.mesh.elemrecv = lm.elemrecv.data();
    op.buffsend.assign(static_cast<std::size_t>(m) * std::max<Int>(c.nelemsend, 1), 0);
    op.buffrecv.assign(static_cast<std::size_t>(m) * std::max<Int>(c.nelemrecv, 1), 0);
    op.tmp.buffsend = op.buffsend.data();
    op.tmp.buffrecv = op.buffrecv.data();

    // Exact request capacity (halo + coupling slots), plus one sentinel.
    Int nreq = 0;
    for (Int q = 0; q < c.nnbsd; q++) nreq += (lm.elemsendpts[q] > 0) + (lm.elemrecvpts[q] > 0);
    for (int direction = 0; direction < 2; direction++) {
        const nonmatchinginterface& d = nm.dir[direction];
        if (d.isdonor && d.nslots > 0)
            for (Int q = 0; q < d.nsendnb; q++) nreq += d.sendcount[q] > 0;
        if (d.isowner)
            for (Int q = 0; q < d.nrecvnb; q++) nreq += d.recvcount[q] > 0;
    }
    op.requests.assign(nreq + 1, MPI_REQUEST_NULL);
    op.statuses.resize(nreq + 1);
    c.requests = op.requests.data();
    c.statuses = op.statuses.data();

    // A donor element on the partition boundary (owned, next to a ghost row).
    long boundaryDonors = 0, total = 0;
    for (int direction = 0; direction < 2; direction++) {
        const nonmatchinginterface& d = nm.dir[direction];
        if (!d.isdonor) continue;
        for (Int q = 0; q < d.npairs; q++) boundaryDonors += d.pairelem[q] >= op.ne0 && d.pairelem[q] < lm.ne1;
    }
    MPI_Allreduce(&boundaryDonors, &total, 1, MPI_LONG, MPI_SUM, sc.comm);
    check(total > 0, tag + ": donor elements adjacent to a partition boundary");

    exasim::nonmatching::compute_nonmatching_coupling<Model>(nm, op.sol, op.res, op.app, op.master, op.mesh,
            op.tmp, op.common, 0, true, 0);

    // Trace values by physical node position, so shared faces agree across ranks.
    std::vector<dstype> v(op.N);
    for (Int f = 0; f < op.nf; f++) {
        const Int e1 = op.f2e[4 * f], l1 = op.f2e[4 * f + 1];
        for (Int j = 0; j < npf; j++) {
            const Int node = op.perm[j + npf * l1];
            const long kx = std::llround(lm.xdg[node + op.npe * 0 + op.npe * op.nd * e1] * 1e7);
            const long ky = std::llround(lm.xdg[node + op.npe * 1 + op.npe * op.nd * e1] * 1e7);
            for (Int cc = 0; cc < op.ncu; cc++)
                v[cc + op.ncu * (j + npf * f)] = hv(7919 * kx + 104729 * ky + 13 * cc, 173 + sc.model) - 0.5;
        }
    }
    std::vector<dstype> w;
    matvec(op, w, op.AE, v, &nm);
    check(op.requests[nreq] == MPI_REQUEST_NULL, tag + ": requests stay within the exact capacity");

    std::vector<dstype> ve, weRef, wRef, wAE;
    gather_trace(op, ve, v); // all local elements, ghosts included
    element_products(op, weRef, op.AE, ve);
    scatter_faces(op, wAE, weRef);
    long multiK = 0, multiR = 0;
    reference_coupling(op, nm, ve, weRef, worldRank, sc.comm, multiK, multiR);
    scatter_faces(op, wRef, weRef);
    std::vector<dstype> a, b, b0;
    for (Int f = 0; f < op.nf; f++) {
        const bool owned1 = op.f2e[4 * f] < lm.ne1;
        const Int e2 = op.f2e[4 * f + 2];
        if (!(owned1 || (e2 >= 0 && e2 < lm.ne1))) continue;
        for (Int i = 0; i < op.ncu * npf; i++) {
            a.push_back(w[i + op.ncu * npf * f]);
            b.push_back(wRef[i + op.ncu * npf * f]);
            b0.push_back(wAE[i + op.ncu * npf * f]);
        }
    }
    const dstype coupling = global_max_abs_diff(b, b0, sc.comm);
    const dstype err = global_max_abs_diff(a, b, sc.comm) / global_max_abs(b, sc.comm);
    maxRoundoff = std::max(maxRoundoff, err);
    check(coupling > 1e-3 * global_max_abs(b, sc.comm), tag + ": coupling contribution is significant");
    check(err < 1e-13, tag + ": halo + coupling hdgMatVec equals the reference");
    (void)nfe;
    long counts[2] = {static_cast<long>(lm.nbsd.size()), multiR}, totals[2] = {0, 0};
    MPI_Allreduce(counts, totals, 2, MPI_LONG, MPI_SUM, sc.comm);
    if (worldRank == 0)
        std::printf("%s: halo + coupling error %.3e, %ld halo neighbor links, %ld boundary donor pairs, "
                    "%ld faces |R_l|>1, %d requests on rank 0\n",
                    tag.c_str(), err, totals[0], total, totals[1], static_cast<int>(nreq));
    exasim::nonmatching::free_nonmatching_data(nm);
    MPI_Comm_free(&local);
}

// Conformal res.Hi of the interface block from the production element-face
// and Schur kernels, with the local solver of op (D = I, B = 0, DinvF = op.F,
// D^{-1}Ru = op.Ru supplied in their native layouts).
template <class Model>
void conformal_interface_matrices(RankOperator& op, const LocalMesh& lm, const MasterData& mst)
{
    const Int npe = op.npe, npf = op.npf, nfe = op.nfe, ngf = op.ngf, nd = op.nd, ncu = op.ncu;
    const Int ncq = op.ncq, ncu12 = op.ncu12, n = op.n, m = op.m, m12 = op.m12;
    const Int ne = op.ne, e1 = op.ne0, neb = op.neI, ncx = nd;
    check(op.ne - op.ne0 == op.neI, "conformal: interface block is [ne0, ne)");

    // Face geometry of the block: xg, nlg, jac (block-local layout).
    op.elemfaceg.assign(static_cast<std::size_t>(ngf) * nfe * ne * (ncx + nd + 1), 0);
    const Int nga = ngf * nfe * neb;
    dstype* xg = &op.elemfaceg[static_cast<std::size_t>(ngf) * nfe * e1 * (ncx + nd + 1)];
    dstype* nlg = xg + static_cast<std::size_t>(nga) * ncx;
    dstype* jac = xg + static_cast<std::size_t>(nga) * (ncx + nd);
    for (Int eb = 0; eb < neb; eb++)
        for (Int lf = 0; lf < nfe; lf++)
            for (Int g = 0; g < ngf; g++) {
                dstype x[2] = {0, 0}, t[2] = {0, 0};
                for (Int j = 0; j < npf; j++) {
                    const Int node = mst.perm[j + npf * lf];
                    for (Int d = 0; d < nd; d++) {
                        const dstype X = lm.xdg[node + npe * d + npe * nd * (e1 + eb)];
                        x[d] += mst.shapfgt[g + ngf * j] * X;
                        t[d] += mst.shapfgt[ngf * npf + g + ngf * j] * X;
                    }
                }
                const dstype J = std::hypot(t[0], t[1]);
                const Int gf = g + ngf * (lf + nfe * eb);
                for (Int d = 0; d < nd; d++) xg[gf + nga * d] = x[d];
                nlg[gf + nga * 0] = t[1] / J;
                nlg[gf + nga * 1] = -t[0] / J;
                jac[gf] = J;
            }
    op.sol.elemfaceg = op.elemfaceg.data();

    std::vector<dstype> Ru(static_cast<std::size_t>(n) * ne, 0), Rh(static_cast<std::size_t>(m) * ne, 0);
    std::vector<dstype> D(static_cast<std::size_t>(n) * n * neb, 0), B(static_cast<std::size_t>(npe) * npe * ncu * ncq * neb, 0);
    std::vector<dstype> F(static_cast<std::size_t>(n) * m * ne, 0), G(static_cast<std::size_t>(npf) * nfe * npe * neb * ncu * ncq, 0);
    std::vector<dstype> K(static_cast<std::size_t>(npf) * nfe * npe * neb * ncu * ncu, 0);
    std::vector<dstype> H(static_cast<std::size_t>(m) * m * ne, 0);
    op.confRi.assign(static_cast<std::size_t>(m12) * neb, 0);
    op.confHi.assign(static_cast<std::size_t>(m12) * m * neb, 0);
    op.confKi.assign(static_cast<std::size_t>(m12) * n * neb, 0);
    op.confGi.assign(static_cast<std::size_t>(npf) * npe * neb * ncu12 * ncq, 0);
    resstruct conf;
    conf.Ru = Ru.data(); conf.Rh = Rh.data(); conf.D = D.data(); conf.B = B.data();
    conf.F = F.data(); conf.G = G.data(); conf.K = K.data(); conf.H = H.data();
    conf.C = op.C.data(); conf.E = op.E.data();
    conf.Ri = op.confRi.data(); conf.Hi = op.confHi.data(); conf.Ki = op.confKi.data(); conf.Gi = op.confGi.data();
    conf.ipiv = op.ipiv.data();

    uEquationElemFaceBlock<Model>(op.sol, conf, op.app, op.master, op.mesh, op.tmp, op.common, 0, 1, 0);
    // Native pre-Schur layouts of the block: D npe x npe x neb x ncu x ncu,
    // F npe x (npf*nfe) x neb x ncu x ncu, Ru npe x neb x ncu.
    std::fill(D.begin(), D.end(), 0);
    for (Int eb = 0; eb < neb; eb++)
        for (Int c = 0; c < ncu; c++)
            for (Int i = 0; i < npe; i++)
                D[i + npe * i + static_cast<std::size_t>(npe) * npe * (eb + neb * (c + ncu * c))] = 1;
    std::fill(B.begin(), B.end(), 0);
    const Int ndf = npf * nfe;
    for (Int eb = 0; eb < neb; eb++) {
        const Int e = e1 + eb;
        for (Int tn = 0; tn < ndf; tn++)
            for (Int tc = 0; tc < ncu; tc++)
                for (Int ec = 0; ec < ncu; ec++)
                    for (Int en = 0; en < npe; en++) {
                        const std::size_t transformed = en + npe * ec + static_cast<std::size_t>(n) * (tc + ncu * tn) +
                                                        static_cast<std::size_t>(n) * m * e;
                        const std::size_t native = static_cast<std::size_t>(n) * m * e1 + en + npe * tn +
                            static_cast<std::size_t>(npe) * ndf * (eb + neb * (ec + ncu * tc));
                        F[native] = -op.F[transformed];
                    }
        for (Int c = 0; c < ncu; c++)
            for (Int j = 0; j < npe; j++)
                Ru[static_cast<std::size_t>(n) * e1 + j + npe * (eb + neb * c)] = op.Ru[j + npe * c + n * e];
    }
    uEquationSchurBlock<Model>(op.sol, conf, op.app, op.master, op.mesh, op.tmp, op.common, 0, 1, 0);

    // The Schur kernel must reproduce the supplied local solver.
    dstype dev = 0;
    for (Int e = e1; e < ne; e++)
        for (Int i = 0; i < n * m; i++)
            dev = std::max(dev, std::abs(F[static_cast<std::size_t>(n) * m * e + i] -
                                         op.F[static_cast<std::size_t>(n) * m * e + i]));
    check(dev < 1e-14, "conformal: Schur block reproduces DinvF");
}

// Conformal equivalence and GMRES iteration counts on a matching mesh.
template <class Model>
void matching_case(const std::string& name, const ModelMesh& m1, const ModelMesh& m2, MasterData& mst, int n)
{
    SubComm sc(n);
    if (sc.comm == MPI_COMM_NULL) return;
    const std::string tag = name + " (" + std::to_string(sc.k) + " per model)";
    LocalMesh lm;
    build_local_mesh(lm, sc.model == 0 ? m1 : m2, mst, sc.part, sc.k, tag);

    RankStructs rc, rn;
    fill_rank_structs(rc, lm, mst, Model::ncu, Model::nfint, 0, Model::ncq);
    std::vector<Int> lsize(1, 0);
    rc.mesh.lsize = lsize.data();
    exasim::interfacepartition::build_runtime_interface_partition(rc.app, rc.master, rc.mesh, rc.sol,
            static_cast<Int>(n), static_cast<Int>(worldRank), static_cast<Int>(sc.model * sc.k));
    check(rc.mesh.sznbintf > 0 && rc.mesh.szfaceperm == mst.npf, tag + ": conformal partition built");
    bool reversed = rc.mesh.szfaceperm == mst.npf;
    for (int i = 0; reversed && i < mst.npf; i++) reversed = rc.mesh.faceperm[i] == mst.npf - 1 - i;
    check(reversed, tag + ": faceperm is the nontrivial reversal");
    long secondPerm = 0;
    for (int f = 0; f < lm.nf; f++)
        if (lm.f2e[4 * f + 2] >= 0) {
            const int e2 = lm.f2e[4 * f + 2], l2 = lm.f2e[4 * f + 3];
            if (lm.elemcon[0 + mst.npf * l2 + mst.npf * mst.nfe * e2] != mst.npf * f) secondPerm++;
        }
    check(secondPerm > 0, tag + ": elemcon permutes the second element of interior faces");

    fill_rank_structs(rn, lm, mst, Model::ncu, Model::nfint, 1, Model::ncq);
    nonmatchingdata nm;
    const bool ran = exasim::nonmatching::setup_nonmatching_interface(rn.app, rn.master, rn.mesh, rn.sol,
            static_cast<Int>(n), static_cast<Int>(worldRank), static_cast<Int>(sc.model * sc.k), nm);
    check(ran && nm.interfacetype == 1, tag + ": setup ran");

    RankOperator op;
    init_operator<Model>(op, lm, mst, sc.model);
    conformal_interface_matrices<Model>(op, lm, mst);
    use_nonmatching(op);
    exasim::nonmatching::compute_nonmatching_coupling<Model>(nm, op.sol, op.res, op.app, op.master, op.mesh,
            op.tmp, op.common, 0, true, 0);

    std::vector<dstype> v(op.N), wc, wn, ve, weAE, wAE;
    for (Int i = 0; i < op.N; i++) v[i] = hv(i, 67 + sc.model) - 0.5;
    use_conformal(op, rc.mesh);
    check(!exasim::nonmatching::validate_coupling_selection(op.common, static_cast<nonmatchingdata*>(nullptr),
                                                            Int(0), "test"), tag + ": conformal selection");
    matvec(op, wc, op.AE, v, nullptr);
    use_nonmatching(op);
    matvec(op, wn, op.AE, v, &nm);
    gather_trace(op, ve, v);
    element_products(op, weAE, op.AE, ve);
    scatter_faces(op, wAE, weAE);
    const dstype couplingScale = global_max_abs_diff(wc, wAE, sc.comm);
    check(couplingScale > 1e-3 * global_max_abs(wc, sc.comm), tag + ": conformal coupling is significant");
    const dstype err = global_max_abs_diff(wc, wn, sc.comm) / couplingScale;
    maxRoundoff = std::max(maxRoundoff, err);
    check(err < 1e-12, tag + ": conformal and non-matching hdgMatVec agree");

    // GMRES(30), right preconditioning with the elemental-block preconditioner
    // applied through hdgMatVec (decision D1), relative tolerance 1e-10.
    std::vector<dstype> b(op.N);
    for (Int i = 0; i < op.N; i++) b[i] = hv(i, 71 + sc.model) - 0.5;
    auto gmres = [&](bool conformal, std::vector<dstype>& x) {
        auto A = [&](std::vector<dstype>& out, const std::vector<dstype>& in, std::vector<dstype>& mat) {
            if (conformal) { use_conformal(op, rc.mesh); matvec(op, out, mat, in, nullptr); }
            else { use_nonmatching(op); matvec(op, out, mat, in, &nm); }
        };
        const int restart = 30, maxit = 600;
        const dstype tol = 1e-10, bnorm = std::sqrt(global_dot(b, b, sc.comm));
        x.assign(op.N, 0);
        int its = 0;
        dstype rel = 1;
        while (its < maxit) {
            std::vector<dstype> r(b), Ax;
            A(Ax, x, op.AE);
            for (Int i = 0; i < op.N; i++) r[i] -= Ax[i];
            const dstype beta = std::sqrt(global_dot(r, r, sc.comm));
            rel = beta / bnorm;
            if (rel < tol) break;
            std::vector<std::vector<dstype>> V(1, r), Z;
            for (auto& vi : V[0]) vi /= beta;
            std::vector<dstype> Hm(static_cast<std::size_t>(restart + 1) * restart, 0), cs(restart), sn(restart),
                g(restart + 1, 0);
            g[0] = beta;
            int j = 0;
            for (; j < restart && its < maxit; j++) {
                std::vector<dstype> z, wv;
                A(z, V[j], op.PE);
                A(wv, z, op.AE);
                Z.push_back(z);
                for (int i = 0; i <= j; i++) {
                    const dstype h = global_dot(wv, V[i], sc.comm);
                    Hm[i + (restart + 1) * j] = h;
                    for (Int q = 0; q < op.N; q++) wv[q] -= h * V[i][q];
                }
                const dstype h = std::sqrt(global_dot(wv, wv, sc.comm));
                Hm[j + 1 + (restart + 1) * j] = h;
                for (auto& q : wv) q /= h;
                V.push_back(wv);
                for (int i = 0; i < j; i++) {
                    const dstype a = Hm[i + (restart + 1) * j], c = Hm[i + 1 + (restart + 1) * j];
                    Hm[i + (restart + 1) * j] = cs[i] * a + sn[i] * c;
                    Hm[i + 1 + (restart + 1) * j] = -sn[i] * a + cs[i] * c;
                }
                const dstype a = Hm[j + (restart + 1) * j], c = Hm[j + 1 + (restart + 1) * j];
                const dstype rr = std::hypot(a, c);
                cs[j] = a / rr; sn[j] = c / rr;
                Hm[j + (restart + 1) * j] = rr;
                Hm[j + 1 + (restart + 1) * j] = 0;
                g[j + 1] = -sn[j] * g[j];
                g[j] = cs[j] * g[j];
                its++;
                if (std::abs(g[j + 1]) / bnorm < tol) { j++; break; }
            }
            std::vector<dstype> y(j, 0);
            for (int i = j - 1; i >= 0; i--) {
                dstype s = g[i];
                for (int l = i + 1; l < j; l++) s -= Hm[i + (restart + 1) * l] * y[l];
                y[i] = s / Hm[i + (restart + 1) * i];
            }
            for (int i = 0; i < j; i++)
                for (Int q = 0; q < op.N; q++) x[q] += y[i] * Z[i][q];
        }
        std::vector<dstype> r(b), Ax;
        A(Ax, x, op.AE);
        for (Int i = 0; i < op.N; i++) r[i] -= Ax[i];
        rel = std::sqrt(global_dot(r, r, sc.comm)) / bnorm;
        check(rel < 1e-9, tag + ": GMRES converged");
        return its;
    };
    std::vector<dstype> xc, xn;
    const int itsConformal = gmres(true, xc);
    const int itsNonmatching = gmres(false, xn);
    check(itsConformal == itsNonmatching, tag + ": identical GMRES iteration counts");
    check(itsConformal > 5, tag + ": GMRES needs several iterations");
    const dstype dx = global_max_abs_diff(xc, xn, sc.comm) / std::max<dstype>(global_max_abs(xc, sc.comm), 1e-300);
    check(dx < 1e-9, tag + ": GMRES solutions agree");
    if (worldRank == 0)
        std::printf("%s: conformal/non-matching hdgMatVec %.3e, GMRES iterations %d / %d, solution diff %.3e\n",
                    tag.c_str(), err, itsConformal, itsNonmatching, dx);

    exasim::nonmatching::free_nonmatching_data(nm);
    CPUFREE(rc.mesh.nsize);
    CPUFREE(rc.mesh.faceperm);
    CPUFREE(rc.mesh.nbintf);
    CPUFREE(rc.mesh.facesend);
    CPUFREE(rc.mesh.facesendpts);
    CPUFREE(rc.mesh.facerecv);
    CPUFREE(rc.mesh.facerecvpts);
}


// Consistent selections pass and report whether type-1 coupling is active.
void selection_case()
{
    using exasim::nonmatching::validate_coupling_selection;
    commonstruct common;
    nonmatchingdata nm;
    nonmatchingdata* none = nullptr;
    common.couplingparams.coupledcondition = 1;
    common.couplingparams.coupledinterface = 1;
    common.couplingparams.interfacetype = 0;
    check(!validate_coupling_selection(common, none, Int(0), "test"), "selection: conformal, no data");
    check(!validate_coupling_selection(common, &nm, Int(2), "test"), "selection: conformal on a GPU backend");
    nm.interfacetype = 1;
    common.couplingparams.interfacetype = 1;
    check(validate_coupling_selection(common, &nm, Int(0), "test"), "selection: type 1 active");
    check(validate_coupling_selection(common, &nm, Int(1), "test"), "selection: type 1 active (OpenMP)");
    common.couplingparams.coupledinterface = 0;
    check(!validate_coupling_selection(common, none, Int(2), "test"), "selection: type 1 disabled, no data");
    check(!validate_coupling_selection(common, &nm, Int(2), "test"), "selection: type 1 disabled with data");
}

// Negative selections: each runs in a child process forked before MPI_Init
// (error() then exits instead of calling MPI_Abort). The child must exit with
// a failure status and print the expected message.
struct NegativeCase {
    const char* name;
    const char* expect;
    std::function<void()> run;
};

void run_negative_cases()
{
    auto active = []() {
        commonstruct common;
        common.couplingparams.coupledcondition = 1;
        common.couplingparams.coupledinterface = 1;
        common.couplingparams.interfacetype = 1;
        return common;
    };
    // A rank with no donor or owner role (empty type-1 data).
    static nonmatchingdata typeOne, notRun;
    static exasim::nonmatching::NonmatchingRuntime<dstype, Int> runtimeTypeOne, runtimeNotRun;
    typeOne.interfacetype = 1;
    runtimeTypeOne.interfacetype = 1;
    std::vector<NegativeCase> cases = {
        {"active type 1 without data", "requires the non-matching data",
         [&]() { commonstruct c = active(); exasim::nonmatching::validate_coupling_selection(c, static_cast<nonmatchingdata*>(nullptr), Int(0), "test"); }},
        {"active type 1, setup did not run", "non-matching setup did not run",
         [&]() { commonstruct c = active(); exasim::nonmatching::validate_coupling_selection(c, &notRun, Int(0), "test"); }},
        {"type-1 data with conformal selection", "setup data has interfacetype 1",
         [&]() { commonstruct c = active(); c.couplingparams.interfacetype = 0; exasim::nonmatching::validate_coupling_selection(c, &typeOne, Int(0), "test"); }},
        {"active type 1 on CUDA, rank without donor/owner role", "runs on the CPU only",
         [&]() { commonstruct c = active(); exasim::nonmatching::validate_coupling_selection(c, &typeOne, Int(2), "test"); }},
        {"active type 1 on HIP", "runs on the CPU only",
         [&]() { commonstruct c = active(); exasim::nonmatching::validate_coupling_selection(c, &typeOne, Int(3), "test"); }},
        {"active type 1 without MPI", "requires MPI",
         [&]() { commonstruct c = active(); exasim::nonmatching::check_coupling_selection(c, &typeOne, Int(0), "test", false); }},
        {"hdgMatVec, active type 1 without data", "requires a run-time cache",
         [&]() {
             commonstruct c = active(); resstruct res; appstruct app; meshstruct mesh; tempstruct tmp;
             hdgMatVec<dstype, Int>(nullptr, nullptr, nullptr, nullptr, nullptr, res, app, mesh, c, tmp, 0, 0,
                                    static_cast<exasim::nonmatching::NonmatchingRuntime<dstype, Int>*>(nullptr));
         }},
        {"hdgMatVec, active type 1 on CUDA", "awaits device validation",
         [&]() {
             commonstruct c = active(); resstruct res; appstruct app; meshstruct mesh; tempstruct tmp;
             hdgMatVec<dstype, Int>(nullptr, nullptr, nullptr, nullptr, nullptr, res, app, mesh, c, tmp, 0, 2,
                                    &runtimeTypeOne);
         }},
    };
    for (const NegativeCase& nc : cases) {
        int fds[2];
        if (pipe(fds) != 0) { check(false, std::string("negative: pipe for ") + nc.name); continue; }
        std::fflush(stdout);
        std::fflush(stderr);
        const pid_t pid = fork();
        if (pid == 0) {
            close(fds[0]);
            dup2(fds[1], STDERR_FILENO);
            dup2(fds[1], STDOUT_FILENO);
            nc.run();
            std::printf("NO ERROR RAISED\n");
            std::fflush(stdout);
            _exit(0);
        }
        close(fds[1]);
        std::string output;
        char buffer[512];
        ssize_t got;
        while ((got = read(fds[0], buffer, sizeof(buffer))) > 0) output.append(buffer, static_cast<std::size_t>(got));
        close(fds[0]);
        int status = 0;
        waitpid(pid, &status, 0);
        const bool failed = WIFEXITED(status) && WEXITSTATUS(status) != 0;
        const bool message = output.find(nc.expect) != std::string::npos &&
                             output.find("ERROR") != std::string::npos;
        check(failed && message, std::string("negative selection fails clearly: ") + nc.name);
        std::printf("negative selection \"%s\": exit status %d, %s\n", nc.name,
                    WIFEXITED(status) ? WEXITSTATUS(status) : -1,
                    message ? "expected message" : "MESSAGE MISSING");
    }
}

// ---------------------------------------------------------------------------
// [nonmatching] Phase 9 B3: changed-count rebuild followed by the production
// residual, Jacobian, compact exchange, owner assembly and hdgMatVec, without
// reconstructing the runtime in between. The rebuild goes through the Phase 8
// transaction (rebuild_nonmatching_interface with a unique_ptr runtime), which
// also replaces common.requests/statuses, so those start as nullptr here.

// One complete type-1 evaluation with the persistent runtime; every product is
// appended to out in a fixed order (Ri per direction, assembled residual, Ri and
// Hi per direction after the Jacobian pass, hdgMatVec product).
template <class Model>
void runtime_cycle(RankOperator& op, exasim::nonmatching::NonmatchingRuntime<dstype, Int>& rt,
                   const nonmatchingdata& nm, int model, std::vector<std::vector<dstype>>& out)
{
    out.clear();
    exasim::nonmatching::compute_nonmatching_coupling<Model>(rt, op.sol, op.res, op.app, op.master, op.mesh,
            op.tmp, op.common, 0, false, 0);
    for (int d = 0; d < 2; d++)
        out.emplace_back(rt.dir[d].Ri.data, rt.dir[d].Ri.data + nm.dir[d].szRi);
    Int count = 0;
    exasim::nonmatching::begin_coupling_slot_exchange(rt, op.common, count);
    MPI_Waitall(count, op.common.requests, op.common.statuses);
    std::vector<dstype> wc(static_cast<std::size_t>(op.m) * op.ne, 0);
    exasim::nonmatching::assemble_coupling_slot_owner(wc.data(), rt, op.mesh, op.app, op.common);
    out.push_back(wc);
    exasim::nonmatching::compute_nonmatching_coupling<Model>(rt, op.sol, op.res, op.app, op.master, op.mesh,
            op.tmp, op.common, 0, true, 0);
    for (int d = 0; d < 2; d++) {
        out.emplace_back(rt.dir[d].Ri.data, rt.dir[d].Ri.data + nm.dir[d].szRi);
        out.emplace_back(rt.dir[d].Hi.data, rt.dir[d].Hi.data + nm.dir[d].szHi);
    }
    std::vector<dstype> v(op.N), w(op.N, 0);
    for (Int i = 0; i < op.N; i++) v[i] = hv(i, 71 + model) - 0.5;
    hdgMatVec<dstype, Int>(w.data(), op.AE.data(), v.data(), op.ve.data(), op.we.data(), op.res, op.app,
                           op.mesh, op.common, op.tmp, 0, 0, &rt);
    out.push_back(w);
}

template <class Model>
void rebuild_case(const std::string& name, const ModelMesh& m1, const ModelMesh& m2,
                  MasterData& mst, int n, bool mixedFamilies = false,
                  bool model0Triangle = false)
{
    SubComm sc(n);
    if (sc.comm == MPI_COMM_NULL) return;
    const std::string tag = name + " (" + std::to_string(sc.k) + " per model)";
    using Runtime = exasim::nonmatching::NonmatchingRuntime<dstype, Int>;
    MasterData triangle, quad;
    if (mixedFamilies) {
        triangle = make_master(0, mst.porder);
        quad = make_master(1, mst.porder);
    }
    MasterData& localMaster = mixedFamilies ?
            (((sc.model == 0) == model0Triangle) ? triangle : quad) : mst;
    LocalMesh lm;
    build_local_mesh(lm, sc.model == 0 ? m1 : m2, localMaster, sc.part, sc.k, tag);
    // Mesh motion of model 2: y -> y + 0.15 sin(pi y) (monotone). The outer
    // boundaries and the interface line x = 0.5 stay in place; the model-2
    // interface faces slide along it, so the per-rank donor points, the pairs and
    // (with more than one rank per model) the slots change.
    LocalMesh moved = lm;
    const dstype pi = 3.14159265358979323846;
    if (sc.model == 1)
        for (int e = 0; e < moved.ne; e++)
            for (int j = 0; j < localMaster.npe; j++) {
                dstype& y = moved.xdg[j + localMaster.npe * (1 + localMaster.nd * e)];
                y += 0.15 * std::sin(pi * y);
            }

    auto prepare = [&](RankStructs& rs, RankOperator& op, const LocalMesh& l) {
        fill_rank_structs(rs, l, localMaster, Model::ncu, Model::nfint, 1, Model::ncq);
        init_operator<Model>(op, l, localMaster, sc.model);
        use_nonmatching(op);
        rs.app.interfacefluxmap = op.map.data();
        op.common.requests = nullptr;   // owned by the rebuild transaction
        op.common.statuses = nullptr;
    };
    auto rebuild = [&](RankStructs& rs, RankOperator& op, nonmatchingdata& nm,
                       std::unique_ptr<Runtime>& rt,
                       std::unique_ptr<CompositeOwnerQuadrature<dstype, Int>>& sidecar) {
        exasim::nonmatching::rebuild_nonmatching_interface(rs.app, rs.master, rs.mesh, rs.sol, op.common,
                static_cast<Int>(n), static_cast<Int>(worldRank), static_cast<Int>(sc.model * sc.k), nm, rt,
                Int(0), static_cast<DonorSearchResult<dstype, Int>*>(nullptr),
                DonorSearchParams(), true, mixedFamilies ? &sidecar : nullptr);
        check(rt && rt->interfacetype == 1, tag + ": rebuild publishes a type-1 runtime");
    };
    // Totals (points, slots, pairs, send and receive neighbors) in c, and the
    // per-rank (points, slots, pairs) in perRank.
    auto counts = [&](const nonmatchingdata& nm, long long c[5], std::vector<long long>& perRank) {
        long long local[5] = {0, 0, 0, 0, 0};
        for (int d = 0; d < 2; d++) {
            local[0] += nm.dir[d].npts; local[1] += nm.dir[d].nslots; local[2] += nm.dir[d].npairs;
            local[3] += nm.dir[d].nsendnb; local[4] += nm.dir[d].nrecvnb;
        }
        MPI_Allreduce(local, c, 5, MPI_LONG_LONG, MPI_SUM, sc.comm);
        perRank.assign(3 * static_cast<std::size_t>(n), 0);
        MPI_Allgather(local, 3, MPI_LONG_LONG, perRank.data(), 3, MPI_LONG_LONG, sc.comm);
    };

    RankStructs rs;
    RankOperator op;
    prepare(rs, op, lm);
    nonmatchingdata nm;
    std::unique_ptr<Runtime> rt;
    std::unique_ptr<CompositeOwnerQuadrature<dstype, Int>> sidecar;
    rebuild(rs, op, nm, rt, sidecar);
    std::vector<std::vector<dstype>> before, rebuilt, fresh;
    runtime_cycle<Model>(op, *rt, nm, sc.model, before);
    long long c0[5], c1[5];
    std::vector<long long> p0, p1, pf;
    counts(nm, c0, p0);

    // Move the geometry in place and rebuild the same nm/runtime objects.
    rs.xdg = moved.xdg;
    rs.sol.xdg = rs.xdg.data();
    rebuild(rs, op, nm, rt, sidecar);
    counts(nm, c1, p1);
    bool changed[3] = {false, false, false};
    for (int r = 0; r < n; r++)
        for (int k = 0; k < 3; k++) changed[k] = changed[k] || p1[3 * r + k] != p0[3 * r + k];
    // With one rank per model every donor point and slot stays on the single
    // donor rank (slots = owner faces), so only the pairs can change there.
    check(changed[2] && c1[2] != c0[2], tag + ": rebuild changes the pair counts");
    check(changed[0] || sc.k == 1, tag + ": rebuild changes the per-rank donor point counts");
    check(changed[1] || sc.k == 1, tag + ": rebuild changes the per-rank slot counts");
    runtime_cycle<Model>(op, *rt, nm, sc.model, rebuilt);

    // Production host CPU path on the rebuilt setup data (independent of the runtime).
    for (int withJacobian = 0; withJacobian < 2; withJacobian++) {
        exasim::nonmatching::compute_nonmatching_coupling<Model>(nm, op.sol, op.res, op.app, op.master, op.mesh,
                op.tmp, op.common, 0, withJacobian == 1, 0, sidecar.get());
        for (int d = 0; d < 2; d++) {
            const std::vector<dstype>& ri = rebuilt[withJacobian == 0 ? d : 3 + 2 * d];
            bool same = true;
            for (Int i = 0; i < nm.dir[d].szRi; i++) same = same && nm.dir[d].Ri[i] == ri[i];
            if (withJacobian == 1) {
                const std::vector<dstype>& hi = rebuilt[4 + 2 * d];
                for (Int i = 0; i < nm.dir[d].szHi; i++) same = same && nm.dir[d].Hi[i] == hi[i];
            }
            check(same, tag + ": rebuilt runtime residual/matrix equal the host CPU path bitwise");
        }
    }

    // Fresh construction directly on the moved geometry.
    RankStructs rsFresh;
    RankOperator opFresh;
    prepare(rsFresh, opFresh, moved);
    nonmatchingdata nmFresh;
    std::unique_ptr<Runtime> rtFresh;
    std::unique_ptr<CompositeOwnerQuadrature<dstype, Int>> sidecarFresh;
    rebuild(rsFresh, opFresh, nmFresh, rtFresh, sidecarFresh);
    long long cf[5];
    counts(nmFresh, cf, pf);
    bool sameCounts = pf == p1;
    for (int k = 0; k < 5; k++) sameCounts = sameCounts && cf[k] == c1[k];
    check(sameCounts, tag + ": rebuilt and fresh counts agree");
    runtime_cycle<Model>(opFresh, *rtFresh, nmFresh, sc.model, fresh);

    const char* what[] = {"Ri dir0 (residual)", "Ri dir1 (residual)", "owner-assembled residual",
                          "Ri dir0 (Jacobian pass)", "Hi dir0", "Ri dir1 (Jacobian pass)", "Hi dir1",
                          "hdgMatVec product"};
    bool identical = rebuilt.size() == fresh.size();
    for (std::size_t k = 0; identical && k < rebuilt.size(); k++) {
        const bool ok = rebuilt[k] == fresh[k];
        check(ok, tag + ": rebuilt equals fresh bitwise: " + what[k]);
        identical = identical && ok;
    }
    int localNonzero = 0;
    for (dstype x : rebuilt[2]) localNonzero |= x != 0;
    int nonzero = 0;
    MPI_Allreduce(&localNonzero, &nonzero, 1, MPI_INT, MPI_MAX, sc.comm);
    check(nonzero == 1, tag + ": owner assembly after rebuild delivers coupling residuals");
    if (worldRank == 0)
        std::printf("%s: counts (points, slots, pairs, send nb, recv nb) %lld %lld %lld %lld %lld -> "
                    "%lld %lld %lld %lld %lld; residual, Jacobian, exchange, owner assembly and "
                    "hdgMatVec after rebuild %s a fresh construction bitwise\n",
                    tag.c_str(), c0[0], c0[1], c0[2], c0[3], c0[4], c1[0], c1[1], c1[2], c1[3], c1[4],
                    identical ? "equal" : "DIFFER FROM");

    exasim::nonmatching::free_nonmatching_data(nm);
    exasim::nonmatching::free_nonmatching_data(nmFresh);
    for (RankOperator* o : {&op, &opFresh}) {
        CPUFREE(o->common.requests);
        CPUFREE(o->common.statuses);
    }
}

// ---------------------------------------------------------------------------
// [nonmatching] Phase 9 B1 diagnostic: consistency of the two transfers of the
// coupled Poisson example on straight unequal meshes (one element column per
// model, x = 0.5 interface). Model 1 donates uhat (fint = -uh) to model-2 owner
// points; model 2 donates the flux q.n + tau (u - uh) to model-1 owner points.
// The nodal state is the nodal interpolant of an exact field: u, q = -grad u at
// the element nodes, uh = u at the trace nodes. For each donor direction:
//   eu, eq, euh   max error of the transferred u, q and uhat at the owner points
//                 (production build_coupling_point_data, incl. the d-correction);
//   eR            max |Ri - Ri_exact| / h_owner, where Ri is the production
//                 uncondensed slot residual (eq. (40)) and Ri_exact the same owner
//                 quadrature applied to the exact u, q, uhat at the same points;
//   pd            max |d| of the trace correction (zero on a straight interface).
// Exact field 0: sin(pi x + 0.7) sin(pi y + 0.3) (rates; shifted so that q.n does
// not vanish on x = 0.5, unlike the example's sin(pi x) sin(pi y)); 1: degree-p polynomial
// (0.3 + x + 2y)^p - 0.7 y^p, reproduced exactly by Q_p and the owner rule.
void exact_field(int kind, int p, dstype x, dstype y, dstype& u, dstype& qx, dstype& qy)
{
    const dstype pi = 3.14159265358979323846;
    if (kind == 0) {
        u = std::sin(pi * x + 0.7) * std::sin(pi * y + 0.3);
        qx = -pi * std::cos(pi * x + 0.7) * std::sin(pi * y + 0.3);
        qy = -pi * std::sin(pi * x + 0.7) * std::cos(pi * y + 0.3);
        return;
    }
    const dstype a = 0.3 + x + 2 * y;
    u = std::pow(a, p) - 0.7 * std::pow(y, p);
    qx = -p * std::pow(a, p - 1);
    qy = -(2 * p * std::pow(a, p - 1) - 0.7 * p * std::pow(y, p - 1));
}

// Returns, on every rank of the first two, {eu, eq, euh, eR, Rscale, pd} of the
// direction donated by model 1 (entries 0..5) and by model 2 (entries 6..11).
void b1_transfer(int ny1, int ny2, MasterData& mst, int kind, dstype result[12])
{
    std::fill(result, result + 12, 0.0);
    SubComm sc(2);
    if (sc.comm == MPI_COMM_NULL) return;
    const int p = mst.porder;
    const dstype h[2] = {1.0 / ny1, 1.0 / ny2};
    ModelMesh mm;
    if (sc.model == 0) { mm.xs = {0.5 - h[0], 0.5}; mm.ys = uniform(0.0, 1.0, ny1); mm.xfar = 0.5 - h[0]; }
    else               { mm.xs = {0.5, 0.5 + h[1]}; mm.ys = uniform(0.0, 1.0, ny2); mm.xfar = 0.5 + h[1]; }
    mm.xiface = 0.5;
    const std::string tag = "B1 transfer";
    LocalMesh lm;
    build_local_mesh(lm, mm, mst, 0, 1, tag);
    RankStructs rs;
    fill_rank_structs(rs, lm, mst, 1, 1, 1);
    nonmatchingdata nm;
    const bool ran = exasim::nonmatching::setup_nonmatching_interface(rs.app, rs.master, rs.mesh, rs.sol,
            Int(2), static_cast<Int>(worldRank), static_cast<Int>(sc.model), nm);
    check(ran && nm.interfacetype == 1, tag + ": setup ran");
    RankOperator op;
    init_operator<PoissonCoupling>(op, lm, mst, sc.model);
    use_nonmatching(op);
    op.param[0] = sc.model == 0 ? 0.0 : 1.0;

    const Int npe = op.npe, npf = op.npf, nfe = op.nfe, nd = op.nd, nc = op.nc;
    for (Int e = 0; e < op.ne; e++)
        for (Int j = 0; j < npe; j++) {
            dstype u, qx, qy;
            exact_field(kind, p, lm.xdg[j + npe * (0 + nd * e)], lm.xdg[j + npe * (1 + nd * e)], u, qx, qy);
            op.udg[j + npe * (0 + nc * e)] = u;
            op.udg[j + npe * (1 + nc * e)] = qx;
            op.udg[j + npe * (2 + nc * e)] = qy;
        }
    for (Int e = 0; e < op.ne; e++)
        for (Int a = 0; a < nfe; a++)
            for (Int j = 0; j < npf; j++) {
                const Int node = mst.perm[j + npf * a];
                dstype u, qx, qy;
                exact_field(kind, p, lm.xdg[node + npe * (0 + nd * e)], lm.xdg[node + npe * (1 + nd * e)], u, qx, qy);
                op.uh[op.elemcon[j + npf * a + npf * nfe * e]] = u;
            }

    dstype local[6] = {0, 0, 0, 0, 0, 0};
    for (int direction = 0; direction < 2; direction++) {
        auto& donor = nm.dir[direction];
        if (!donor.isdonor || donor.npts == 0) continue;
        const Int npts = donor.npts;
        exasim::nonmatching::CouplingPointData<dstype, Int> data;
        exasim::nonmatching::build_coupling_point_data(data, donor, op.sol, op.mesh, op.common);
        std::vector<dstype> ue(npts), qxe(npts), qye(npts);
        for (Int q = 0; q < npts; q++) {
            exact_field(kind, p, data.x[q], data.x[q + npts], ue[q], qxe[q], qye[q]);
            local[0] = std::max(local[0], std::abs(data.udg[q] - ue[q]));
            local[1] = std::max(local[1], std::max(std::abs(data.udg[q + npts] - qxe[q]),
                                                   std::abs(data.udg[q + 2 * npts] - qye[q])));
            local[2] = std::max(local[2], std::abs(data.uhg[q] - ue[q]));
            for (Int d = 0; d < nd; d++) local[5] = std::max(local[5], std::abs(donor.pd[q + npts * d]));
        }
        exasim::nonmatching::evaluate_coupling_flux<PoissonCoupling>(data, donor, op.app, op.master, op.mesh,
                op.sol, op.tmp, op.common, false, 0);
        exasim::nonmatching::form_uncondensed_slot_residual(donor, op.master, data);
        const std::vector<dstype> Ri(donor.Ri, donor.Ri + donor.szRi);
        for (Int q = 0; q < npts; q++) {
            data.udg[q] = ue[q];
            data.udg[q + npts] = qxe[q];
            data.udg[q + 2 * npts] = qye[q];
            data.uhg[q] = ue[q];
        }
        exasim::nonmatching::evaluate_coupling_flux<PoissonCoupling>(data, donor, op.app, op.master, op.mesh,
                op.sol, op.tmp, op.common, false, 0);
        exasim::nonmatching::form_uncondensed_slot_residual(donor, op.master, data);
        for (Int i = 0; i < donor.szRi; i++) {
            local[3] = std::max(local[3], std::abs(Ri[i] - donor.Ri[i]));
            local[4] = std::max(local[4], std::abs(donor.Ri[i]));
        }
    }
    local[3] /= h[1 - sc.model];   // owner mesh size of this donor direction
    MPI_Allgather(local, 6, MPI_DOUBLE, result, 6, MPI_DOUBLE, sc.comm);
    exasim::nonmatching::free_nonmatching_data(nm);
}

void b1_transfer_study()
{
    struct Cfg { const char* name; int r1, r2; } cfgs[] = {{"A", 4, 6}, {"S", 6, 4}, {"R", 4, 8}};
    const int levels[] = {1, 2, 4, 8};
    const char* kinds[] = {"smooth", "polynomial"};
    for (int p = 1; p <= 4; p++) {
        MasterData mst = make_master(1, p);
        for (const Cfg& cfg : cfgs)
            for (int kind = 0; kind < 2; kind++) {
                dstype r[4][12];
                for (int l = 0; l < 4; l++) b1_transfer(cfg.r1 * levels[l], cfg.r2 * levels[l], mst, kind, r[l]);
                if (worldRank >= 2) continue;
                for (int side = 0; side < 2; side++) {
                    const char* dir = side == 0 ? "uhat 1->2" : "flux 2->1";
                    const int rd = side == 0 ? cfg.r1 : cfg.r2, ro = side == 0 ? cfg.r2 : cfg.r1;
                    const char* sense = rd < ro ? "coarse->fine" : (rd > ro ? "fine->coarse" : "equal");
                    const dstype* a = r[0] + 6 * side;
                    const dstype* b = r[3] + 6 * side;
                    const dstype* c = r[2] + 6 * side;
                    if (kind == 1) {
                        dstype worst = 0;
                        for (int l = 0; l < 4; l++) {
                            const dstype* x = r[l] + 6 * side;
                            worst = std::max({worst, x[0], x[1], x[2], x[3] / std::max<dstype>(1, x[4])});
                        }
                        check(worst < 1e-11, "B1 polynomial transfer is exact");
                        if (worldRank == 0)
                            std::printf("B1 transfer p=%d %s %s %-10s %-12s max error over N=1..8 %.3e (eu, eq, euh, eR)\n",
                                        p, cfg.name, kinds[kind], dir, sense, worst);
                        continue;
                    }
                    dstype rate[4];
                    for (int k = 0; k < 4; k++) rate[k] = std::log2(c[k] / b[k]);
                    for (int k = 0; k < 4; k++)
                        check(rate[k] > p + 1 - 0.3, "B1 smooth transfer converges at order p+1");
                    if (worldRank == 0)
                        std::printf("B1 transfer p=%d %s %s %-10s %-12s N=8: eu %.3e eq %.3e euh %.3e eR/h %.3e "
                                    "|d| %.1e; rate N=4->8: eu %.2f eq %.2f euh %.2f eR/h %.2f (N=1: eR/h %.3e)\n",
                                    p, cfg.name, kinds[kind], dir, sense, b[0], b[1], b[2], b[3], b[5],
                                    rate[0], rate[1], rate[2], rate[3], a[3]);
                }
            }
    }
}

// ---------------------------------------------------------------------------
// [nonmatching] Phase 9 A1: composite owner-face quadrature on coincident 2D interfaces
// (docs/nonmatching/reports/phase9_plan_amendment_r1.md, revision 3). Modes:
//   --quadrature             T1-T7 and the coincidence threshold at n = 2, 4, ..., worldSize
//   --quadrature-mismatch    expected failure: option differs across ranks
//   --quadrature-piece-limit expected failure: more pieces than maxCompositePiecesPerFace
//   --quadrature-cost        A/S/R measurements (points, slots, pairs, timing, memory)
// Fixture (one element column per model, interface x = 0.5, p = 2): model 1 has faces of
// length 1/4; model 2 has graded rows with a short face [0.29, 0.30] lying between the
// owner Gauss points 0.2782 and 0.375 of the model-1 face [0.25, 0.5]. Its neighbours
// (rows 2 and 4) are on different ranks for 2 and 4 ranks per model.

using CompositeQuad = exasim::nonmatching::CompositeOwnerQuadrature<dstype, Int>;

const std::vector<dstype> quadYs1 = {0.0, 0.25, 0.5, 0.75, 1.0};
const std::vector<dstype> quadYs2 = {0.0, 0.12, 0.2, 0.29, 0.30, 0.5, 0.65, 0.8, 1.0};

ModelMesh column_mesh(int model, const std::vector<dstype>& ys, dstype shift = 0, dstype amp = 0)
{
    ModelMesh mm;
    mm.ys = ys;
    if (model == 0) { mm.xs = {0.25, 0.5}; mm.xiface = 0.5; mm.xfar = 0.25; }
    else { mm.xs = {0.5 + shift, 0.6 + shift}; mm.xiface = 0.5 + shift; mm.xfar = 0.6 + shift; mm.amp = amp; }
    return mm;
}

// One type-1 setup of this rank's model on the current SubComm.
struct QuadSetup {
    LocalMesh lm;
    RankStructs rs;
    nonmatchingdata nm;
    CompositeQuad quad;
    std::unique_ptr<RankOperator> op;
    bool withQuad = true;
    ~QuadSetup() { exasim::nonmatching::free_nonmatching_data(nm); }
};

void quad_setup(QuadSetup& s, const SubComm& sc, const ModelMesh& mm, MasterData& mst, int n,
                bool option, Int maxPieces = 64, bool withQuad = true, Int endpointIterations = -1)
{
    s.withQuad = withQuad;
    build_local_mesh(s.lm, mm, mst, sc.part, sc.k, "quadrature");
    fill_rank_structs(s.rs, s.lm, mst, 1, 1, 1);
    s.rs.master.gpf = mst.gpf.data();
    exasim::nonmatching::DonorSearchParams prm;
    prm.compositeOwnerQuadrature = option;
    prm.maxCompositePiecesPerFace = maxPieces;
    prm.endpointMaxProjectionIterations = endpointIterations;
    const bool ran = exasim::nonmatching::setup_nonmatching_interface(s.rs.app, s.rs.master, s.rs.mesh,
            s.rs.sol, static_cast<Int>(n), static_cast<Int>(worldRank), static_cast<Int>(sc.model * sc.k), s.nm,
            static_cast<exasim::nonmatching::DonorSearchResult<dstype, Int>*>(nullptr), prm, true,
            static_cast<exasim::nonmatching::SearchStatistics*>(nullptr), withQuad ? &s.quad : nullptr);
    check(ran && s.nm.interfacetype == 1, "quadrature: setup ran");
    s.op = std::make_unique<RankOperator>();
    init_operator<PoissonCoupling>(*s.op, s.lm, mst, sc.model);
    use_nonmatching(*s.op);
    s.op->param[0] = sc.model == 0 ? 0.0 : 1.0;
}

// Piecewise polynomial state (degree 2 in y): u and q per element, uh per interface face,
// coefficients from the global element / face number and the model.
dstype quad_poly(long key, int seed, dstype y)
{
    return (hv(3 * key, seed) - 0.5) + (hv(3 * key + 1, seed) - 0.5) * y + (hv(3 * key + 2, seed) - 0.5) * y * y;
}

void quad_fill_state(QuadSetup& s, const MasterData& mst, int model)
{
    RankOperator& op = *s.op;
    const Int npe = op.npe, npf = op.npf, nfe = op.nfe, nd = op.nd, nc = op.nc;
    for (Int e = 0; e < op.ne; e++) {
        const long ge = s.lm.elempart[e];
        for (Int j = 0; j < npe; j++) {
            const dstype y = s.lm.xdg[j + npe * (1 + nd * e)];
            op.udg[j + npe * (0 + nc * e)] = quad_poly(ge, 300 + model, y);
            op.udg[j + npe * (1 + nc * e)] = quad_poly(ge, 400 + model, y);
            op.udg[j + npe * (2 + nc * e)] = 0.3;
        }
        for (Int a = 0; a < nfe; a++)
            for (Int j = 0; j < npf; j++) {
                const Int node = mst.perm[j + npf * a];
                const dstype y = s.lm.xdg[node + npe * (1 + nd * e)];
                op.uh[op.elemcon[j + npf * a + npf * nfe * e]] = quad_poly(nfe * ge + a, 500 + model, y);
            }
    }
}

// Gathers fixed-length double rows to rank 0 of comm (rows of length w).
std::vector<std::vector<dstype>> gather_rows(const std::vector<dstype>& local, int w, MPI_Comm comm)
{
    int size = 0, rank = 0;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    int count = static_cast<int>(local.size());
    std::vector<int> counts(size), displs(size, 0);
    MPI_Gather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);
    std::vector<dstype> all;
    if (rank == 0) {
        for (int q = 1; q < size; q++) displs[q] = displs[q - 1] + counts[q - 1];
        all.resize(displs[size - 1] + counts[size - 1]);
    }
    MPI_Gatherv(local.data(), count, MPI_DOUBLE, all.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
    std::vector<std::vector<dstype>> rows;
    for (std::size_t i = 0; i + w <= all.size(); i += w) rows.emplace_back(all.begin() + i, all.begin() + i + w);
    std::sort(rows.begin(), rows.end());
    return rows;
}

// Revision 2: every field of the frozen nonmatchinginterfaceT of both directions, bitwise:
// identification, cached sizes and counts, the owner arrays (ownface, ownfaceglobal, recvnb,
// recvcount, recvface), every per-point array including pdphibar (and the null/non-null state
// of each pointer), slots, pairs, send lists, the allocated sizes and the work arrays Ri, Hi,
// sendbuf and recvbuf. The work arrays are deterministic in these tests: setup zero-fills them
// (allocate_zeroed) and the residual/Jacobian evaluations write Ri and Hi; no exchange runs,
// so sendbuf and recvbuf stay zero-filled. On a mismatch the first differing field is printed.
bool same_interface(const nonmatchingdata& a, const nonmatchingdata& b, const std::string& tag = "")
{
    std::string first;
    auto note = [&](bool ok, const std::string& name) { if (!ok && first.empty()) first = name; return ok; };
    note(a.interfacetype == b.interfacetype, "interfacetype");
    for (int k = 0; k < 2; k++) {
        const auto& x = a.dir[k]; const auto& y = b.dir[k];
        const std::string d = "dir" + std::to_string(k) + ".";
        auto eqS = [&](Int u, Int v, const char* name) { return note(u == v, d + name); };
        bool sizes = eqS(x.direction, y.direction, "direction") & eqS(x.isowner, y.isowner, "isowner") &
            eqS(x.isdonor, y.isdonor, "isdonor") & eqS(x.nd, y.nd, "nd") & eqS(x.npe, y.npe, "npe") &
            eqS(x.npf, y.npf, "npf") & eqS(x.nfe, y.nfe, "nfe") & eqS(x.ngf, y.ngf, "ngf") &
            eqS(x.ncu, y.ncu, "ncu") & eqS(x.ncq, y.ncq, "ncq") & eqS(x.ncu12, y.ncu12, "ncu12") &
            eqS(x.nownfaces, y.nownfaces, "nownfaces") & eqS(x.nrecvnb, y.nrecvnb, "nrecvnb") &
            eqS(x.npts, y.npts, "npts") & eqS(x.nslots, y.nslots, "nslots") & eqS(x.npairs, y.npairs, "npairs") &
            eqS(x.nsendnb, y.nsendnb, "nsendnb") & eqS(x.szRi, y.szRi, "szRi") & eqS(x.szHi, y.szHi, "szHi") &
            eqS(x.szsendbuf, y.szsendbuf, "szsendbuf") & eqS(x.szrecvbuf, y.szrecvbuf, "szrecvbuf");
        if (!sizes) continue;
        auto eqI = [&](const Int* p, const Int* q, Int n, const char* name) {
            bool ok = (p == nullptr) == (q == nullptr) && (n == 0 || !p || std::equal(p, p + n, q));
            return note(ok, d + name);
        };
        auto eqT = [&](const dstype* p, const dstype* q, Int n, const char* name) {
            bool ok = (p == nullptr) == (q == nullptr) &&
                      (n == 0 || !p || std::memcmp(p, q, sizeof(dstype) * static_cast<std::size_t>(n)) == 0);
            return note(ok, d + name);
        };
        const Int P = x.npts, nd = x.nd, npe = x.npe, npf = x.npf;
        // recvface length from each side's own recvcount (the counts arrays are compared above
        // only when both exist); recvface is compared only when the totals agree, so neither
        // allocation is read beyond its own length.
        auto recv_total = [](const nonmatchinginterfaceT<dstype, Int>& z) {
            long long total = 0;
            for (Int q = 0; q < z.nrecvnb && z.recvcount; q++) total += z.recvcount[q];
            return total;
        };
        const long long xRecv = recv_total(x), yRecv = recv_total(y);
        const bool recvTotalsAgree = note(xRecv == yRecv, d + "recvface length (sum of recvcount)");
        const Int nrecvblocks = static_cast<Int>(xRecv);
        eqI(x.ownface, y.ownface, x.nownfaces, "ownface");
        eqI(x.ownfaceglobal, y.ownfaceglobal, x.nownfaces, "ownfaceglobal");
        eqI(x.recvnb, y.recvnb, x.nrecvnb, "recvnb");
        eqI(x.recvcount, y.recvcount, x.nrecvnb, "recvcount");
        if (recvTotalsAgree) eqI(x.recvface, y.recvface, nrecvblocks, "recvface");
        eqI(x.pe, y.pe, P, "pe"); eqI(x.pa, y.pa, P, "pa"); eqI(x.pf, y.pf, P, "pf");
        eqI(x.pslot, y.pslot, P, "pslot"); eqI(x.pg, y.pg, P, "pg");
        eqT(x.px, y.px, P * nd, "px"); eqT(x.pnl, y.pnl, P * nd, "pnl"); eqT(x.pwJ, y.pwJ, P, "pwJ");
        eqT(x.pxi, y.pxi, P * (nd - 1), "pxi"); eqT(x.pzeta, y.pzeta, P * nd, "pzeta");
        eqT(x.pphi, y.pphi, P * npe, "pphi"); eqT(x.pphibar, y.pphibar, P * npe, "pphibar");
        eqT(x.ppsi, y.ppsi, P * npf, "ppsi"); eqT(x.pd, y.pd, P * nd, "pd");
        eqT(x.pdphibar, y.pdphibar, x.pdphibar ? P * nd * npe : 0, "pdphibar");
        eqI(x.slotrank, y.slotrank, x.nslots, "slotrank"); eqI(x.slotface, y.slotface, x.nslots, "slotface");
        eqI(x.slotptr, y.slotptr, x.nslots + (x.nslots > 0), "slotptr");
        eqI(x.pairslot, y.pairslot, x.npairs, "pairslot"); eqI(x.pairelem, y.pairelem, x.npairs, "pairelem");
        eqI(x.pairptr, y.pairptr, x.npairs + (x.npairs > 0), "pairptr");
        eqI(x.sendnb, y.sendnb, x.nsendnb, "sendnb"); eqI(x.sendcount, y.sendcount, x.nsendnb, "sendcount");
        eqT(x.Ri, y.Ri, x.szRi, "Ri"); eqT(x.Hi, y.Hi, x.szHi, "Hi");
        eqT(x.sendbuf, y.sendbuf, x.szsendbuf, "sendbuf"); eqT(x.recvbuf, y.recvbuf, x.szrecvbuf, "recvbuf");
    }
    if (!first.empty() && !tag.empty())
        std::printf("[rank %d] %s: first differing field %s\n", worldRank, tag.c_str(), first.c_str());
    return first.empty();
}

// Revision 2: the complete composite-quadrature sidecar, bitwise (flags, diagnostics and every
// vector of both directions).
bool same_quad(const exasim::nonmatching::CompositeOwnerQuadrature<dstype, Int>& a,
               const exasim::nonmatching::CompositeOwnerQuadrature<dstype, Int>& b, const std::string& tag = "")
{
    std::string first;
    auto note = [&](bool ok, const std::string& name) { if (!ok && first.empty()) first = name; };
    auto eqV = [](const std::vector<dstype>& u, const std::vector<dstype>& v) {
        return u.size() == v.size() && (u.empty() || std::memcmp(u.data(), v.data(), sizeof(dstype) * u.size()) == 0);
    };
    note(a.requested == b.requested, "requested");
    for (int k = 0; k < 2; k++) {
        const auto& x = a.dir[k]; const auto& y = b.dir[k];
        const std::string d = "dir" + std::to_string(k) + ".";
        note(x.applied == y.applied, d + "applied");
        note(x.coincident == y.coincident, d + "coincident");
        note(x.breakpointsTotal == y.breakpointsTotal, d + "breakpointsTotal");
        note(x.maxPiecesPerFace == y.maxPiecesPerFace, d + "maxPiecesPerFace");
        note(x.facebreakptr == y.facebreakptr, d + "facebreakptr");
        note(eqV(x.breakpoints, y.breakpoints), d + "breakpoints");
        note(eqV(x.ptest, y.ptest), d + "ptest");
        note(eqV(x.ps, y.ps), d + "ps");
        note(x.ppiece == y.ppiece, d + "ppiece");
    }
    if (!first.empty() && !tag.empty())
        std::printf("[rank %d] %s: first differing sidecar field %s\n", worldRank, tag.c_str(), first.c_str());
    return first.empty();
}

// Host residual and Jacobian with the sidecar (the runtime path is compared separately).
void quad_compute(QuadSetup& s, bool withJacobian)
{
    RankOperator& op = *s.op;
    exasim::nonmatching::compute_nonmatching_coupling<PoissonCoupling>(s.nm, op.sol, op.res, op.app,
            op.master, op.mesh, op.tmp, op.common, 0, withJacobian, 0, s.withQuad ? &s.quad : nullptr);
}

// 12-point Gauss-Legendre on [-1, 1] (Newton on P_12), independent of the solver tables.
void gauss_legendre12(std::vector<dstype>& x, std::vector<dstype>& w)
{
    const int n = 12;
    x.resize(n); w.resize(n);
    const dstype pi = 3.14159265358979323846;
    for (int i = 0; i < n; i++) {
        dstype z = std::cos(pi * (i + 0.75) / (n + 0.5)), dp = 0;
        for (int it = 0; it < 100; it++) {
            dstype p0 = 1, p1 = z;
            for (int k = 2; k <= n; k++) { const dstype p2 = ((2 * k - 1) * z * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p2; }
            dp = n * (z * p1 - p0) / (z * z - 1);
            const dstype dz = p1 / dp;
            z -= dz;
            if (std::abs(dz) < 1e-16) break;
        }
        x[i] = z; w[i] = 2 / ((1 - z * z) * dp * dp);
    }
}

// Independent analytic reference R_i = -int_F psi_i(s(y)) f(y) dy of every owner face of
// direction k, from the test's own donor intervals (quadYs of the donor model) and its own
// Lagrange basis; f is the donor's piecewise data: -uh (k = 1, donor model 1) or
// q.n + tau (u - uh) with n = (-1, 0), tau = 1 (k = 0, donor model 2).
std::vector<dstype> quad_reference(int k, const std::vector<dstype>& ownerNodesY, const MasterData& mst,
        const std::vector<dstype>& donorYs, long donorFaceLocal, int donorNfe = 4,
        bool donorTriangle = false, const std::vector<dstype>* testNodes = nullptr)
{
    const int npf = mst.npf, donorModel = 1 - k;
    std::vector<dstype> gx, gw;
    gauss_legendre12(gx, gw);
    auto lagrange = [&](const std::vector<dstype>& nodes, dstype s, int i) {
        dstype v = 1;
        for (int m = 0; m < npf; m++) if (m != i) v *= (s - nodes[m]) / (nodes[i] - nodes[m]);
        return v;
    };
    dstype y0 = 0, y1 = 0;
    for (int j = 0; j < npf; j++) {
        y0 += lagrange(mst.xpf, 0, j) * ownerNodesY[j];
        y1 += lagrange(mst.xpf, 1, j) * ownerNodesY[j];
    }
    const dstype ya = std::min(y0, y1), yb = std::max(y0, y1);
    const std::vector<dstype>& test = testNodes ? *testNodes : mst.xpf;
    std::vector<dstype> R(npf, 0);
    for (std::size_t r = 0; r + 1 < donorYs.size(); r++) {
        const dstype a = std::max(ya, donorYs[r]), b = std::min(yb, donorYs[r + 1]);
        if (!(b > a)) continue;
        // One logical cell per row. A triangular cell is split into T0/T1;
        // the left model's interface uses T0 and the right model uses T1.
        const long ge = donorTriangle ? 2 * static_cast<long>(r) + donorModel
                                      : static_cast<long>(r);
        for (int g = 0; g < 12; g++) {
            const dstype y = 0.5 * (a + b) + 0.5 * (b - a) * gx[g], wy = 0.5 * (b - a) * gw[g];
            const dstype uh = quad_poly(donorNfe * ge + donorFaceLocal, 500 + donorModel, y);
            dstype f = -uh;
            if (k == 0)
                f = -quad_poly(ge, 400 + donorModel, y) + (quad_poly(ge, 300 + donorModel, y) - uh);
            const dstype s = (y - y0) / (y1 - y0);
            for (int i = 0; i < npf; i++) R[i] -= wy * lagrange(test, s, i) * f;
        }
    }
    return R;
}

struct QuadSignature {
    std::vector<std::vector<dstype>> breakpoints, points, pairs, residual;
};
std::vector<QuadSignature> quadSignatures; // world rank 0: one per n (T2)

// One full T1/T2/T5/T6 run on n ranks; checks on rank 0 of the SubComm.
void quadrature_fixture_case(MasterData& carrier, int n, bool mixedFamilies = false,
                             bool model0Triangle = false)
{
    SubComm sc(n);
    if (sc.comm == MPI_COMM_NULL) return;
    MasterData triangle, quad;
    if (mixedFamilies) {
        triangle = make_master(0, carrier.porder);
        quad = make_master(1, carrier.porder);
    }
    MasterData& mst = mixedFamilies ?
            (((sc.model == 0) == model0Triangle) ? triangle : quad) : carrier;
    const std::string tag = std::string(mixedFamilies ?
            (model0Triangle ? "quadrature mixed tri-quad (" : "quadrature mixed quad-tri (") :
            "quadrature (") + std::to_string(sc.k) + " per model)";
    if (sc.k >= 2) check((2 * sc.k) / 8 != (4 * sc.k) / 8, tag + ": short face neighbours on different ranks");
    ModelMesh mm = column_mesh(sc.model, sc.model == 0 ? quadYs1 : quadYs2);
    if (mixedFamilies) mm.tri = ((sc.model == 0) == model0Triangle);
    QuadSetup on, off;
    quad_setup(on, sc, mm, mst, n, true);
    quad_setup(off, sc, mm, mst, n, false);
    for (int k = 0; k < 2; k++) {
        check(on.quad.dir[k].coincident == 1 && on.quad.dir[k].applied == 1, tag + ": both directions applied");
        check(off.quad.requested == 0 && off.quad.dir[k].applied == 0, tag + ": option off applies nothing");
    }
    quad_fill_state(on, mst, sc.model);
    quad_fill_state(off, mst, sc.model);
    quad_compute(on, false);
    quad_compute(off, false);

    const Int npf = mst.npf, nfe = mst.nfe, npe = mst.npe;
    const int own = sc.model;
    // Breakpoints of the own faces in physical y: [k, owner face, y].
    std::vector<dstype> bp, ownerNodes;
    {
        const auto& side = on.quad.dir[own];
        for (Int l = 0; l < on.nm.dir[own].nownfaces; l++) {
            const Int e = on.lm.intfaces[l] / nfe, lf = on.lm.intfaces[l] % nfe;
            std::vector<dstype> yn(npf);
            for (Int j = 0; j < npf; j++) yn[j] = on.lm.xdg[mst.perm[j + npf * lf] + npe * (1 + 2 * e)];
            ownerNodes.push_back(own); ownerNodes.push_back(on.nm.dir[own].ownfaceglobal[l]);
            ownerNodes.insert(ownerNodes.end(), yn.begin(), yn.end());
            for (Int b = side.facebreakptr[l]; b < side.facebreakptr[l + 1]; b++) {
                const dstype s = side.breakpoints[b];
                dstype y = 0;
                for (Int j = 0; j < npf; j++) {
                    dstype v = 1;
                    for (Int m = 0; m < npf; m++) if (m != j) v *= (s - mst.xpf[m]) / (mst.xpf[j] - mst.xpf[m]);
                    y += v * yn[j];
                }
                bp.push_back(own); bp.push_back(on.nm.dir[own].ownfaceglobal[l]); bp.push_back(s); bp.push_back(y);
            }
        }
    }
    // Donor-side quadrature tuples and pair keys; per-slot residuals (composite and off).
    std::vector<dstype> pts, prs, res, resOff;
    for (int k = 0; k < 2; k++) {
        const auto& d = on.nm.dir[k];
        if (!d.isdonor) continue;
        const auto& side = on.quad.dir[k];
        for (Int p = 0; p < d.npts; p++) {
            const Int face = d.slotface[d.pslot[p]];
            pts.push_back(k); pts.push_back(face); pts.push_back(side.ppiece[p]); pts.push_back(d.pg[p]);
            pts.push_back(d.px[p]); pts.push_back(d.px[p + d.npts]); pts.push_back(d.pwJ[p]);
            for (Int i = 0; i < npf; i++) pts.push_back(side.ptest[i + npf * p]);
            prs.push_back(k); prs.push_back(face); prs.push_back(d.pf[p]); prs.push_back(on.lm.elempart[d.pe[p]]);
        }
        for (Int s = 0; s < d.nslots; s++) {
            res.push_back(k); res.push_back(d.slotface[s]);
            for (Int i = 0; i < npf; i++) res.push_back(d.Ri[i + npf * s]);
        }
        const auto& o = off.nm.dir[k];
        for (Int s = 0; s < o.nslots; s++) {
            resOff.push_back(k); resOff.push_back(o.slotface[s]);
            for (Int i = 0; i < npf; i++) resOff.push_back(o.Ri[i + npf * s]);
        }
    }
    // Interface local face of each model (one column: every element has the same one).
    dstype ifaceLocal[2] = {-1, -1}, mine = on.lm.intfaces.empty() ? -1 : on.lm.intfaces[0] % nfe;
    {
        dstype local[2] = {-1, -1}; local[own] = mine;
        MPI_Allreduce(local, ifaceLocal, 2, MPI_DOUBLE, MPI_MAX, sc.comm);
    }
    if (mixedFamilies && model0Triangle) {
        int localPa3 = 0;
        if (sc.model == 1 && on.nm.dir[0].isdonor)
            for (Int p = 0; p < on.nm.dir[0].npts; ++p)
                localPa3 = std::max(localPa3, on.nm.dir[0].pa[p] == 3 ? 1 : 0);
        int globalPa3 = 0;
        MPI_Allreduce(&localPa3, &globalPa3, 1, MPI_INT, MPI_MAX, sc.comm);
        check(ifaceLocal[1] == 3 && globalPa3 == 1,
              tag + ": triangle owner uses quadrilateral donor local face 3");
    }
    QuadSignature sig;
    sig.breakpoints = gather_rows(bp, 4, sc.comm);
    sig.points = gather_rows(pts, 7 + npf, sc.comm);
    auto pairRows = gather_rows(prs, 4, sc.comm);
    pairRows.erase(std::unique(pairRows.begin(), pairRows.end()), pairRows.end());
    sig.pairs = pairRows;
    auto owners = gather_rows(ownerNodes, 2 + npf, sc.comm);
    auto slotRows = gather_rows(res, 2 + npf, sc.comm);
    auto slotRowsOff = gather_rows(resOff, 2 + npf, sc.comm);
    // Owner-face sums over donor slots (order of the sorted rows).
    auto sum_rows = [&](const std::vector<std::vector<dstype>>& rows) {
        std::map<std::pair<int, long>, std::vector<dstype>> sums;
        for (const auto& r : rows) {
            auto& v = sums[{static_cast<int>(r[0]), static_cast<long>(r[1])}];
            v.resize(npf, 0);
            for (Int i = 0; i < npf; i++) v[i] += r[2 + i];
        }
        std::vector<std::vector<dstype>> out;
        for (const auto& kv : sums) {
            std::vector<dstype> row = {static_cast<dstype>(kv.first.first), static_cast<dstype>(kv.first.second)};
            row.insert(row.end(), kv.second.begin(), kv.second.end());
            out.push_back(row);
        }
        return out;
    };
    sig.residual = sum_rows(slotRows);
    const auto residualOff = sum_rows(slotRowsOff);
    if (sc.comm != MPI_COMM_NULL) {
        int r = 0; MPI_Comm_rank(sc.comm, &r);
        if (r == 0) {
            // T1: breakpoints of the model-1 face [0.25, 0.5] are y = 0.29 and 0.30.
            std::vector<dstype> ys;
            for (const auto& b : sig.breakpoints) {
                if (b[0] != 0) continue;
                const dstype y = b[3];
                if (y > 0.25 && y < 0.5) ys.push_back(y);
            }
            std::sort(ys.begin(), ys.end());
            check(ys.size() == 2 && std::abs(ys[0] - 0.29) < 1e-14 && std::abs(ys[1] - 0.30) < 1e-14,
                  tag + ": short donor face gives breakpoints y = 0.29, 0.30 on [0.25, 0.5]");
            std::map<long, int> perFace;
            for (const auto& p : sig.points) if (p[0] == 0) perFace[static_cast<long>(p[1])]++;
            int maxPts = 0;
            for (const auto& kv : perFace) maxPts = std::max(maxPts, kv.second);
            // T5/T6: both coupling laws against the independent piecewise integrals.
            dstype worst[2] = {0, 0}, worstOff[2] = {0, 0}, scale[2] = {0, 0};
            dstype uniformExact = 0, uniformScale = 0, oldBasisDifference = 0;
            int uniformExactFaces = 0;
            for (const auto& o : owners) {
                const int k = static_cast<int>(o[0]);
                const long face = static_cast<long>(o[1]);
                std::vector<dstype> yn(o.begin() + 2, o.end());
                MasterData& ownerMaster = mixedFamilies ?
                        (((k == 0) == model0Triangle) ? triangle : quad) : carrier;
                MasterData& donorMaster = mixedFamilies ?
                        ((((1 - k) == 0) == model0Triangle) ? triangle : quad) : carrier;
                const auto R = quad_reference(k, yn, ownerMaster, k == 0 ? quadYs2 : quadYs1,
                                              static_cast<long>(ifaceLocal[1 - k]), donorMaster.nfe,
                                              donorMaster.elemtype == 0);
                const auto Rold = quad_reference(k, yn, ownerMaster, k == 0 ? quadYs2 : quadYs1,
                                                 static_cast<long>(ifaceLocal[1 - k]), donorMaster.nfe,
                                                 donorMaster.elemtype == 0, &donorMaster.xpf);
                auto find = [&](const std::vector<std::vector<dstype>>& rows) {
                    for (const auto& r : rows) if (r[0] == k && static_cast<long>(r[1]) == face) return r;
                    return std::vector<dstype>();
                };
                const auto got = find(sig.residual), gotOff = find(residualOff);
                for (Int i = 0; i < npf; i++) {
                    scale[k] = std::max(scale[k], std::abs(R[i]));
                    worst[k] = std::max(worst[k], got.empty() ? 1e300 : std::abs(got[2 + i] - R[i]));
                    worstOff[k] = std::max(worstOff[k], gotOff.empty() ? 1e300 : std::abs(gotOff[2 + i] - R[i]));
                }
                dstype y0 = 0, y1 = 0;
                for (Int j = 0; j < npf; ++j) {
                    dstype l0 = 1, l1 = 1;
                    for (Int m = 0; m < npf; ++m) if (m != j) {
                        l0 *= (0 - ownerMaster.xpf[m]) / (ownerMaster.xpf[j] - ownerMaster.xpf[m]);
                        l1 *= (1 - ownerMaster.xpf[m]) / (ownerMaster.xpf[j] - ownerMaster.xpf[m]);
                    }
                    y0 += l0 * yn[j]; y1 += l1 * yn[j];
                }
                const dstype ya = std::min(y0, y1), yb = std::max(y0, y1);
                const auto& dys = k == 0 ? quadYs2 : quadYs1;
                bool hasInteriorBreakpoint = false;
                for (dstype y : dys)
                    hasInteriorBreakpoint = hasInteriorBreakpoint || (y > ya + 1e-13 && y < yb - 1e-13);
                if (!hasInteriorBreakpoint && !gotOff.empty()) {
                    ++uniformExactFaces;
                    for (Int i = 0; i < npf; ++i) {
                        uniformScale = std::max(uniformScale, std::abs(R[i]));
                        uniformExact = std::max(uniformExact, std::abs(gotOff[2 + i] - R[i]));
                        oldBasisDifference = std::max(oldBasisDifference, std::abs(Rold[i] - R[i]));
                    }
                }
            }
            for (int k = 0; k < 2; k++) {
                check(worst[k] <= 1e-13 * scale[k], tag + (k == 0 ? ": fint = q.n + tau(u - uh) matches the analytic piecewise integral"
                                                                : ": fint = -uh matches the analytic piecewise integral"));
                maxRoundoff = std::max(maxRoundoff, worst[k] / scale[k]);
            }
            if (mixedFamilies && carrier.porder >= 3) {
                check(uniformExactFaces > 0 && uniformExact <= 1e-13 * uniformScale,
                      tag + ": A1-off residual uses the owner test basis on unsplit faces");
                check(oldBasisDifference >= 1e-3 * uniformScale,
                      tag + ": the old donor test basis gives a measurably wrong residual");
            }
            std::printf("%s: breakpoints %zu, max points per model-1 owner face %d, pairs %zu; "
                        "analytic piecewise integral error (relative) fint=q.n+tau(u-uh): composite %.3e, "
                        "uniform %.3e; fint=-uh: composite %.3e, uniform %.3e\n",
                        tag.c_str(), sig.breakpoints.size(), maxPts, sig.pairs.size(),
                        worst[0] / scale[0], worstOff[0] / scale[0], worst[1] / scale[1], worstOff[1] / scale[1]);
            if (worldRank == 0 && !mixedFamilies) quadSignatures.push_back(sig);
        }
    }

    // Runtime path with the sidecar mirror: residual and Jacobian equal the host path bitwise.
    {
        exasim::nonmatching::NonmatchingRuntime<dstype, Int> rt;
        RankOperator& op = *on.op;
        exasim::nonmatching::refresh_nonmatching_runtime(rt, on.nm, op.app, op.mesh, op.common, Int(0), &on.quad);
        for (int k = 0; k < 2; k++)
            check((rt.dir[k].view.ptest != nullptr) == (on.nm.dir[k].isdonor && on.nm.dir[k].npts > 0),
                  tag + ": runtime ptest mirror present on applied donor directions");
        for (int withJacobian = 0; withJacobian < 2; withJacobian++) {
            quad_compute(on, withJacobian == 1);
            exasim::nonmatching::compute_nonmatching_coupling<PoissonCoupling>(rt, op.sol, op.res, op.app,
                    op.master, op.mesh, op.tmp, op.common, 0, withJacobian == 1, 0);
            bool same = true;
            for (int k = 0; k < 2; k++) {
                const auto& h = on.nm.dir[k];
                same = same && (h.szRi == 0 || std::memcmp(h.Ri, rt.dir[k].Ri.data, sizeof(dstype) * h.szRi) == 0);
                if (withJacobian) same = same && (h.szHi == 0 || std::memcmp(h.Hi, rt.dir[k].Hi.data, sizeof(dstype) * h.szHi) == 0);
            }
            check(same, tag + ": composite runtime residual/matrix equal the host path bitwise");
        }
    }

    // Transactional rebuild (Phase 8 API) with the sidecar: the published sidecar, runtime
    // mirror and residual equal the direct setup, and a second rebuild replaces them.
    {
        RankOperator& op = *on.op;
        RankStructs rs2;
        fill_rank_structs(rs2, on.lm, mst, 1, 1, 1);
        rs2.master.gpf = mst.gpf.data();
        rs2.app.interfacefluxmap = op.map.data();
        commonstruct common = op.common;
        common.requests = nullptr; common.statuses = nullptr;
        nonmatchingdata nm2;
        std::unique_ptr<exasim::nonmatching::NonmatchingRuntime<dstype, Int>> rt2;
        std::unique_ptr<CompositeQuad> quad2;
        exasim::nonmatching::DonorSearchParams prm;
        prm.compositeOwnerQuadrature = true;
        for (int pass = 0; pass < 2; pass++)
            exasim::nonmatching::rebuild_nonmatching_interface(rs2.app, rs2.master, rs2.mesh, rs2.sol, common,
                    static_cast<Int>(n), static_cast<Int>(worldRank), static_cast<Int>(sc.model * sc.k), nm2, rt2,
                    Int(0), static_cast<exasim::nonmatching::DonorSearchResult<dstype, Int>*>(nullptr), prm, true, &quad2);
        bool ok = quad2 && rt2 && quad2->requested == 1 && same_quad(*quad2, on.quad, tag + " transactional sidecar");
        for (int k = 0; ok && k < 2; k++)
            ok = (rt2->dir[k].view.ptest != nullptr) == (nm2.dir[k].isdonor && nm2.dir[k].npts > 0);
        check(ok, tag + ": transactional rebuild publishes the complete sidecar and its runtime mirror");
        // Both host evaluations on both objects, so every work array (Ri, Hi) is written alike.
        bool sameNm = true;
        for (int withJacobian = 1; withJacobian >= 0; withJacobian--) {
            quad_compute(on, withJacobian == 1);
            exasim::nonmatching::compute_nonmatching_coupling<PoissonCoupling>(nm2, op.sol, op.res, op.app,
                    op.master, op.mesh, op.tmp, op.common, 0, withJacobian == 1, 0, quad2.get());
            sameNm = sameNm && same_interface(nm2, on.nm, tag + " transactional nm");
        }
        check(sameNm, tag + ": transactional rebuild arrays, residual and Jacobian equal the direct setup bitwise");
        exasim::nonmatching::compute_nonmatching_coupling<PoissonCoupling>(*rt2, op.sol, op.res, op.app,
                op.master, op.mesh, op.tmp, op.common, 0, false, 0);
        bool same = true;
        for (int k = 0; k < 2; k++)
            same = same && (on.nm.dir[k].szRi == 0 ||
                   std::memcmp(on.nm.dir[k].Ri, rt2->dir[k].Ri.data, sizeof(dstype) * on.nm.dir[k].szRi) == 0);
        check(same, tag + ": transactional rebuild runtime residual equals the direct setup bitwise");
        exasim::nonmatching::free_nonmatching_data(nm2);
        CPUFREE(common.requests);
        CPUFREE(common.statuses);
    }

    // T3: option off with the sidecar equals no sidecar, bitwise (setup, residual, Jacobian).
    if (!mixedFamilies) {
        QuadSetup bare;
        quad_setup(bare, sc, mm, mst, n, false, 64, false);
        quad_fill_state(bare, mst, sc.model);
        quad_fill_state(off, mst, sc.model);
        for (int withJacobian = 0; withJacobian < 2; withJacobian++) {
            quad_compute(bare, withJacobian == 1);
            quad_compute(off, withJacobian == 1);
            check(same_interface(off.nm, bare.nm, tag + " T3"), tag + ": option off is bitwise the path without the sidecar");
        }
    }
}

// T4 (matching), T7 (gapped, independently curved) and the coincidence threshold: option on
// must equal option off bitwise unless the interface is coincident with interior breakpoints.
void quadrature_fallback_case(MasterData& mst, int n)
{
    SubComm sc(n);
    if (sc.comm == MPI_COMM_NULL) return;
    const std::string tag = "quadrature fallback (" + std::to_string(sc.k) + " per model)";
    struct Variant { const char* name; std::vector<dstype> ys2; dstype shift, amp; int expectApplied, expectCoincident; };
    const std::vector<Variant> variants = {
        {"matching", quadYs1, 0, 0, 0, 1},
        {"gapped 1e-2", quadYs2, 1e-2, 0, 0, 0},
        {"independently curved", quadYs2, 0, 0.01, 0, 0},
        {"gap 1e-9 above onFaceRelTol*h_F", quadYs2, 1e-9, 0, 0, 0},
        {"gap 1e-13 below onFaceRelTol*h_F", quadYs2, 1e-13, 0, 1, 1},
    };
    for (const Variant& v : variants) {
        const ModelMesh mm = column_mesh(sc.model, sc.model == 0 ? quadYs1 : v.ys2, v.shift, v.amp);
        QuadSetup on, off, again;
        quad_setup(on, sc, mm, mst, n, true);
        quad_setup(off, sc, mm, mst, n, false);
        bool flags = true;
        for (int k = 0; k < 2; k++)
            flags = flags && on.quad.dir[k].applied == v.expectApplied && on.quad.dir[k].coincident == v.expectCoincident;
        check(flags, tag + ": " + v.name + ": applied/coincident flags");
        quad_fill_state(on, mst, sc.model);
        quad_fill_state(off, mst, sc.model);
        const std::string vt = tag + " " + v.name;
        if (!v.expectApplied) {
            bool same = same_interface(on.nm, off.nm, vt);
            for (int withJacobian = 0; withJacobian < 2; withJacobian++) {
                quad_compute(on, withJacobian == 1);
                quad_compute(off, withJacobian == 1);
                same = same && same_interface(on.nm, off.nm, vt);
            }
            check(same, tag + ": " + v.name + ": option on is bitwise the uniform path");
            // The sidecar of a fallback holds no composite data: only the request and the
            // coincidence flags differ from option off.
            bool empty = on.quad.requested == 1;
            for (int k = 0; k < 2; k++) {
                const auto& q = on.quad.dir[k];
                empty = empty && q.applied == 0 && q.breakpointsTotal == 0 && q.facebreakptr.empty() &&
                        q.breakpoints.empty() && q.ptest.empty() && q.ps.empty() && q.ppiece.empty();
            }
            check(empty, tag + ": " + v.name + ": fallback sidecar holds no composite data");
        }
        else {
            // Deterministic application: a second setup gives the same complete sidecar, arrays,
            // residual and Jacobian bitwise.
            quad_setup(again, sc, mm, mst, n, true);
            quad_fill_state(again, mst, sc.model);
            bool same = same_quad(on.quad, again.quad, vt) && same_interface(on.nm, again.nm, vt);
            for (int withJacobian = 0; withJacobian < 2; withJacobian++) {
                quad_compute(on, withJacobian == 1);
                quad_compute(again, withJacobian == 1);
                same = same && same_interface(on.nm, again.nm, vt);
            }
            check(same, tag + ": " + v.name + ": application is deterministic");
        }
        if (worldRank == 0)
            std::printf("%s: %s: applied %d/%d, coincident %d/%d\n", tag.c_str(), v.name,
                        static_cast<int>(on.quad.dir[0].applied), static_cast<int>(on.quad.dir[1].applied),
                        static_cast<int>(on.quad.dir[0].coincident), static_cast<int>(on.quad.dir[1].coincident));
    }
}

void quadrature_study()
{
    MasterData mst = make_master(1, 2);
    MasterData mixed3 = make_master(1, 3);
    for (int n = 2; n <= worldSize; n *= 2) {
        quadrature_fixture_case(mst, n);
        quadrature_fixture_case(mst, n, true, true);
        quadrature_fixture_case(mst, n, true, false);
        quadrature_fixture_case(mixed3, n, true, true);
        quadrature_fixture_case(mixed3, n, true, false);
        quadrature_fallback_case(mst, n);
    }
    // T2: breakpoints and quadrature tuples bitwise, pair sets equal, residual to 1e-14.
    if (worldRank == 0 && quadSignatures.size() > 1) {
        const QuadSignature& ref = quadSignatures[0];
        for (std::size_t i = 1; i < quadSignatures.size(); i++) {
            const QuadSignature& s = quadSignatures[i];
            check(s.breakpoints == ref.breakpoints, "quadrature T2: breakpoint lists are partition independent");
            check(s.points == ref.points, "quadrature T2: quadrature tuples are partition independent");
            check(s.pairs == ref.pairs, "quadrature T2: slot/pair keys are partition independent");
            dstype err = 0, scale = 0;
            bool shape = s.residual.size() == ref.residual.size();
            for (std::size_t r = 0; shape && r < ref.residual.size(); r++)
                for (std::size_t c = 2; c < ref.residual[r].size(); c++) {
                    err = std::max(err, std::abs(s.residual[r][c] - ref.residual[r][c]));
                    scale = std::max(scale, std::abs(ref.residual[r][c]));
                }
            check(shape && err <= 1e-14 * scale, "quadrature T2: owner residuals are partition independent");
            std::printf("quadrature T2: run %zu vs 1+1: owner residual max rel diff %.3e\n", i, scale > 0 ? err / scale : 0.0);
        }
    }
}

// Expected failures (separate MPI tests, PASS_REGULAR_EXPRESSION on the message).
//   0: option differs across ranks; 1: more pieces than maxCompositePiecesPerFace;
//   2 (Revision 2): endpoint candidate failure, forced by a zero projection-iteration limit
//      for the reverse endpoint search only (the forward search keeps its limit). A coincident
//      interface: the error must be the endpoint-search failure, not a noncoincident fallback.
void quadrature_expected_failure(int mode)
{
    MasterData mst = make_master(1, 2);
    SubComm sc(worldSize);
    if (sc.comm == MPI_COMM_NULL) return;
    const ModelMesh mm = column_mesh(sc.model, sc.model == 0 ? quadYs1 : quadYs2);
    QuadSetup s;
    if (mode == 0) quad_setup(s, sc, mm, mst, worldSize, worldRank == 0);
    else if (mode == 1) quad_setup(s, sc, mm, mst, worldSize, true, 2); // the face [0, 0.25] needs 3 pieces
    else quad_setup(s, sc, mm, mst, worldSize, true, 64, true, 0);
    std::printf("UNEXPECTED: setup returned without the expected error (applied %d/%d, coincident %d/%d)\n",
                static_cast<int>(s.quad.dir[0].applied), static_cast<int>(s.quad.dir[1].applied),
                static_cast<int>(s.quad.dir[0].coincident), static_cast<int>(s.quad.dir[1].coincident));
}

// A/S/R measurements at N = 8: setup time, residual and Jacobian time (runtime path, median of
// 20), points/slots/pairs, maximum points per owner face, host sidecar and runtime mirror bytes.
template <class V>
std::size_t runtime_buffer_bytes(const exasim::nonmatching::RuntimeBuffer<V, Int>& b)
{
    return static_cast<std::size_t>(b.capacity) * sizeof(V);
}

struct RuntimeMemory {
    std::size_t total = 0, ptest = 0, hi = 0, fpair = 0, temporary = 0;
};

RuntimeMemory runtime_memory(const exasim::nonmatching::NonmatchingRuntime<dstype, Int>& rt)
{
    RuntimeMemory out;
    for (int k = 0; k < 2; ++k) {
        const auto& d = rt.dir[k];
#define ADD_RUNTIME_BUFFER(name) out.total += runtime_buffer_bytes(d.name)
        ADD_RUNTIME_BUFFER(pe); ADD_RUNTIME_BUFFER(pa); ADD_RUNTIME_BUFFER(pslot); ADD_RUNTIME_BUFFER(pg);
        ADD_RUNTIME_BUFFER(slotptr); ADD_RUNTIME_BUFFER(pairslot); ADD_RUNTIME_BUFFER(pairelem);
        ADD_RUNTIME_BUFFER(pairptr); ADD_RUNTIME_BUFFER(recvface); ADD_RUNTIME_BUFFER(ownface);
        ADD_RUNTIME_BUFFER(pairSlotPtr); ADD_RUNTIME_BUFFER(ownerTarget); ADD_RUNTIME_BUFFER(faceBlockPtr);
        ADD_RUNTIME_BUFFER(faceBlocks); ADD_RUNTIME_BUFFER(px); ADD_RUNTIME_BUFFER(pnl); ADD_RUNTIME_BUFFER(pwJ);
        ADD_RUNTIME_BUFFER(pphi); ADD_RUNTIME_BUFFER(pphibar); ADD_RUNTIME_BUFFER(ppsi); ADD_RUNTIME_BUFFER(pd);
        ADD_RUNTIME_BUFFER(pdphibar); ADD_RUNTIME_BUFFER(ptest); ADD_RUNTIME_BUFFER(Ri); ADD_RUNTIME_BUFFER(Hi);
        ADD_RUNTIME_BUFFER(recvbuf); ADD_RUNTIME_BUFFER(x); ADD_RUNTIME_BUFFER(udg); ADD_RUNTIME_BUFFER(odg);
        ADD_RUNTIME_BUFFER(wdg); ADD_RUNTIME_BUFFER(uhg); ADD_RUNTIME_BUFFER(nl); ADD_RUNTIME_BUFFER(wsrc);
        ADD_RUNTIME_BUFFER(flux); ADD_RUNTIME_BUFFER(fluxUdg); ADD_RUNTIME_BUFFER(fluxWdg);
        ADD_RUNTIME_BUFFER(fluxUhg); ADD_RUNTIME_BUFFER(wdgUdg); ADD_RUNTIME_BUFFER(equationUdg);
        ADD_RUNTIME_BUFFER(wScratch); ADD_RUNTIME_BUFFER(iScratch); ADD_RUNTIME_BUFFER(K);
        ADD_RUNTIME_BUFFER(G); ADD_RUNTIME_BUFFER(Cpair); ADD_RUNTIME_BUFFER(Epair);
        ADD_RUNTIME_BUFFER(pairResidual); ADD_RUNTIME_BUFFER(vpair); ADD_RUNTIME_BUFFER(rpair);
        ADD_RUNTIME_BUFFER(Fpair); ADD_RUNTIME_BUFFER(Rupair);
#undef ADD_RUNTIME_BUFFER
        out.ptest += runtime_buffer_bytes(d.ptest);
        out.hi += runtime_buffer_bytes(d.Hi);
        out.fpair += runtime_buffer_bytes(d.Fpair);
    }
    out.temporary = out.total - out.ptest - out.hi - out.fpair;
    return out;
}

double median_time(std::vector<double>& values)
{
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

void mixed_runtime_cost_case(int n, int family, bool composite)
{
    SubComm sc(n);
    if (sc.comm == MPI_COMM_NULL) return;
    const bool mixed = family != 0;
    const bool model0Triangle = family == 1;
    MasterData triangle = make_master(0, 3), quad = make_master(1, 3);
    MasterData& mst = mixed ? (((sc.model == 0) == model0Triangle) ? triangle : quad) : quad;
    const int rows = (sc.model == 0 ? 4 : 6) * 8;
    ModelMesh mm;
    mm.ys = uniform(0.0, 1.0, rows);
    if (sc.model == 0) { mm.xs = uniform(0.0, 0.5, rows / 2); mm.xiface = 0.5; mm.xfar = 0.0; }
    else { mm.xs = uniform(0.5, 1.0, rows / 2); mm.xiface = 0.5; mm.xfar = 1.0; }
    mm.tri = mixed && ((sc.model == 0) == model0Triangle);

    MPI_Barrier(sc.comm);
    const double setupStart = MPI_Wtime();
    QuadSetup s;
    quad_setup(s, sc, mm, mst, n, composite);
    const double setupSeconds = MPI_Wtime() - setupStart;
    quad_fill_state(s, mst, sc.model);
    RankOperator& op = *s.op;
    exasim::nonmatching::NonmatchingRuntime<dstype, Int> rt;
    exasim::nonmatching::refresh_nonmatching_runtime(rt, s.nm, op.app, op.mesh, op.common, Int(0), &s.quad);
    const long long allocationsAfterRefresh = rt.allocations;

    std::vector<dstype> v(op.N), w(op.N, 0);
    for (Int i = 0; i < op.N; ++i) v[i] = hv(i, 911 + sc.model) - 0.5;
    auto timed = [&](auto&& fn) {
        std::vector<double> times;
        times.reserve(20);
        for (int it = 0; it < 20; ++it) {
            MPI_Barrier(sc.comm);
            const double start = MPI_Wtime();
            fn();
            times.push_back(MPI_Wtime() - start);
        }
        return median_time(times);
    };
    const double residualSeconds = timed([&]() {
        exasim::nonmatching::compute_nonmatching_coupling<PoissonCoupling>(rt, op.sol, op.res, op.app,
                op.master, op.mesh, op.tmp, op.common, 0, false, 0);
    });
    const double jacobianSeconds = timed([&]() {
        exasim::nonmatching::compute_nonmatching_coupling<PoissonCoupling>(rt, op.sol, op.res, op.app,
                op.master, op.mesh, op.tmp, op.common, 0, true, 0);
    });
    exasim::nonmatching::compute_nonmatching_coupling<PoissonCoupling>(rt, op.sol, op.res, op.app,
            op.master, op.mesh, op.tmp, op.common, 0, false, 0);
    const double mpiSeconds = timed([&]() {
        Int count = 0;
        exasim::nonmatching::begin_coupling_slot_exchange(rt, op.common, count);
        MPI_Waitall(count, op.common.requests, op.common.statuses);
    });
    const double matvecSeconds = timed([&]() {
        std::fill(w.begin(), w.end(), 0);
        hdgMatVec<dstype, Int>(w.data(), op.AE.data(), v.data(), op.ve.data(), op.we.data(), op.res,
                op.app, op.mesh, op.common, op.tmp, 0, 0, &rt);
    });
    check(rt.allocations == allocationsAfterRefresh,
          "mixed cost: residual, Jacobian, MPI exchange and matvec allocate no runtime buffers");

    RuntimeMemory memory = runtime_memory(rt);
    double local[13] = {setupSeconds, residualSeconds, jacobianSeconds, mpiSeconds, matvecSeconds,
                        0, 0, 0, static_cast<double>(memory.total), static_cast<double>(memory.ptest),
                        static_cast<double>(memory.hi), static_cast<double>(memory.fpair),
                        static_cast<double>(memory.temporary)};
    for (int k = 0; k < 2; ++k) {
        local[5] += s.nm.dir[k].npts;
        local[6] += s.nm.dir[k].npairs;
        local[7] += s.nm.dir[k].szRi * sizeof(dstype);
    }
    double maxima[5], sums[8];
    MPI_Allreduce(local, maxima, 5, MPI_DOUBLE, MPI_MAX, sc.comm);
    MPI_Allreduce(local + 5, sums, 8, MPI_DOUBLE, MPI_SUM, sc.comm);
    if (worldRank == 0) {
        const char* label = family == 0 ? "quad-quad" : (family == 1 ? "tri-quad" : "quad-tri");
        std::printf("MIXED COST %s A1=%d p=3 %d+%d: setup %.3f ms, residual %.3f ms, Jacobian %.3f ms, "
                    "MPI %.3f ms, matvec %.3f ms; points %.0f, pairs %.0f, message %.0f B; runtime %.0f B "
                    "(%.1f B/point, %.1f B/pair), ptest %.0f B, Hi %.0f B, Fpair %.0f B, other %.0f B\n",
                    label, composite ? 1 : 0, sc.k, sc.k, 1e3 * maxima[0], 1e3 * maxima[1],
                    1e3 * maxima[2], 1e3 * maxima[3], 1e3 * maxima[4], sums[0], sums[1], sums[2], sums[3],
                    sums[0] > 0 ? sums[3] / sums[0] : 0, sums[1] > 0 ? sums[3] / sums[1] : 0,
                    sums[4], sums[5], sums[6], sums[7]);
    }
}

void quadrature_cost_study()
{
    struct Cfg { const char* name; int r1, r2; } cfgs[] = {{"A", 4, 6}, {"S", 6, 4}, {"R", 4, 8}};
    const int N = 8;
    for (int p = 1; p <= 4; p++) {
        MasterData mst = make_master(1, p);
        for (const Cfg& cfg : cfgs)
            for (int n = 2; n <= std::min(worldSize, 4); n *= 2) {
                SubComm sc(n);
                if (sc.comm == MPI_COMM_NULL) continue;
                const int rows = (sc.model == 0 ? cfg.r1 : cfg.r2) * N;
                ModelMesh mm;
                mm.ys = uniform(0.0, 1.0, rows);
                if (sc.model == 0) { mm.xs = uniform(0.0, 0.5, rows / 2); mm.xiface = 0.5; mm.xfar = 0.0; }
                else { mm.xs = uniform(0.5, 1.0, rows / 2); mm.xiface = 0.5; mm.xfar = 1.0; }
                double stats[2][8] = {{0}};
                for (int option = 0; option < 2; option++) {
                    QuadSetup s;
                    MPI_Barrier(sc.comm);
                    const double t0 = MPI_Wtime();
                    quad_setup(s, sc, mm, mst, n, option == 1);
                    const double tSetup = MPI_Wtime() - t0;
                    quad_fill_state(s, mst, sc.model);
                    exasim::nonmatching::NonmatchingRuntime<dstype, Int> rt;
                    RankOperator& op = *s.op;
                    exasim::nonmatching::refresh_nonmatching_runtime(rt, s.nm, op.app, op.mesh, op.common, Int(0), &s.quad);
                    double tm[2];
                    for (int jac = 0; jac < 2; jac++) {
                        std::vector<double> times;
                        for (int it = 0; it < 20; it++) {
                            const double a = MPI_Wtime();
                            exasim::nonmatching::compute_nonmatching_coupling<PoissonCoupling>(rt, op.sol, op.res, op.app,
                                    op.master, op.mesh, op.tmp, op.common, 0, jac == 1, 0);
                            times.push_back(MPI_Wtime() - a);
                        }
                        std::sort(times.begin(), times.end());
                        tm[jac] = times[times.size() / 2];
                    }
                    double local[8] = {tSetup, tm[0], tm[1], 0, 0, 0,
                                       static_cast<double>(s.quad.host_bytes()), 0};
                    for (int k = 0; k < 2; k++) {
                        local[3] += s.nm.dir[k].npts; local[4] += s.nm.dir[k].nslots; local[5] += s.nm.dir[k].npairs;
                        local[7] += static_cast<double>(rt.dir[k].ptest.capacity) * sizeof(dstype);
                    }
                    double gmax[3], gsum[5];
                    MPI_Allreduce(local, gmax, 3, MPI_DOUBLE, MPI_MAX, sc.comm);
                    MPI_Allreduce(local + 3, gsum, 5, MPI_DOUBLE, MPI_SUM, sc.comm);
                    for (int i = 0; i < 3; i++) stats[option][i] = gmax[i];
                    for (int i = 0; i < 5; i++) stats[option][3 + i] = gsum[i];
                    if (option == 1 && worldRank == 0)
                        std::printf("COST %s p=%d %d+%d: applied %d/%d, max pieces per owner face %d/%d (max points %d/%d)\n",
                                    cfg.name, p, sc.k, sc.k, static_cast<int>(s.quad.dir[0].applied),
                                    static_cast<int>(s.quad.dir[1].applied),
                                    static_cast<int>(s.quad.dir[0].maxPiecesPerFace), static_cast<int>(s.quad.dir[1].maxPiecesPerFace),
                                    static_cast<int>(s.quad.dir[0].maxPiecesPerFace * mst.ngf),
                                    static_cast<int>(s.quad.dir[1].maxPiecesPerFace * mst.ngf));
                }
                if (worldRank == 0) {
                    // Existing host per-point setup storage (frozen-header point arrays), for the ratio.
                    const double perPoint = sizeof(dstype) * (4.0 * mst.nd + 1 + (mst.nd - 1) + 2.0 * mst.npe + mst.npf) +
                                            sizeof(Int) * 5.0;
                    std::printf("COST %s p=%d %d+%d: points %.0f -> %.0f (x%.2f), slots %.0f -> %.0f, pairs %.0f -> %.0f; "
                                "setup %.3f -> %.3f ms (x%.2f); residual %.3f -> %.3f ms (x%.2f); Jacobian %.3f -> %.3f ms (x%.2f); "
                                "host sidecar %.0f B (%.3f of existing point storage), runtime ptest mirror %.0f B\n",
                                cfg.name, p, sc.k, sc.k, stats[0][3], stats[1][3], stats[1][3] / stats[0][3],
                                stats[0][4], stats[1][4], stats[0][5], stats[1][5],
                                1e3 * stats[0][0], 1e3 * stats[1][0], stats[1][0] / stats[0][0],
                                1e3 * stats[0][1], 1e3 * stats[1][1], stats[1][1] / stats[0][1],
                                1e3 * stats[0][2], 1e3 * stats[1][2], stats[1][2] / stats[0][2],
                                stats[1][6], stats[1][6] / (perPoint * stats[1][3]), stats[1][7]);
                }
            }
    }
    for (int family = 0; family < 3; ++family)
        for (int composite = 0; composite < 2; ++composite)
            mixed_runtime_cost_case(worldSize, family, composite != 0);
}

} // namespace

int main(int argc, char** argv)
{
    // Negative selection tests fork before MPI_Init, on the first process only.
    // [nonmatching] Phase 9: --b1-transfer runs only the B1 transfer-consistency study;
    // --quadrature* run only the A1 composite owner-quadrature modes.
    bool b1Only = false;
    std::string quadMode;
    for (int i = 1; i < argc; i++) {
        const std::string arg(argv[i]);
        if (arg == "--b1-transfer" || arg.rfind("--quadrature", 0) == 0) {
            if (arg == "--b1-transfer") b1Only = true;
            else quadMode = arg;
            for (int k = i; k + 1 < argc; k++) argv[k] = argv[k + 1];
            argc--;
            break;
        }
    }
    const bool special = b1Only || !quadMode.empty();
    bool firstProcess = true;
    for (const char* var : {"OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK"})
        if (const char* r = std::getenv(var)) { firstProcess = std::atoi(r) == 0; break; }
    if (firstProcess && !special) run_negative_cases();
    MPI_Init(&argc, &argv);
    EXASIM_COMM_WORLD = MPI_COMM_WORLD;
    EXASIM_COMM_LOCAL = MPI_COMM_WORLD;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (worldSize < 2 || (worldSize & (worldSize - 1)) != 0) {
        if (worldRank == 0) std::printf("nonmatching_matvec needs 2, 4, 8, ... ranks\n");
        MPI_Finalize();
        return 1;
    }
    Kokkos::initialize(argc, argv);
    if (b1Only) {
        b1_transfer_study();
    }
    else if (quadMode == "--quadrature") {
        quadrature_study();
    }
    else if (quadMode == "--quadrature-mismatch" || quadMode == "--quadrature-piece-limit" ||
             quadMode == "--quadrature-endpoint-failure") {
        quadrature_expected_failure(quadMode == "--quadrature-mismatch" ? 0 :
                                    (quadMode == "--quadrature-piece-limit" ? 1 : 2));
    }
    else if (quadMode == "--quadrature-cost") {
        quadrature_cost_study();
    }
    else {
        selection_case();

        MasterData quad2 = make_master(1, 2);
        MasterData mixed3 = make_master(1, 3); // porder carrier; each mixed rank builds its local family
        MasterData mixed4 = make_master(1, 4);
        // Non-matching curved quads (d != 0, |K_l| > 1, misaligned row strips).
        ModelMesh c1, c2;
        c1.xs = uniform(0.0, 0.5, 2); c1.ys = uniform(0.0, 1.0, 6); c1.xiface = 0.5; c1.xfar = 0.0;
        c2.xs = uniform(0.5, 1.0, 2); c2.ys = {0.0, 0.11, 0.27, 0.38, 0.55, 0.71, 0.82, 0.9, 1.0};
        c2.xiface = 0.5; c2.xfar = 1.0; c2.amp = 0.01;
        // Matching straight quads.
        ModelMesh s1, s2;
        s1.xs = uniform(0.0, 0.5, 2); s1.ys = uniform(0.0, 1.0, 4); s1.xiface = 0.5; s1.xfar = 0.0;
        s2.xs = uniform(0.5, 1.0, 2); s2.ys = uniform(0.0, 1.0, 4); s2.xiface = 0.5; s2.xfar = 1.0;

        ModelMesh tq1 = c1, tq2 = c2;
        tq1.tri = true; tq2.tri = false;
        ModelMesh qt1 = c1, qt2 = c2;
        qt1.tri = false; qt2.tri = true;

        for (int n = 2; n <= worldSize; n *= 2) {
            nonmatching_case<NLCoupled>("nonmatching NL", c1, c2, quad2, n, true);
            nonmatching_case<WCoupled>("nonmatching ncw=2", c1, c2, quad2, n, true);
            matching_case<NLCoupled>("matching NL", s1, s2, quad2, n);
            halo_case<NLCoupled>("halo NL", c1, c2, quad2, n);
            rebuild_case<NLCoupled>("rebuild NL", c1, c2, quad2, n);
            nonmatching_case<NLCoupled>("mixed tri-quad NL", tq1, tq2, mixed3, n,
                    true, true, true);
            nonmatching_case<NLCoupled>("mixed quad-tri NL", qt1, qt2, mixed3, n,
                    true, true, false);
            nonmatching_case<QFreeCoupled>("mixed tri-quad q-free", tq1, tq2, mixed3, n,
                    true, true, true);
            nonmatching_case<QFreeCoupled>("mixed quad-tri q-free", qt1, qt2, mixed3, n,
                    true, true, false);
            nonmatching_case<WCoupled>("mixed tri-quad ncw=2", tq1, tq2, mixed3, n,
                    false, true, true);
            nonmatching_case<NLCoupled>("mixed tri-quad p=4", tq1, tq2, mixed4, n,
                    true, true, true);
            nonmatching_case<NLCoupled>("mixed quad-tri p=4", qt1, qt2, mixed4, n,
                    true, true, false);
            rebuild_case<NLCoupled>("mixed rebuild NL", tq1, tq2, mixed3, n,
                    true, true);
        }
    }
    Kokkos::finalize();

    int total = 0;
    MPI_Allreduce(&failures, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    dstype roundoff = 0, fd = 0;
    MPI_Allreduce(&maxRoundoff, &roundoff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&maxFdError, &fd, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (worldRank == 0)
        std::printf("%s nonmatching_matvec%s (%d ranks, up to %d per model), max round-off %.3e, max FD error %.3e\n",
                    total == 0 ? "PASS" : "FAIL", b1Only ? " --b1-transfer" : (quadMode.empty() ? "" : (" " + quadMode).c_str()), worldSize, worldSize / 2, roundoff, fd);
    MPI_Finalize();
    return total == 0 ? 0 : 1;
}
