// Phase 2 of the non-matching interface solver: donor search
// (backend/Discretization/nonmatchingsetup.hpp, docs/nonmatching/nonmatching_multiphysics.tex,
// eqs. (3), (56), (60), Section 3.2).
//
// MPI test, run with 2^j ranks. For every n = 2, 4, ..., worldSize, the first n
// world ranks form EXASIM_COMM_WORLD of one run: ranks [0, n/2) hold model 1 and
// [n/2, n) model 2 (fileoffset = n/2), as in ExasimSolver.cpp. Each rank builds its
// partition of synthetic 2D meshes as backend structs and calls the real entry points.
// On world rank 0, each partition's donor tuples must equal a serial reference
// bitwise, and the rank-independent part (owner face, g, f, a, dist, xi) must be
// identical for all partitions of the run. The serial reference is checked against an
// exhaustive, independently coded evaluation. Registered with 2, 4 and 8 ranks, so
// the np8 test compares 1, 2 and 4 ranks per model directly; a rank-independent
// checksum is printed for each partition.
//
// Cases:
//   conformal: interfacetype = 0 on the matching mesh; build_runtime_interface_partition
//      builds nbintf, facesend/recv(pts), faceperm exactly as expected; the non-matching
//      setup does nothing (review F3).
//   matching: geometry of examples/CoupledProblems/Poisson/matching (two 8x16 quad meshes of
//      [0,0.5]x[0,1] and [0.5,1]x[0,1], p = 3). Donor = conformal partner, dist < 1e-12.
//   nonmatching: triangles, p = 2, 12 vs 7 faces with non-uniform rows, curved and
//      perturbed model-2 interface (gaps and overlaps), model 2 shorter than model 1.
//   tie (mesh): model-2 vertices at the heights of model-1 Gauss points, offset by a gap;
//      equidistant points go to the smaller global index on every partition.
//   near ties (serial): a lower-index face farther by 1e-9 .. 1e-13 is never selected;
//      exact mirror and shared-vertex ties go to the smaller index (review F2).
//   curved adversarial (serial): a p = 4 face whose image leaves its nodal box, closer
//      than a straight face that is closer than the box; the curved face wins (review F1).
//   tie thresholds (serial): the farther face has the smaller index and is farther by
//      0.5, 0.85, 1.15 and 2 times tau = (d_L - lower_L) + (upper_H - d_H) from the
//      certified bounds and analytic reference distances; straight faces (p = 1, 4) and
//      a curved parabola face, farther or closer (p = 2, 4) (reviews F2-R1, F2-R2).
//   safeguard stagnation (serial): no resolvable descent far from a stationary point is
//      reported as non-converged (review F1-R1).
//   failed candidate (serial): a far, degenerate face fails the point (review F4-R1).
//   basis accuracy (serial): production face basis against an analytic Lagrange
//      evaluation, per value and weighted with adversarial coordinates (p = 1..4, both
//      element types); production map = Bernstein form (review F2-R2).

#ifdef HAVE_MPI
#include <mpi.h>
MPI_Comm EXASIM_COMM_WORLD = MPI_COMM_NULL;
MPI_Comm EXASIM_COMM_LOCAL = MPI_COMM_NULL;
#endif

#include <fstream>
#include <iostream>

#include <Kokkos_Core.hpp>
#include "../../backend/Discretization/nonmatchinggeometry.hpp"
#include "../../backend/Common/cpuimpl.h"
#include "../../backend/Discretization/interfacepartition.hpp"
#include "../../backend/Discretization/nonmatchingsetup.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <tuple>
#include <vector>

using namespace exasim::nonmatching;

namespace {

int failures = 0;
int worldRank = 0, worldSize = 1;

void check(bool ok, const std::string& what)
{
    if (!ok) {
        std::printf("[rank %d] FAIL %s\n", worldRank, what.c_str());
        ++failures;
    }
}

std::string data_path(const std::string& file)
{
    const std::string source = __FILE__;
    const std::string marker = "/tests/unit/nonmatching_donorsearch.cpp";
    return source.substr(0, source.rfind(marker)) + "/backend/Preprocessing/" + file;
}

// Master data shared by both models (one master element, phase 0 decision).
struct MasterData {
    int nd = 2, elemtype = 1, porder = 1, npe = 0, npf = 0, nfe = 0, ngf = 0;
    std::vector<dstype> xpe, xpf, gpf, gwf, shapfgt;
    std::vector<int> perm;
    std::vector<Int> ndims;
};

MasterData make_master(int elemtype, int porder)
{
    MasterData m;
    m.elemtype = elemtype;
    m.porder = porder;
    std::vector<int> telem, tface;
    masternodes(m.xpe, telem, m.xpf, tface, m.perm, porder, m.nd, elemtype, data_path("masternodes.bin"));
    gaussnodes(m.gpf, m.gwf, 2 * porder, m.nd - 1, elemtype, data_path("gaussnodes.bin"));
    m.npe = static_cast<int>(m.xpe.size()) / m.nd;
    m.npf = static_cast<int>(m.xpf.size()) / (m.nd - 1);
    m.nfe = static_cast<int>(m.perm.size()) / m.npf;
    m.ngf = static_cast<int>(m.gwf.size());
    m.shapfgt.assign(static_cast<std::size_t>(m.nd * m.ngf * m.npf), 0.0);
    std::vector<dstype> psi(m.npf), dpsi(m.npf * (m.nd - 1));
    for (int g = 0; g < m.ngf; g++) {
        dstype xi[2] = {m.gpf[g], 0.0};
        EvaluateFaceShapeAndGradients(psi.data(), dpsi.data(), xi, m.xpf.data(),
                m.nd, m.npf, elemtype, porder);
        for (int j = 0; j < m.npf; j++) {
            m.shapfgt[g + m.ngf * j] = psi[j];
            m.shapfgt[m.ngf * m.npf + g + m.ngf * j] = dpsi[j];
        }
    }
    m.ndims.assign(20, 0);
    m.ndims[0] = m.nd; m.ndims[1] = elemtype; m.ndims[3] = porder; m.ndims[4] = 2 * porder;
    m.ndims[5] = m.npe; m.ndims[6] = m.npf; m.ndims[8] = m.ngf;
    return m;
}

// Structured 2D mesh of one model. Logical cells [xs[i],xs[i+1]] x [ys[j],ys[j+1]]
// (quads, or two triangles per cell), elements numbered row by row. Physical map:
// x = s + amp*sin(2*pi*(t-ys0)/(ysN-ys0)) * w(s), y = t, w = 1 on the interface column,
// 0 on the far column, linear in between. The interface is the column s = xiface.
struct ModelMesh {
    std::vector<dstype> xs, ys;
    bool tri = false;
    dstype amp = 0.0;
    dstype xiface = 0.0, xfar = 0.0;
    int bcInterface = 2;

    int nx() const { return static_cast<int>(xs.size()) - 1; }
    int ny() const { return static_cast<int>(ys.size()) - 1; }
    int ne() const { return nx() * ny() * (tri ? 2 : 1); }
    int row(int ge) const { return ge / (nx() * (tri ? 2 : 1)); }

    void physical(dstype* x, dstype s, dstype t) const
    {
        const dstype pi = 3.14159265358979323846;
        const dstype w = (s - xfar) / (xiface - xfar);
        x[0] = s + amp * std::sin(2.0 * pi * (t - ys.front()) / (ys.back() - ys.front())) * w;
        x[1] = t;
    }

    // Logical coordinates of reference point zeta in global element ge.
    void logical(dstype& s, dstype& t, int ge, const dstype* zeta) const
    {
        const int per = tri ? 2 : 1;
        const int cell = ge / per, k = ge % per;
        const int i = cell % nx(), j = cell / nx();
        const dstype x0 = xs[i], x1 = xs[i + 1], y0 = ys[j], y1 = ys[j + 1];
        const dstype r = zeta[0], q = zeta[1];
        if (!tri) { s = x0 + r * (x1 - x0); t = y0 + q * (y1 - y0); return; }
        // T0: (x0,y0),(x1,y0),(x1,y1); T1: (x0,y0),(x1,y1),(x0,y1)
        if (k == 0) { s = x0 + r * (x1 - x0) + q * (x1 - x0); t = y0 + q * (y1 - y0); }
        else        { s = x0 + r * (x1 - x0); t = y0 + r * (y1 - y0) + q * (y1 - y0); }
    }
};

// Element nodes (npe x nd) and interface flags (nfe) of global element ge.
void element_data(std::vector<dstype>& xe, std::vector<int>& bfe, const ModelMesh& mm,
                  const MasterData& m, int ge)
{
    xe.assign(static_cast<std::size_t>(m.npe * m.nd), 0.0);
    std::vector<dstype> sl(m.npe);
    for (int n = 0; n < m.npe; n++) {
        const dstype zeta[2] = {m.xpe[n], m.xpe[n + m.npe]};
        dstype s, t, x[2];
        mm.logical(s, t, ge, zeta);
        sl[n] = s;
        mm.physical(x, s, t);
        xe[n] = x[0];
        xe[n + m.npe] = x[1];
    }
    bfe.assign(m.nfe, 1);
    const dstype tol = 1e-12;
    for (int lf = 0; lf < m.nfe; lf++) {
        bool on = true;
        for (int j = 0; j < m.npf; j++)
            if (std::abs(sl[m.perm[j + m.npf * lf]] - mm.xiface) > tol) on = false;
        if (on) bfe[lf] = mm.bcInterface;
    }
}

// Backend structs of one rank. Pointers refer to the vectors below (the structs
// have no destructors).
struct RankStructs {
    std::vector<Int> problem, appnsize, appndims, meshndims, bf, perm, elempart;
    std::vector<dstype> xdg;
    appstructT<dstype, Int> app;
    masterstructT<dstype, Int> master;
    meshstructT<dstype, Int> mesh;
    solstructT<dstype, Int> sol;
};

void build_rank_structs(RankStructs& rs, const ModelMesh& mm, MasterData& m, int part, int nparts,
                        int interfacetype)
{
    rs.problem.assign(35, 0);
    rs.problem[28] = 1; rs.problem[29] = 1; rs.problem[30] = mm.bcInterface;
    rs.problem[34] = interfacetype;
    rs.appnsize.assign(30, 0);
    rs.appnsize[2] = 35;   // length of app.problem
    rs.appnsize[14] = 1;   // szinterfacefluxmap
    rs.appndims.assign(30, 0);
    rs.appndims[AppNdims::nd] = m.nd;
    rs.appndims[AppNdims::nc] = 1;
    rs.appndims[AppNdims::ncu] = 1;
    rs.appndims[AppNdims::ncq] = 0;
    rs.appndims[AppNdims::ncx] = m.nd;

    rs.elempart.clear();
    rs.bf.clear();
    rs.xdg.clear();
    std::vector<dstype> xe;
    std::vector<int> bfe;
    for (int ge = 0; ge < mm.ne(); ge++) {
        if ((mm.row(ge) * nparts) / mm.ny() != part) continue; // row strips
        rs.elempart.push_back(ge);
        element_data(xe, bfe, mm, m, ge);
        rs.xdg.insert(rs.xdg.end(), xe.begin(), xe.end());
        rs.bf.insert(rs.bf.end(), bfe.begin(), bfe.end());
    }
    const int ne = static_cast<int>(rs.elempart.size());
    rs.meshndims.assign(20, 0);
    rs.meshndims[0] = m.nd; rs.meshndims[1] = ne; rs.meshndims[4] = m.nfe;
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
    rs.mesh.szelempart = ne;
    rs.sol.xdg = rs.xdg.data();
    // ne > 0 needs non-null pointers for the guards; an empty rank still takes part.
    static Int dummyI = 0;
    static dstype dummyD = 0;
    if (ne == 0) { rs.mesh.bf = &dummyI; rs.mesh.elempart = &dummyI; rs.sol.xdg = &dummyD; }
}

// Global interface faces of one model (serial reference), with the owning world rank.
InterfaceFaceSet<dstype, Int> global_faces(const ModelMesh& mm, const MasterData& m, int model,
                                           int nparts)
{
    InterfaceFaceSet<dstype, Int> fs;
    fs.nd = m.nd;
    fs.npf = m.npf;
    std::vector<int> localIndex(mm.ne(), -1), count(nparts, 0);
    for (int ge = 0; ge < mm.ne(); ge++) {
        const int part = (mm.row(ge) * nparts) / mm.ny();
        localIndex[ge] = count[part]++;
    }
    std::vector<dstype> xe;
    std::vector<int> bfe;
    for (int ge = 0; ge < mm.ne(); ge++) {
        element_data(xe, bfe, mm, m, ge);
        const int part = (mm.row(ge) * nparts) / mm.ny();
        for (int lf = 0; lf < m.nfe; lf++) {
            if (bfe[lf] != mm.bcInterface) continue;
            fs.rank.push_back(part + model * nparts);
            fs.model.push_back(model);
            fs.elem.push_back(localIndex[ge]);
            fs.lface.push_back(lf);
            fs.global.push_back(m.nfe * ge + lf);
            for (int d = 0; d < m.nd; d++)
                for (int j = 0; j < m.npf; j++)
                    fs.nodes.push_back(xe[m.perm[j + m.npf * lf] + m.npe * d]);
        }
    }
    return fs;
}

// Packed tuple of one point: direction, owner global face, g, r, e, f, a, dist, xi.
const int NPACK = 9;
using Key = std::pair<long, long>; // (direction*2^40 + owner global face, g)

void pack(std::vector<double>& out, const DonorSearchResult<dstype, Int>& res)
{
    for (int l = 0; l < res.nfaces; l++)
        for (int g = 0; g < res.ngf; g++) {
            const int p = g + res.ngf * l;
            const double v[NPACK] = {double(res.direction), double(res.ownglobal[l]), double(g),
                double(res.r[p]), double(res.e[p]), double(res.f[p]), double(res.a[p]),
                double(res.dist[p]), double(res.xi[p])};
            out.insert(out.end(), v, v + NPACK);
        }
}

std::map<Key, std::vector<double>> to_map(const std::vector<double>& packed)
{
    std::map<Key, std::vector<double>> m;
    for (std::size_t i = 0; i + NPACK <= packed.size(); i += NPACK) {
        const Key k((long(packed[i]) << 40) + long(packed[i + 1]), long(packed[i + 2]));
        m[k] = std::vector<double>(packed.begin() + i, packed.begin() + i + NPACK);
    }
    return m;
}

// Rank-independent checksum (FNV-1a over the bits of the tuples in key order).
unsigned long long checksum(const std::map<Key, std::vector<double>>& m)
{
    unsigned long long h = 1469598103934665603ULL;
    for (const auto& kv : m)
        for (double v : kv.second) {
            unsigned long long bits;
            std::memcpy(&bits, &v, sizeof(bits));
            for (int b = 0; b < 8; b++) { h ^= (bits >> (8 * b)) & 0xffULL; h *= 1099511628211ULL; }
        }
    return h;
}

// Owner-side consistency on this rank: recvnb increasing, sum(recvcount) = sum |R_l|,
// per donor the faces in increasing global index, each face listed under every r in R_l.
void check_owner_side(const nonmatchinginterfaceT<dstype, Int>& d, const DonorSearchResult<dstype, Int>& res,
                      const std::string& name)
{
    check(d.nownfaces == res.nfaces, name + ": nownfaces");
    long sumR = 0;
    for (int l = 0; l < res.nfaces; l++) {
        Int nK = 0, nR = 0;
        face_set_sizes(nK, nR, res, l);
        sumR += nR;
        check(d.ownface[l] == l && d.ownfaceglobal[l] == res.ownglobal[l], name + ": ownface");
    }
    long sumc = 0, off = 0;
    for (int n = 0; n < d.nrecvnb; n++) {
        if (n > 0) check(d.recvnb[n] > d.recvnb[n - 1], name + ": recvnb increasing");
        for (int c = 0; c < d.recvcount[n]; c++) {
            const int l = d.recvface[off + c];
            if (c > 0) check(d.ownfaceglobal[d.recvface[off + c - 1]] < d.ownfaceglobal[l], name + ": list order");
            bool has = false;
            for (int g = 0; g < res.ngf; g++) has = has || res.r[g + res.ngf * l] == d.recvnb[n];
            check(has, name + ": recvface donor");
        }
        off += d.recvcount[n];
        sumc += d.recvcount[n];
    }
    check(sumc == sumR, name + ": sum recvcount = sum |R_l|");
}

// Independent check of a donor search result against an exhaustive evaluation:
// every candidate projected (phase 1) and bounded by face_distance_bounds; with
// U = min upper_m, the donor must be the smallest global index among
// {m : lower_m <= U}. Each lower bound must not exceed a dense sampling of the
// face (2001 points, an independent upper estimate of d*_m). xi must lie in F_hat
// with |x - G(xi)| = dist, and every projection must have converged.
void check_against_exhaustive(const DonorSearchResult<dstype, Int>& sr, const InterfaceFaceSet<dstype, Int>& cand,
                              const MasterData& m, const std::string& name)
{
    DonorSearchParams prm;
    const auto bm = make_bernstein_master(m.xpf.data(), Int(m.porder));
    std::vector<dstype> psi(m.npf);
    std::vector<std::vector<dstype>> dense(2001, std::vector<dstype>(m.npf));
    for (int i = 0; i <= 2000; i++) {
        const dstype t[2] = {dstype(i) / 2000, 0.0};
        EvaluateFaceShape(dense[i].data(), t, m.xpf.data(), m.nd, m.npf, m.elemtype, m.porder);
    }
    for (int p = 0; p < sr.npts; p++) {
        const dstype xp[2] = {sr.x[p], sr.x[p + sr.npts]};
        std::vector<dstype> lo(cand.size()), up(cand.size());
        dstype U = 1e300;
        for (int c = 0; c < cand.size(); c++) {
            const dstype* y = &cand.nodes[m.npf * m.nd * c];
            dstype xi[2], xb[2], dd;
            const bool ok = ClosestPointOnFace(xi, xb, &dd, xp, y, m.xpf.data(),
                               m.nd, m.npf, m.elemtype, m.porder, prm.maxProjectionIterations, prm.projectionTolerance);
            const auto cb = face_distance_bounds(bm, y, Int(m.npf), xp, xi[0]);
            check(ok && cb.ok, name + ": exhaustive projection converged with valid bounds");
            lo[c] = cb.lower;
            up[c] = cb.upper;
            U = std::min(U, up[c]);
            dstype dmin = 1e300;
            for (int i = 0; i <= 2000; i++) {
                dstype g0 = 0, g1 = 0;
                for (int j = 0; j < m.npf; j++) { g0 += dense[i][j] * y[j]; g1 += dense[i][j] * y[j + m.npf]; }
                dmin = std::min(dmin, std::hypot(g0 - xp[0], g1 - xp[1]));
            }
            check(cb.lower <= dmin + 1e-14 && cb.lower <= dd + 1e-14, name + ": lower bound below sampled distance");
        }
        Int fbest = -1;
        int cbest = -1;
        for (int c = 0; c < cand.size(); c++)
            if (lo[c] <= U && (fbest < 0 || cand.global[c] < fbest)) { fbest = cand.global[c]; cbest = c; }
        check(sr.f[p] == fbest, name + ": exhaustive donor face");
        check(sr.a[p] == cand.lface[cbest] && sr.e[p] == cand.elem[cbest] && sr.r[p] == cand.rank[cbest],
              name + ": donor (r, e, a)");
        check(sr.converged[p] == 1, name + ": projection converged");
        const dstype xi[2] = {sr.xi[p], 0.0};
        check(xi[0] >= 0.0 && xi[0] <= 1.0, name + ": xi in reference face");
        dstype xb[2], tg[2];
        std::vector<dstype> psi(m.npf), dpsi(m.npf);
        MapReferenceFaceToPhysical(xb, tg, psi.data(), dpsi.data(), xi, &cand.nodes[m.npf * m.nd * cbest],
                                   m.xpf.data(), m.nd, m.npf, m.elemtype, m.porder);
        check(std::abs(std::hypot(xp[0] - xb[0], xp[1] - xb[1]) - sr.dist[p]) <= 1e-14,
              name + ": |x - G(xi)| = dist");
    }
}

// Global per-rank ordinal of each face in a global face list (order of collect_own_interface_faces).
std::vector<int> rank_ordinals(const InterfaceFaceSet<dstype, Int>& fs)
{
    std::map<Int, int> next;
    std::vector<int> ord(fs.size());
    for (int i = 0; i < fs.size(); i++) ord[i] = next[fs.rank[i]]++;
    return ord;
}

// Partner of face l of 'own' in 'other' by exact matching of the face end points (2D, y range).
int partner_index(const InterfaceFaceSet<dstype, Int>& own, int l, const InterfaceFaceSet<dstype, Int>& other,
                  const MasterData& m)
{
    auto ends = [&](const InterfaceFaceSet<dstype, Int>& fs, int f) {
        dstype lo = 1e300, hi = -1e300;
        for (int j = 0; j < m.npf; j++) {
            lo = std::min(lo, fs.nodes[j + m.npf + m.npf * m.nd * f]);
            hi = std::max(hi, fs.nodes[j + m.npf + m.npf * m.nd * f]);
        }
        return std::make_pair(lo, hi);
    };
    const auto e = ends(own, l);
    int found = -1;
    for (int c = 0; c < other.size(); c++) {
        const auto o = ends(other, c);
        if (std::abs(o.first - e.first) < 1e-14 && std::abs(o.second - e.second) < 1e-14) found = c;
    }
    return found;
}

// One partition: the first n world ranks, n/2 per model. Every rank of the world
// calls this; ranks >= n take no part. Returns the distributed tuples on world rank 0.
std::map<Key, std::vector<double>> run_partition(const std::string& name, const ModelMesh& m1,
        const ModelMesh& m2, MasterData& m, int n)
{
    MPI_Comm sub;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < n ? 0 : MPI_UNDEFINED, worldRank, &sub);
    std::vector<double> all;
    if (sub != MPI_COMM_NULL) {
        EXASIM_COMM_WORLD = sub; // the code under test communicates on EXASIM_COMM_WORLD
        const int k = n / 2;
        const int model = (worldRank < k) ? 0 : 1;
        const int part = worldRank - model * k;
        const int fileoffset = model * k;
        const std::string tag = name + " (" + std::to_string(k) + " per model)";

        RankStructs rs;
        build_rank_structs(rs, model == 0 ? m1 : m2, m, part, k, 1);

        // The conformal builder must return before its exact match when interfacetype = 1.
        exasim::interfacepartition::build_runtime_interface_partition(rs.app, rs.master, rs.mesh, rs.sol,
                static_cast<Int>(n), static_cast<Int>(worldRank), static_cast<Int>(fileoffset));
        check(rs.mesh.faceperm == nullptr && rs.mesh.nbintf == nullptr, tag + ": conformal arrays untouched");

        nonmatchingdataT<dstype, Int> nm;
        DonorSearchResult<dstype, Int> res;
        const bool ran = setup_nonmatching_interface(rs.app, rs.master, rs.mesh, rs.sol,
                static_cast<Int>(n), static_cast<Int>(worldRank), static_cast<Int>(fileoffset), nm, &res,
                DonorSearchParams(), false); // Isolate the Phase 2 tuple/checksum contract; Phase 3 separately
                                             // runs full setup on these geometries and its own matching/tie data.
        check(ran && nm.interfacetype == 1, tag + ": setup ran");
        check(res.direction == model, tag + ": direction");
        check_owner_side(nm.dir[model], res, tag);
        check(nm.dir[1 - model].nownfaces == 0, tag + ": other direction empty on owner side");
        free_nonmatching_data(nm);

        std::vector<double> mine;
        pack(mine, res);
        int cnt = static_cast<int>(mine.size());
        std::vector<int> counts(n), displs(n, 0);
        MPI_Gather(&cnt, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, sub);
        int total = 0;
        for (int r = 0; r < n; r++) { displs[r] = total; total += counts[r]; }
        all.resize(worldRank == 0 ? total : 0);
        MPI_Gatherv(mine.data(), cnt, MPI_DOUBLE, all.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, sub);
        MPI_Comm_free(&sub);
        EXASIM_COMM_WORLD = MPI_COMM_WORLD;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return to_map(all);
}

struct CaseResult {
    std::map<Key, std::vector<double>> dist;   // tuples of the largest partition (rank 0)
    InterfaceFaceSet<dstype, Int> g1, g2;      // global faces (rank 0)
};

// Runs one case for every partition with 1, 2, ..., worldSize/2 ranks per model. On
// rank 0: every partition's tuples must equal the serial reference and each other
// bitwise; the serial reference is checked against an exhaustive evaluation.
CaseResult run_case(const std::string& name, const ModelMesh& m1, const ModelMesh& m2, MasterData& m)
{
    std::vector<std::map<Key, std::vector<double>>> parts;
    std::vector<int> sizes;
    for (int n = 2; n <= worldSize; n *= 2) {
        parts.push_back(run_partition(name, m1, m2, m, n));
        sizes.push_back(n / 2);
    }

    CaseResult out;
    if (worldRank != 0) return out;
    out.dist = parts.back();
    out.g1 = global_faces(m1, m, 0, worldSize / 2);
    out.g2 = global_faces(m2, m, 1, worldSize / 2);

    for (std::size_t i = 0; i < parts.size(); i++) {
        const int k = sizes[i];
        const auto g1 = global_faces(m1, m, 0, k);
        const auto g2 = global_faces(m2, m, 1, k);
        std::vector<double> ref;
        for (int dir = 0; dir < 2; dir++) {
            const auto& own = dir == 0 ? g1 : g2;
            const auto& cand = dir == 0 ? g2 : g1;
            DonorSearchResult<dstype, Int> sr;
            sr.direction = dir;
            own_quadrature_points(sr, own, m.shapfgt.data(), static_cast<Int>(m.ngf));
            donor_search(sr, cand, m.xpf.data(), static_cast<Int>(m.elemtype), static_cast<Int>(m.porder));
            pack(ref, sr);
            if (i == 0) check_against_exhaustive(sr, cand, m, name);
        }
        const auto refmap = to_map(ref);
        long mismatch = 0;
        for (const auto& kv : refmap) {
            auto it = parts[i].find(kv.first);
            if (it == parts[i].end() || it->second != kv.second) mismatch++;
        }
        check(parts[i].size() == refmap.size() && mismatch == 0,
              name + ": " + std::to_string(k) + " per model equals serial reference (" +
              std::to_string(mismatch) + " differ)");

        // Rank-independent part (direction, owner face, g, f, a, dist, xi) identical across partitions.
        auto strip = [](std::map<Key, std::vector<double>> mm) {
            for (auto& kv : mm) { kv.second[3] = 0; kv.second[4] = 0; }
            return mm;
        };
        if (i > 0)
            check(strip(parts[i]) == strip(parts[0]),
                  name + ": " + std::to_string(k) + " per model identical to 1 per model");
        std::printf("%s: %d per model, %zu points, rank-independent checksum %016llx\n", name.c_str(), k,
                    parts[i].size(), checksum(strip(parts[i])));
    }
    return out;
}

std::vector<dstype> uniform(dstype a, dstype b, int n)
{
    std::vector<dstype> v(n + 1);
    for (int i = 0; i <= n; i++) v[i] = a + (b - a) * dstype(i) / dstype(n);
    v[n] = b;
    return v;
}

void matching_meshes(ModelMesh& m1, ModelMesh& m2)
{
    m1.xs = uniform(0.0, 0.5, 8);  m1.ys = uniform(0.0, 1.0, 16); m1.xiface = 0.5; m1.xfar = 0.0;
    m2.xs = uniform(0.5, 1.0, 8);  m2.ys = uniform(0.0, 1.0, 16); m2.xiface = 0.5; m2.xfar = 1.0;
}

void case_matching()
{
    MasterData m = make_master(1, 3);
    ModelMesh m1, m2;
    matching_meshes(m1, m2);
    CaseResult r = run_case("matching", m1, m2, m);
    if (worldRank != 0) return;
    for (int dir = 0; dir < 2; dir++) {
        const auto& own = dir == 0 ? r.g1 : r.g2;
        const auto& oth = dir == 0 ? r.g2 : r.g1;
        for (int l = 0; l < own.size(); l++) {
            const int pc = partner_index(own, l, oth, m);
            check(pc >= 0, "matching: partner exists");
            if (pc < 0) continue;
            for (int g = 0; g < m.ngf; g++) {
                const auto& t = r.dist.at(Key((long(dir) << 40) + long(own.global[l]), g));
                check(Int(t[5]) == oth.global[pc], "matching: donor = conformal partner");
                check(t[7] < 1e-12, "matching: dist < 1e-12");
            }
        }
    }
}

// Conformal regression (review F3): interfacetype = 0 on the matching two-model mesh.
// build_runtime_interface_partition must build the conformal arrays exactly as expected
// from an independent geometric pairing, and the non-matching setup must do nothing.
void case_conformal()
{
    MasterData m = make_master(1, 3);
    ModelMesh m1, m2;
    matching_meshes(m1, m2);
    for (int n = 2; n <= worldSize; n *= 2) {
        MPI_Comm sub;
        MPI_Comm_split(MPI_COMM_WORLD, worldRank < n ? 0 : MPI_UNDEFINED, worldRank, &sub);
        if (sub != MPI_COMM_NULL) {
            EXASIM_COMM_WORLD = sub;
            const int k = n / 2;
            const int model = (worldRank < k) ? 0 : 1;
            const int part = worldRank - model * k;
            const std::string tag = "conformal (" + std::to_string(k) + " per model)";

            RankStructs rs;
            build_rank_structs(rs, model == 0 ? m1 : m2, m, part, k, 0);
            std::vector<Int> lsize(1, 0);
            rs.mesh.lsize = lsize.data();
            exasim::interfacepartition::build_runtime_interface_partition(rs.app, rs.master, rs.mesh, rs.sol,
                    static_cast<Int>(n), static_cast<Int>(worldRank), static_cast<Int>(model * k));

            // Expected arrays (same rules as the conformal builder, computed from the global meshes).
            const auto g1 = global_faces(m1, m, 0, k);
            const auto g2 = global_faces(m2, m, 1, k);
            const auto& own = model == 0 ? g1 : g2;
            const auto& oth = model == 0 ? g2 : g1;
            const auto ownOrd = rank_ordinals(own), othOrd = rank_ordinals(oth);
            std::map<Int, std::vector<std::pair<int, int>>> peers;
            std::vector<Int> perm;
            for (int l = 0; l < own.size(); l++) {
                if (own.rank[l] != worldRank) continue;
                const int pc = partner_index(own, l, oth, m);
                check(pc >= 0, tag + ": partner exists");
                if (pc < 0) continue;
                const int canonical = (model == 0) ? ownOrd[l] : othOrd[pc];
                peers[oth.rank[pc]].push_back(std::make_pair(canonical, ownOrd[l]));
                std::vector<Int> fp(m.npf, -1);
                for (int a = 0; a < m.npf; a++)
                    for (int b = 0; b < m.npf; b++)
                        if (std::abs(own.nodes[a + m.npf * m.nd * l] - oth.nodes[b + m.npf * m.nd * pc]) < 1e-12 &&
                            std::abs(own.nodes[a + m.npf + m.npf * m.nd * l] - oth.nodes[b + m.npf + m.npf * m.nd * pc]) < 1e-12)
                            fp[a] = b;
                if (perm.empty()) perm = fp;
                check(fp == perm, tag + ": one node permutation for all faces");
            }
            std::vector<Int> nb, faces, pts;
            for (auto& kv : peers) {
                std::sort(kv.second.begin(), kv.second.end());
                nb.push_back(kv.first);
                pts.push_back(static_cast<Int>(kv.second.size()));
                for (const auto& it : kv.second) faces.push_back(it.second);
            }
            auto same = [](const Int* a, Int sz, const std::vector<Int>& v) {
                if (sz != static_cast<Int>(v.size())) return false;
                for (Int i = 0; i < sz; i++) if (a[i] != v[i]) return false;
                return true;
            };
            check(same(rs.mesh.nbintf, rs.mesh.sznbintf, nb), tag + ": nbintf");
            check(same(rs.mesh.facesend, rs.mesh.szfacesend, faces), tag + ": facesend");
            check(same(rs.mesh.facerecv, rs.mesh.szfacerecv, faces), tag + ": facerecv");
            check(same(rs.mesh.facesendpts, rs.mesh.szfacesendpts, pts), tag + ": facesendpts");
            check(same(rs.mesh.facerecvpts, rs.mesh.szfacerecvpts, pts), tag + ": facerecvpts");
            check(same(rs.mesh.faceperm, rs.mesh.szfaceperm, perm), tag + ": faceperm");
            check(rs.mesh.lsize[0] >= 45 && rs.mesh.nsize[39] == rs.mesh.szfaceperm &&
                  rs.mesh.nsize[40] == rs.mesh.sznbintf && rs.mesh.nsize[41] == rs.mesh.szfacesend,
                  tag + ": mesh.nsize slots 39-44");
            check(!nb.empty(), tag + ": rank has a conformal neighbor");

            nonmatchingdataT<dstype, Int> nm;
            check(!setup_nonmatching_interface(rs.app, rs.master, rs.mesh, rs.sol, static_cast<Int>(n),
                      static_cast<Int>(worldRank), static_cast<Int>(model * k), nm) &&
                  nm.interfacetype == 0 && nm.dir[0].ownface == nullptr && nm.dir[1].ownface == nullptr &&
                  nm.dir[0].direction == -1 && nm.dir[1].direction == -1,
                  tag + ": non-matching setup is a no-op");

            CPUFREE(rs.mesh.nsize);
            CPUFREE(rs.mesh.faceperm);
            CPUFREE(rs.mesh.nbintf);
            CPUFREE(rs.mesh.facesend);
            CPUFREE(rs.mesh.facesendpts);
            CPUFREE(rs.mesh.facerecv);
            CPUFREE(rs.mesh.facerecvpts);
            if (worldRank == 0) std::printf("%s: conformal arrays as expected\n", tag.c_str());
            MPI_Comm_free(&sub);
            EXASIM_COMM_WORLD = MPI_COMM_WORLD;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

void case_nonmatching()
{
    MasterData m = make_master(0, 2);
    ModelMesh m1, m2;
    m1.tri = true;
    m1.xs = uniform(0.0, 0.5, 4); m1.ys = uniform(0.0, 1.0, 12); m1.xiface = 0.5; m1.xfar = 0.0;
    m2.tri = true;
    m2.xs = uniform(0.5, 1.0, 3);
    m2.ys = {0.0, 0.11, 0.27, 0.38, 0.55, 0.71, 0.82, 0.9}; // 7 faces, model 2 shorter (y <= 0.9)
    m2.xiface = 0.5; m2.xfar = 1.0; m2.amp = 0.01;         // curved: gaps and overlaps
    CaseResult r = run_case("nonmatching", m1, m2, m);
    if (worldRank != 0) return;
    long beyond = 0, off = 0, multiK = 0;
    std::map<std::pair<long, long>, std::vector<long>> K;
    for (const auto& kv : r.dist) {
        const auto& t = kv.second;
        if (t[0] == 0 && t[7] > 1e-8) off++;
        if (t[0] == 0 && t[7] > 0.02) beyond++; // model-1 points above y = 0.9: beyond the edge of Gamma^2
        K[std::make_pair(long(t[0]), long(t[1]))].push_back(long(t[5]));
    }
    for (auto& kv : K) {
        std::sort(kv.second.begin(), kv.second.end());
        if (std::unique(kv.second.begin(), kv.second.end()) - kv.second.begin() > 1) multiK++;
    }
    check(off > 0, "nonmatching: some points off their donor face");
    check(multiK > 0, "nonmatching: some faces with |K_l| > 1");
    check(beyond > 0, "nonmatching: points beyond the edge of Gamma^2");
    std::printf("nonmatching: %zu points, %ld off donor face, %ld beyond edge, %ld faces with |K_l|>1\n",
                r.dist.size(), off, beyond, multiK);
}

void case_tie_mesh()
{
    MasterData m = make_master(1, 1);
    // Heights of the two Gauss points of the model-1 face on the cell row [0, 0.25]; the
    // face node order is not assumed, so both ystar and 0.25 - ystar become model-2 vertices.
    dstype ystar = 0.0;
    for (int j = 0; j < m.npf; j++) ystar += m.shapfgt[0 + m.ngf * j] * (j == 0 ? 0.0 : 0.25);
    ModelMesh m1, m2;
    m1.xs = uniform(0.0, 0.5, 2); m1.ys = uniform(0.0, 1.0, 4); m1.xiface = 0.5; m1.xfar = 0.0;
    const dstype gap = 0.01;
    m2.xs = uniform(0.5 + gap, 1.0, 2); m2.xiface = 0.5 + gap; m2.xfar = 1.0;
    const dstype y1 = 0.25 - ystar;
    m2.ys = {0.0, std::min(ystar, y1), std::max(ystar, y1), 0.6, 1.0};
    CaseResult r = run_case("tie", m1, m2, m);
    if (worldRank != 0) return;

    int ties = 0;
    for (int l = 0; l < r.g1.size(); l++) {
        if (r.g1.nodes[m.npf + m.npf * m.nd * l] > 0.25 + 1e-12 || r.g1.nodes[1 + m.npf + m.npf * m.nd * l] > 0.25 + 1e-12)
            continue;
        for (int g = 0; g < m.ngf; g++) {
            const auto& t = r.dist.at(Key(long(r.g1.global[l]), g));
            dstype x[2] = {0, 0};
            for (int j = 0; j < m.npf; j++) {
                x[0] += m.shapfgt[g + m.ngf * j] * r.g1.nodes[j + m.npf * m.nd * l];
                x[1] += m.shapfgt[g + m.ngf * j] * r.g1.nodes[j + m.npf + m.npf * m.nd * l];
            }
            std::vector<Int> tied;
            for (int c = 0; c < r.g2.size(); c++) {
                dstype xi[2], xb[2], d;
                ClosestPointOnFace(xi, xb, &d, x, &r.g2.nodes[m.npf * m.nd * c], m.xpf.data(), m.nd, m.npf,
                                   m.elemtype, m.porder, 100, 1e-13);
                if (std::abs(d - t[7]) <= 1e-14) tied.push_back(r.g2.global[c]);
            }
            check(tied.size() == 2, "tie: point equidistant to exactly two faces");
            if (tied.size() == 2) {
                check(Int(t[5]) == std::min(tied[0], tied[1]), "tie: smaller global face index chosen");
                check(std::abs(t[7] - gap) < 1e-14, "tie: distance equals the gap");
                ties++;
            }
        }
    }
    check(ties == 2, "tie: two tied points found");
    std::printf("tie (mesh): %d tied points resolved to the smaller global face index\n", ties);
}

// Two-face candidate sets for direct (serial) checks of donor_search.
InterfaceFaceSet<dstype, Int> face_pair(const MasterData& m, const std::vector<std::vector<dstype>>& xs,
        const std::vector<std::vector<dstype>>& ys, const std::vector<Int>& global)
{
    InterfaceFaceSet<dstype, Int> fs;
    fs.nd = m.nd;
    fs.npf = m.npf;
    for (std::size_t f = 0; f < global.size(); f++) {
        fs.rank.push_back(static_cast<Int>(f));
        fs.model.push_back(1);
        fs.elem.push_back(static_cast<Int>(f));
        fs.lface.push_back(0);
        fs.global.push_back(global[f]);
        for (int j = 0; j < m.npf; j++) fs.nodes.push_back(xs[f][j]);
        for (int j = 0; j < m.npf; j++) fs.nodes.push_back(ys[f][j]);
    }
    return fs;
}

Int donor_of(const InterfaceFaceSet<dstype, Int>& cand, const MasterData& m, dstype px, dstype py, dstype* dist = nullptr)
{
    DonorSearchResult<dstype, Int> res;
    res.nd = m.nd; res.ngf = 1; res.nfaces = 1; res.npts = 1;
    res.x = {px, py};
    donor_search(res, cand, m.xpf.data(), static_cast<Int>(m.elemtype), static_cast<Int>(m.porder));
    check(res.converged[0] == 1, "direct: projection converged");
    if (dist) *dist = res.dist[0];
    return res.f[0];
}

// Straight vertical face x = c, y in [0, 1], nodes at the reference face nodes.
void straight(std::vector<dstype>& x, std::vector<dstype>& y, const MasterData& m, dstype c)
{
    x.assign(m.npf, c);
    y.assign(m.npf, 0.0);
    for (int j = 0; j < m.npf; j++) y[j] = m.xpf[j];
}

// Review F2: a face farther by a resolvable amount is never selected, whatever its index;
// numerically equal distances go to the smaller global index.
void case_near_ties()
{
    if (worldRank != 0) return;
    MasterData m = make_master(1, 1);
    std::vector<dstype> xa, ya, xb, yb;
    const dstype base = 0.1;
    for (dstype delta : {1e-9, 5e-11, 1e-12, 1e-13}) {
        straight(xa, ya, m, -(base + delta)); // lower index, farther by delta
        straight(xb, yb, m, base);
        const auto cand = face_pair(m, {xa, xb}, {ya, yb}, {3, 7});
        const Int f = donor_of(cand, m, 0.0, 0.5);
        check(f == 7, "near tie: closer face wins when farther by " + std::to_string(delta));
    }
    // Exact ties (mirror images): the smaller global index wins, in both index orders.
    straight(xa, ya, m, -base);
    straight(xb, yb, m, base);
    check(donor_of(face_pair(m, {xa, xb}, {ya, yb}, {3, 7}), m, 0.0, 0.5) == 3, "exact tie: smaller index (3, 7)");
    check(donor_of(face_pair(m, {xa, xb}, {ya, yb}, {9, 4}), m, 0.0, 0.5) == 4, "exact tie: smaller index (9, 4)");
    // Shared vertex: point beyond the common end point of two collinear faces.
    std::vector<dstype> y0(m.npf), y1(m.npf);
    for (int j = 0; j < m.npf; j++) { y0[j] = 0.5 * m.xpf[j]; y1[j] = 0.5 + 0.5 * m.xpf[j]; }
    xa.assign(m.npf, 0.2);
    check(donor_of(face_pair(m, {xa, xa}, {y1, y0}, {5, 6}), m, 0.0, 0.5) == 5, "vertex tie: smaller index");
    std::printf("near ties: resolvably farther faces rejected for delta = 1e-9 .. 1e-13; exact ties to the smaller index\n");

}

// Tie-threshold cases (reviews F2-R1, F2-R2). Face L (global 3, the smaller index) is
// farther from P than face H (global 7) by delta. The rule selects L iff
// lower_L <= upper_H (U = upper_H), so the transition is at
//     tau = (d_L - lower_L) + (upper_H - d_H),
// evaluated at delta = 0 from the certified bounds and INDEPENDENT reference
// distances d_L = d_H = D (analytic; refErr bounds their rounding). The bounds must
// enclose the references, and the donor must be L at 0.5 and 0.85 tau and H at 1.15
// and 2 tau. Faces are given as node arrays (x then y) built for a given delta.
using Nodes = std::vector<dstype>;
void threshold_case(const std::string& name, const MasterData& mp, const dstype* P, dstype D, dstype refErr,
                    const std::function<Nodes(dstype)>& L, const std::function<Nodes(dstype)>& H)
{
    const auto bm = make_bernstein_master(mp.xpf.data(), Int(mp.porder));
    auto bounds = [&](const Nodes& y) {
        dstype xi[2], xb[2], d;
        ClosestPointOnFace(xi, xb, &d, P, y.data(), mp.xpf.data(), mp.nd, mp.npf, mp.elemtype, mp.porder, 100, 1e-13);
        return face_distance_bounds(bm, y.data(), Int(mp.npf), P, xi[0]);
    };
    const auto bL = bounds(L(0.0)), bH = bounds(H(0.0));
    check(bL.ok && bH.ok, name + ": bounds valid");
    check(bL.lower <= D + refErr && bL.upper >= D - refErr && bH.lower <= D + refErr && bH.upper >= D - refErr,
          name + ": certified bounds enclose the reference distance");
    const dstype lL = D - bL.lower, uH = bH.upper - D;
    const dstype tau = lL + uH;
    check(tau > 4 * refErr, name + ": threshold resolvable against the reference error");
    for (dstype factor : {0.5, 0.85, 1.15, 2.0}) {
        const dstype delta = factor * tau;
        auto toSet = [&](const Nodes& a, const Nodes& b) {
            return face_pair(mp, {Nodes(a.begin(), a.begin() + mp.npf), Nodes(b.begin(), b.begin() + mp.npf)},
                             {Nodes(a.begin() + mp.npf, a.end()), Nodes(b.begin() + mp.npf, b.end())}, {3, 7});
        };
        const Int f = donor_of(toSet(L(delta), H(delta)), mp, P[0], P[1]);
        const Int expect = factor < 1 ? 3 : 7;
        check(f == expect, name + ": donor at " + std::to_string(factor) + " tau");
        std::printf("%s: tau = %.3e (d_L - lower_L = %.3e, upper_H - d_H = %.3e), delta = %.2f tau -> donor %d (expected %d)\n",
                    name.c_str(), double(tau), double(lL), double(uH), double(factor), int(f), int(expect));
    }
}

// Straight face through Q with direction u (unit length 1 in the reference face).
Nodes straight_face(const MasterData& mp, const dstype* Q, const dstype* u)
{
    Nodes y(2 * mp.npf);
    for (int j = 0; j < mp.npf; j++) {
        y[j] = Q[0] + (mp.xpf[j] - 0.5) * u[0];
        y[j + mp.npf] = Q[1] + (mp.xpf[j] - 0.5) * u[1];
    }
    return y;
}

// Parabola face G(xi) = ((xi - 1/2)^2, xi): nodes at the reference face nodes.
Nodes parabola_face(const MasterData& mp)
{
    Nodes y(2 * mp.npf);
    for (int j = 0; j < mp.npf; j++) { y[j] = (mp.xpf[j] - 0.5) * (mp.xpf[j] - 0.5); y[j + mp.npf] = mp.xpf[j]; }
    return y;
}

void case_tie_thresholds()
{
    if (worldRank != 0) return;
    // Straight faces (retained from Revision 2): P = (0, 0.5), L at x = -(0.1 + delta), H at x = 0.1.
    for (int porder : {1, 4}) {
        MasterData mp = make_master(1, porder);
        const dstype P[2] = {0.0, 0.5};
        const dstype up[2] = {0.0, 1.0};
        threshold_case("threshold straight p=" + std::to_string(porder), mp, P, 0.1, 0.0,
            [&](dstype dl) { const dstype Q[2] = {-(0.1 + dl), 0.5}; return straight_face(mp, Q, up); },
            [&](dstype) { const dstype Q[2] = {0.1, 0.5}; return straight_face(mp, Q, up); });
    }
    // Curved faces (review F2-R2): the parabola G(xi) = ((xi - 1/2)^2, xi) and
    // P = G(0.3) + 0.25 n, n = (-1, -0.4)/sqrt(1.16) the unit normal at xi = 0.3 on the
    // convex side. The region x >= (y - 1/2)^2 is convex and P lies outside it, so the
    // closest point is G(0.3) and the distance is exactly 0.25 (analytic reference). The
    // projection starts at the centroid 0.5 and must iterate (nonzero projection error).
    // Rounding of P and of the node coordinates (the nodes for p = 4 are irrational)
    // changes the true distance by at most refErr = 1e-15.
    for (int porder : {2, 4}) {
        MasterData mp = make_master(1, porder);
        const dstype s = 1.0 / std::sqrt(1.16);
        const dstype nrm[2] = {-s, -0.4 * s}, tan[2] = {0.4 * s, -s}; // tan is perpendicular to nrm
        const dstype G[2] = {0.04, 0.3};
        const dstype P[2] = {G[0] + 0.25 * nrm[0], G[1] + 0.25 * nrm[1]};
        const Nodes curved = parabola_face(mp);
        // curved face farther (smaller index), straight face closer
        threshold_case("threshold curved-farther p=" + std::to_string(porder), mp, P, 0.25, 1e-15,
            [&](dstype) { return curved; },
            [&](dstype dl) { const dstype Q[2] = {P[0] + (0.25 - dl) * nrm[0], P[1] + (0.25 - dl) * nrm[1]};
                             return straight_face(mp, Q, tan); });
        // straight face farther (smaller index), curved face closer
        threshold_case("threshold curved-closer p=" + std::to_string(porder), mp, P, 0.25, 1e-15,
            [&](dstype dl) { const dstype Q[2] = {P[0] + (0.25 + dl) * nrm[0], P[1] + (0.25 + dl) * nrm[1]};
                             return straight_face(mp, Q, tan); },
            [&](dstype) { return curved; });
    }
}

// Review F1: high-order curved faces whose polynomial image leaves the box of
// their nodes. A box of the nodes, even padded by 25% of its diagonal (the
// pruning of the first submission), is not a lower bound of the distance to
// such a face, so box pruning can skip the donor; the exhaustive search must not.
// Face A (global 9): p = 4, y = xi. Two shapes:
//   bump:     x = c at the interior node nearest xi = 0.25, 0 at the other nodes;
//   lebesgue: x_j = -c sign(l_j(xi*)), xi* the maximizer of sum_j |l_j| on F_hat,
//             so x(xi*) = -c Lambda lies outside the nodal box [-c, c] by c (Lambda - 1).
// P lies 0.05 beyond the most negative point of the image (found by dense sampling,
// independent of the projection). Face B (global 2, smaller index) is straight and
// lies between: d_A < d_B < (distance from P to the box of A, padded by 25% of its
// diagonal when that still excludes P).
struct CurvedOutcome {
    bool converged = false; Int donor = -1;
    dstype dist = 0, dA = 0, dB = 0, lbNodal = 0, lbPadded = 0, xmin = 0;
};

CurvedOutcome curved_case(const MasterData& m, dstype c, bool lebesgue)
{
    CurvedOutcome o;
    const int ns = 20000;
    std::vector<dstype> psi(m.npf), xa(m.npf, 0.0), ya(m.npf);
    for (int j = 0; j < m.npf; j++) ya[j] = m.xpf[j];
    if (!lebesgue) {
        int jbump = -1;
        for (int j = 0; j < m.npf; j++)
            if (m.xpf[j] > 0.0 && m.xpf[j] < 1.0 && (jbump < 0 || std::abs(m.xpf[j] - 0.25) < std::abs(m.xpf[jbump] - 0.25)))
                jbump = j;
        xa[jbump] = c;
    } else {
        dstype lmax = -1;
        std::vector<dstype> best(m.npf);
        for (int i = 0; i <= ns; i++) {
            const dstype xi[2] = {dstype(i) / ns, 0.0};
            EvaluateFaceShape(psi.data(), xi, m.xpf.data(), m.nd, m.npf, m.elemtype, m.porder);
            dstype l = 0;
            for (int j = 0; j < m.npf; j++) l += std::abs(psi[j]);
            if (l > lmax) { lmax = l; best = psi; }
        }
        for (int j = 0; j < m.npf; j++) xa[j] = (best[j] >= 0 ? -c : c);
    }
    dstype ymin = 0;
    o.xmin = 1e300;
    std::vector<dstype> sx(ns + 1), sy(ns + 1);
    for (int i = 0; i <= ns; i++) {
        const dstype xi[2] = {dstype(i) / ns, 0.0};
        EvaluateFaceShape(psi.data(), xi, m.xpf.data(), m.nd, m.npf, m.elemtype, m.porder);
        sx[i] = 0; sy[i] = 0;
        for (int j = 0; j < m.npf; j++) { sx[i] += psi[j] * xa[j]; sy[i] += psi[j] * ya[j]; }
        if (sx[i] < o.xmin) { o.xmin = sx[i]; ymin = sy[i]; }
    }
    const dstype px = o.xmin - 0.05, py = ymin;
    o.dA = 1e300;
    for (int i = 0; i <= ns; i++) o.dA = std::min(o.dA, std::hypot(px - sx[i], py - sy[i]));
    dstype bxmin = 1e300, bxmax = -1e300;
    for (int j = 0; j < m.npf; j++) { bxmin = std::min(bxmin, xa[j]); bxmax = std::max(bxmax, xa[j]); }
    const dstype diag = std::hypot(bxmax - bxmin, 1.0);
    o.lbNodal = bxmin - px;                          // P.y lies within [0, 1]
    o.lbPadded = std::max<dstype>(0.0, o.lbNodal - 0.25 * diag);
    // B between A and the (padded, when it still excludes P) box of A, so that box pruning skips A.
    o.dB = 0.5 * (o.dA + (o.lbPadded > o.dA ? o.lbPadded : o.lbNodal));
    std::vector<dstype> xb, yb;
    straight(xb, yb, m, px - o.dB);
    DonorSearchResult<dstype, Int> res;
    res.nd = m.nd; res.ngf = 1; res.nfaces = 1; res.npts = 1;
    res.x = {px, py};
    donor_search(res, face_pair(m, {xa, xb}, {ya, yb}, {9, 2}), m.xpf.data(),
                 static_cast<Int>(m.elemtype), static_cast<Int>(m.porder));
    o.converged = res.converged[0] == 1;
    o.donor = res.f[0];
    o.dist = res.dist[0];
    return o;
}

void case_curved_adversarial()
{
    if (worldRank != 0) return;
    MasterData m = make_master(1, 4); // p = 4 line faces
    bool paddedDefeated = false;
    for (int mode = 0; mode < 2; mode++)
        for (dstype c : {1.0, 4.0}) {
            const CurvedOutcome o = curved_case(m, c, mode == 1);
            const std::string tag = std::string("curved ") + (mode ? "lebesgue" : "bump") + " c=" + std::to_string(c);
            check(o.lbNodal > o.dA, tag + ": image leaves the nodal box (box is not a lower bound)");
            check(o.converged, tag + ": projection converged");
            check(o.donor == 9, tag + ": exhaustive search selects the curved face");
            check(std::abs(o.dist - o.dA) < 1e-6, tag + ": distance matches dense sampling");
            if (o.lbPadded > o.dB) paddedDefeated = true;
            std::printf("%s: image min x = %.4f, d_A = %.6f, d_B = %.6f, box bound %.6f (padded %.6f), donor %d\n",
                        tag.c_str(), o.xmin, o.dA, o.dB, o.lbNodal, o.lbPadded, int(o.donor));
        }
    check(paddedDefeated, "curved: a case where the 25%-padded box would also have pruned the donor");
}

// Review F1-R1: the safeguarded projection must not report convergence when no
// step decreases the objective but the full step is above the tolerance. A point
// at y = 1e20 above a vertical face makes |G(xi) - x|^2 constant in floating point
// (1e20 - y == 1e20), while the full Gauss-Newton step from the first sample (xi = 0)
// to the true minimizer (xi = 1) is 1.
void case_safeguard_stagnation()
{
    if (worldRank != 0) return;
    MasterData m = make_master(1, 2);
    std::vector<dstype> xs, ys;
    straight(xs, ys, m, 0.0);
    std::vector<dstype> nodes(xs);
    nodes.insert(nodes.end(), ys.begin(), ys.end());
    const auto fs = make_face_samples(m.xpf.data(), Int(m.nd), Int(m.npf), Int(m.elemtype), Int(m.porder));
    const dstype x[2] = {0.0, 1e20};
    dstype xi[2] = {0, 0}, d = 0;
    const bool ok = safeguarded_closest_point(xi, &d, x, nodes.data(), m.xpf.data(), fs, Int(m.nd), Int(m.npf),
                                              Int(m.elemtype), Int(m.porder), DonorSearchParams());
    check(!ok, "safeguard: stagnation with a large full step is not reported as converged");
    // Control: an ordinary point converges.
    const dstype x2[2] = {-0.3, 0.4};
    const bool ok2 = safeguarded_closest_point(xi, &d, x2, nodes.data(), m.xpf.data(), fs, Int(m.nd), Int(m.npf),
                                               Int(m.elemtype), Int(m.porder), DonorSearchParams());
    check(ok2 && std::abs(d - 0.3) < 1e-14, "safeguard: ordinary point converges");
    std::printf("safeguard: stagnation reported as failure (%d), ordinary point converged (%d)\n", int(!ok), int(ok2));
}

// Review F4-R1: a failed projection onto ANY candidate fails the point, even a
// far-away one. A degenerate face (all nodes equal, singular tangent) cannot be
// projected; the near face is fine.
void case_failed_candidate()
{
    if (worldRank != 0) return;
    MasterData m = make_master(1, 1);
    std::vector<dstype> xa, ya, xb(m.npf, 50.0), yb(m.npf, 50.0);
    straight(xa, ya, m, 0.1);
    DonorSearchResult<dstype, Int> res;
    res.nd = m.nd; res.ngf = 1; res.nfaces = 1; res.npts = 1;
    res.x = {0.0, 0.5};
    donor_search(res, face_pair(m, {xa, xb}, {ya, yb}, {1, 2}), m.xpf.data(),
                 static_cast<Int>(m.elemtype), static_cast<Int>(m.porder));
    check(res.converged[0] == 0, "failed candidate: point reported as failed");
    std::printf("failed candidate: far degenerate face fails the point (converged = %d)\n", int(res.converged[0]));
}

// Production face basis against an independent analytic Lagrange evaluation
// (review F2-R2), for both element types and p = 1..4 (nd = 2):
//   l_j(xi) = prod_{k != j} (xi - t_k)/(t_j - t_k)   (t = xpf; relative error <= 3p eps),
// at 2001 points of [0, 1] and at the nodes. Checked per value:
//   |psi_j - l_j| <= npf eps (|l_j| + 1)     (the assumption of rounding_scale),
// and, for adversarial nodal coordinates (each unit vector e_k, and the sign pattern
// y_j = sign(psi_j - l_j) that maximizes the weighted error), the weighted form
//   sum_j |psi_j - l_j| |y_j| <= npf eps (sum_j |psi_j||y_j| + sum_j |y_j|).
// The certified distance bounds of the tie rule do not use the production basis;
// the test also checks that the production face map equals the Bernstein form of
// the same interpolant to within these bounds.
void case_basis_accuracy()
{
    if (worldRank != 0) return;
    const dstype eps = std::numeric_limits<dstype>::epsilon();
    for (int et = 0; et < 2; et++)
        for (int p = 1; p <= 4; p++) {
            MasterData m = make_master(et, p);
            const int n = m.npf;
            const auto bm = make_bernstein_master(m.xpf.data(), Int(p));
            std::vector<dstype> psi(n), lag(n);
            dstype worstValue = 0, worstWeighted = 0, worstBern = 0, worstRel = 0, worstAbs = 0;
            std::vector<dstype> pts;
            for (int i = 0; i <= 2000; i++) pts.push_back(dstype(i) / 2000);
            for (int j = 0; j < n; j++) pts.push_back(m.xpf[j]);
            // a test face with adversarial (alternating, large) coordinates
            std::vector<dstype> yf(2 * n);
            for (int j = 0; j < n; j++) { yf[j] = (j % 2 ? -3.0 : 5.0) * (1 + j); yf[j + n] = m.xpf[j]; }
            const auto bf = bernstein_face(bm, yf.data(), Int(n));
            for (dstype t : pts) {
                const dstype xi[2] = {t, 0.0};
                EvaluateFaceShape(psi.data(), xi, m.xpf.data(), m.nd, n, et, p);
                for (int j = 0; j < n; j++) {
                    dstype v = 1;
                    for (int k = 0; k < n; k++)
                        if (k != j) v *= (t - m.xpf[k]) / (m.xpf[j] - m.xpf[k]);
                    lag[j] = v;
                }
                dstype wsign = 0, Ssign = 0, Ysign = 0;
                for (int j = 0; j < n; j++) {
                    const dstype err = std::abs(psi[j] - lag[j]);
                    const dstype refe = 3 * p * eps * std::abs(lag[j]); // error of the reference itself
                    worstValue = std::max(worstValue, (err - refe) / (n * eps * (std::abs(lag[j]) + 1)));
                    if (std::abs(lag[j]) > 1e-3) worstRel = std::max(worstRel, err / (eps * std::abs(lag[j])));
                    worstAbs = std::max(worstAbs, err / eps);
                    // unit vector e_j: weighted error = err, weighted bound = n eps |psi_j| + eps
                    worstWeighted = std::max(worstWeighted, (err - refe) / (n * eps * (std::abs(psi[j]) + 1)));
                    wsign += err - refe;
                    Ssign += std::abs(psi[j]);
                    Ysign += 1;
                }
                worstWeighted = std::max(worstWeighted, wsign / (n * eps * (Ssign + Ysign)));
                // production face map vs Bernstein form of the same interpolant
                dstype g[2], gp[2] = {0, 0};
                bernstein_eval(g, bf, p, t);
                for (int j = 0; j < n; j++) { gp[0] += psi[j] * yf[j]; gp[1] += psi[j] * yf[j + n]; }
                const dstype tolb = (2 * p + 2) * eps * bf.cmax + bf.beta + 2 * n * eps * (5.0 * (n + 1));
                worstBern = std::max(worstBern, std::max(std::abs(g[0] - gp[0]), std::abs(g[1] - gp[1])) / tolb);
            }
            check(worstValue <= 1.0, "basis accuracy: per-value bound");
            check(worstWeighted <= 1.0, "basis accuracy: weighted bound, adversarial coordinates");
            check(worstBern <= 1.0, "basis accuracy: production map = Bernstein form");
            std::printf("basis accuracy et=%d p=%d: max |psi-l| = %.2f eps (max relative, |l| > 1e-3: %.2f), per-value bound "
                        "used %.2f, weighted bound used %.2f, Bernstein agreement %.2f of its bound\n",
                        et, p, double(worstAbs), double(worstRel), double(worstValue), double(worstWeighted),
                        double(worstBern));
        }
}

// ---------------------------------------------------------------------------
// Review F2-R3: high-precision reference for the certified distance bounds.
// Double-double arithmetic (about 106 bits; error-free TwoSum and TwoProd with
// std::fma), confined to this test. For a face given by its double nodes
// (xpf_j, y_j), the exact interpolant G(t) = sum_j l_j(t) y_j (Lagrange product
// form) and f(t) = |G(t) - x|^2 are evaluated in double-double. The global
// minimum over [0, 1] is found by 1000-point sampling, golden-section refinement
// (160 steps, bracket width < 1e-30) around every sampled local minimum and the
// two end points. d* = sqrt(min f) is accurate far below the widths of the
// bounds (the test also prints the smallest margins).
struct DD {
    double hi = 0, lo = 0;
    DD() = default;
    DD(double h) : hi(h), lo(0) {}
    DD(double h, double l) : hi(h), lo(l) {}
};
inline DD dd_two_sum(double a, double b) { const double s = a + b, bb = s - a; return DD(s, (a - (s - bb)) + (b - bb)); }
inline DD dd_quick(double a, double b) { const double s = a + b; return DD(s, b - (s - a)); }
inline DD operator+(const DD& a, const DD& b) { DD s = dd_two_sum(a.hi, b.hi); s.lo += a.lo + b.lo; return dd_quick(s.hi, s.lo); }
inline DD operator-(const DD& a) { return DD(-a.hi, -a.lo); }
inline DD operator-(const DD& a, const DD& b) { return a + (-b); }
inline DD operator*(const DD& a, const DD& b)
{
    const double p = a.hi * b.hi, e = std::fma(a.hi, b.hi, -p);
    return dd_quick(p, e + (a.hi * b.lo + a.lo * b.hi));
}
inline DD operator/(const DD& a, const DD& b)
{
    const double q1 = a.hi / b.hi;
    DD r = a - b * DD(q1);
    const double q2 = r.hi / b.hi;
    r = r - b * DD(q2);
    const double q3 = r.hi / b.hi;
    return dd_quick(q1, q2) + DD(q3);
}
inline bool dd_less(const DD& a, const DD& b) { const DD d = a - b; return d.hi < 0 || (d.hi == 0 && d.lo < 0); }
inline DD dd_sqrt(const DD& a)
{
    if (a.hi <= 0) return DD(0);
    const double q = std::sqrt(a.hi);
    const DD r = a - DD(q) * DD(q);
    return dd_quick(q, r.hi / (2 * q));
}

struct DDFace {
    std::vector<double> t, y; // nodes xpf (n), coordinates y (n x 2)
    int n = 0;
    DD f(double s, const double* x) const
    {
        DD g0(0), g1(0);
        for (int j = 0; j < n; j++) {
            DD l(1);
            for (int k = 0; k < n; k++)
                if (k != j) l = l * ((DD(s) - DD(t[k])) / (DD(t[j]) - DD(t[k])));
            g0 = g0 + l * DD(y[j]);
            g1 = g1 + l * DD(y[j + n]);
        }
        const DD a = g0 - DD(x[0]), b = g1 - DD(x[1]);
        return a * a + b * b;
    }
    DD fdd(const DD& s, const double* x) const // f at a double-double parameter
    {
        DD g0(0), g1(0);
        for (int j = 0; j < n; j++) {
            DD l(1);
            for (int k = 0; k < n; k++)
                if (k != j) l = l * ((s - DD(t[k])) / (DD(t[j]) - DD(t[k])));
            g0 = g0 + l * DD(y[j]);
            g1 = g1 + l * DD(y[j + n]);
        }
        const DD a = g0 - DD(x[0]), b = g1 - DD(x[1]);
        return a * a + b * b;
    }
    DD distance(const double* x) const
    {
        const int N = 1000;
        std::vector<DD> fs(N + 1);
        for (int i = 0; i <= N; i++) fs[i] = f(double(i) / N, x);
        DD best = fs[0];
        if (dd_less(fs[N], best)) best = fs[N];
        const DD gr(0.6180339887498949, -5.4321152036825061e-17);
        for (int i = 1; i < N; i++) {
            if (dd_less(fs[i - 1], fs[i]) || dd_less(fs[i + 1], fs[i])) continue; // not a sampled local minimum
            DD a(double(i - 1) / N), b(double(i + 1) / N);
            DD c = b - gr * (b - a), d = a + gr * (b - a);
            DD fc = fdd(c, x), fd = fdd(d, x);
            for (int it = 0; it < 160; it++) {
                if (dd_less(fc, fd)) { b = d; d = c; fd = fc; c = b - gr * (b - a); fc = fdd(c, x); }
                else { a = c; c = d; fc = fd; d = a + gr * (b - a); fd = fdd(d, x); }
            }
            if (dd_less(fc, best)) best = fc;
            if (dd_less(fd, best)) best = fd;
        }
        return dd_sqrt(best);
    }
};

// Checks lower <= d* <= upper for one face and point. Returns the relative
// width (upper - lower)/scale; margin is the smallest of (d* - lower) and
// (upper - d*) relative to scale.
struct EnclosureStats { long cases = 0, fails = 0; double maxWidth = 0, minMargin = 1e300; };

void check_enclosure(EnclosureStats& st, const MasterData& m, const BernsteinMaster<dstype>& bm,
                     const std::vector<double>& y, const double* x, double scale, const std::string& tag)
{
    DDFace face;
    face.n = m.npf;
    face.t.assign(m.xpf.begin(), m.xpf.end());
    face.y = y;
    const DD dstar = face.distance(x);
    dstype xi[2] = {0.5, 0}, xb[2], d;
    if (!ClosestPointOnFace(xi, xb, &d, x, y.data(), m.xpf.data(), m.nd, m.npf, m.elemtype, m.porder, 100, 1e-13))
        xi[0] = 0.5;
    const auto cb = face_distance_bounds(bm, y.data(), Int(m.npf), x, xi[0]);
    st.cases++;
    const bool okLo = !dd_less(dstar, DD(cb.lower));  // lower <= d*
    const bool okUp = !dd_less(DD(cb.upper), dstar);  // d* <= upper
    if (!cb.ok || !okLo || !okUp) {
        st.fails++;
        std::printf("[rank %d] FAIL enclosure %s: lower %.17g d* %.17g upper %.17g ok %d\n", worldRank, tag.c_str(),
                    double(cb.lower), dstar.hi, double(cb.upper), int(cb.ok));
        ++failures;
        return;
    }
    st.maxWidth = std::max(st.maxWidth, double(cb.upper - cb.lower) / scale);
    st.minMargin = std::min(st.minMargin, std::min((dstar - DD(cb.lower)).hi, (DD(cb.upper) - dstar).hi) / scale);
}

// Adversarial enclosure cases for both element types and p = 1..4:
//   - every nodal basis vector: x_j = h delta_jk, y_j = xpf_j (each Bernstein
//     column, curved with an oscillating image);
//   - nearly collinear control polygons: the same with h = 1e-13;
//   - points far, near the curve (1e-9), exactly at a node (d* = 0), on the chord
//     between two nodes (on or near the hull boundary), and inside the hull;
//   - translated and scaled geometry: y -> T + s y, x -> T + s x for
//     (T, s) = (0, 1), (1e6, 1e-3), (-3e4, 1e2), (0.5, 1e-6).
void case_enclosure_reference()
{
    if (worldRank != 0) return;
    EnclosureStats st;
    const double transforms[4][2] = {{0.0, 1.0}, {1e6, 1e-3}, {-3e4, 1e2}, {0.5, 1e-6}};
    for (int et = 0; et < 2; et++)
        for (int p = 1; p <= 4; p++) {
            MasterData m = make_master(et, p);
            const auto bm = make_bernstein_master(m.xpf.data(), Int(p));
            check(bm.ok, "enclosure: Bernstein master");
            const int n = m.npf;
            for (double h : {1.0, 1e-13})
                for (int k = 0; k < n; k++) {
                    std::vector<double> y0(2 * n);
                    for (int j = 0; j < n; j++) { y0[j] = (j == k) ? h : 0.0; y0[j + n] = m.xpf[j]; }
                    // reference points in the untransformed frame
                    std::vector<std::array<double, 2>> pts = {
                        {{2.0, 0.5}}, {{-1.5, 1.7}},                       // far
                        {{y0[k] + 1e-9, y0[k + n]}},                      // near a node
                        {{y0[k], y0[k + n]}},                             // exactly at a node: d* = 0
                        {{0.5 * (y0[0] + y0[1]), 0.5 * (y0[n] + y0[n + 1])}}, // on the chord of two nodes
                        {{0.25 * h, 0.4}}, {{-0.25 * h, 0.6}},            // inside / near the hull
                        {{1e-14, 0.3}}, {{-1e-14, 0.7}}};                 // near-degenerate hull boundary
                    for (const auto& tr : transforms) {
                        const double T = tr[0], s = tr[1];
                        std::vector<double> y(2 * n);
                        for (int j = 0; j < 2 * n; j++) y[j] = T + s * y0[j];
                        for (const auto& q : pts) {
                            const double x[2] = {T + s * q[0], T + s * q[1]};
                            const std::string tag = "et=" + std::to_string(et) + " p=" + std::to_string(p) +
                                " k=" + std::to_string(k) + " h=" + std::to_string(h) + " T=" + std::to_string(T) +
                                " s=" + std::to_string(s);
                            check_enclosure(st, m, bm, y, x, std::max(s, 1e-300), tag);
                        }
                    }
                }
            // curved parabola and translated copies (the threshold geometry)
            std::vector<double> yp(2 * n);
            for (int j = 0; j < n; j++) { yp[j] = (m.xpf[j] - 0.5) * (m.xpf[j] - 0.5); yp[j + n] = m.xpf[j]; }
            for (const auto& tr : transforms) {
                std::vector<double> y(2 * n);
                for (int j = 0; j < 2 * n; j++) y[j] = tr[0] + tr[1] * yp[j];
                for (double px : {-0.25, 0.0, 0.1, 0.3}) {
                    const double x[2] = {tr[0] + tr[1] * px, tr[0] + tr[1] * 0.4};
                    check_enclosure(st, m, bm, y, x, tr[1], "parabola");
                }
            }
        }
    check(st.fails == 0, "enclosure: lower <= d* <= upper in every case");
    std::printf("enclosure reference: %ld cases, %ld violations, max (upper - lower)/scale = %.3e, "
                "min margin/scale = %.3e\n", st.cases, st.fails, st.maxWidth, st.minMargin);
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    EXASIM_COMM_WORLD = MPI_COMM_WORLD;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (worldSize < 2 || (worldSize & (worldSize - 1)) != 0) {
        if (worldRank == 0) std::printf("nonmatching_donorsearch needs 2, 4, 8, ... ranks\n");
        MPI_Finalize();
        return 1;
    }

    case_conformal();
    case_matching();
    case_nonmatching();
    case_tie_mesh();
    case_near_ties();
    case_curved_adversarial();
    case_safeguard_stagnation();
    case_failed_candidate();
    case_basis_accuracy();
    case_tie_thresholds();
    case_enclosure_reference();

    int total = 0;
    MPI_Allreduce(&failures, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (worldRank == 0) {
        if (total == 0) std::printf("PASS nonmatching_donorsearch (%d ranks, up to %d per model)\n", worldSize, worldSize / 2);
        else std::printf("FAIL nonmatching_donorsearch: %d check(s)\n", total);
    }
    MPI_Finalize();
    return total == 0 ? 0 : 1;
}
