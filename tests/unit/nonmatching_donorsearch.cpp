// Phase 2 of the non-matching interface solver: donor search
// (backend/Discretization/nonmatchingsetup.hpp, docs/nonmatching/nonmatching_multiphysics.tex,
// eqs. (3), (56), (60), Section 3.2).
//
// MPI test. Run with an even number of ranks 2k: world ranks [0, k) hold model 1,
// ranks [k, 2k) hold model 2 (fileoffset = k), as in ExasimSolver.cpp. Each rank
// builds its own partition of synthetic 2D meshes as backend structs and calls the
// real entry point setup_nonmatching_interface. Rank 0 also runs donor_search
// serially on the global meshes; every distributed donor tuple, compared by global
// indices (owner global face, g), must equal the serial one exactly. Registered with
// k = 1, 2, 4, so the three partitions are compared with the same reference.
//
// Cases:
//   1. matching: geometry of examples/Poisson/coupledproblem (two 8x16 quad meshes of
//      [0,0.5]x[0,1] and [0.5,1]x[0,1], p = 3). Donor = conformal partner, dist < 1e-12.
//   2. non-matching: triangles, p = 2, 12 vs 7 faces with non-uniform rows, curved and
//      perturbed model-2 interface (gaps and overlaps), model 2 shorter than model 1.
//   3. tie: a model-2 vertex at the height of a model-1 Gauss point, offset by a gap;
//      the point is equidistant to two faces, the smaller global index must win.
//   4. interfacetype = 0: the entry point does nothing.

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
#include <cmath>
#include <cstdio>
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
    m.shapfgt.assign(static_cast<std::size_t>(m.ngf * m.npf), 0.0);
    std::vector<dstype> psi(m.npf);
    for (int g = 0; g < m.ngf; g++) {
        dstype xi[2] = {m.gpf[g], 0.0};
        EvaluateFaceShape(psi.data(), xi, m.xpf.data(), m.nd, m.npf, elemtype, porder);
        for (int j = 0; j < m.npf; j++) m.shapfgt[g + m.ngf * j] = psi[j];
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

struct CaseResult {
    std::map<Key, std::vector<double>> dist;   // distributed tuples (rank 0)
    std::map<Key, std::vector<double>> serial; // serial reference (rank 0)
    InterfaceFaceSet<dstype, Int> g1, g2;      // global faces (rank 0)
};

// Runs one case on all ranks; on rank 0 compares the distributed tuples with the serial reference.
CaseResult run_case(const std::string& name, const ModelMesh& m1, const ModelMesh& m2, MasterData& m)
{
    const int k = worldSize / 2;
    const int model = (worldRank < k) ? 0 : 1;
    const int part = worldRank - model * k;
    const int fileoffset = model * k;

    RankStructs rs;
    build_rank_structs(rs, model == 0 ? m1 : m2, m, part, k, 1);

    // The conformal builder must return before its exact match when interfacetype = 1.
    exasim::interfacepartition::build_runtime_interface_partition(rs.app, rs.master, rs.mesh, rs.sol,
            static_cast<Int>(worldSize), static_cast<Int>(worldRank), static_cast<Int>(fileoffset));
    check(rs.mesh.faceperm == nullptr && rs.mesh.nbintf == nullptr, name + ": conformal arrays untouched");

    nonmatchingdataT<dstype, Int> nm;
    DonorSearchResult<dstype, Int> res;
    const bool ran = setup_nonmatching_interface(rs.app, rs.master, rs.mesh, rs.sol,
            static_cast<Int>(worldSize), static_cast<Int>(worldRank), static_cast<Int>(fileoffset), nm, &res);
    check(ran && nm.interfacetype == 1, name + ": setup ran");
    check(res.direction == model, name + ": direction");
    check_owner_side(nm.dir[model], res, name);
    check(nm.dir[1 - model].nownfaces == 0, name + ": other direction empty on owner side");
    free_nonmatching_data(nm);

    std::vector<double> mine;
    pack(mine, res);
    int n = static_cast<int>(mine.size());
    std::vector<int> counts(worldSize), displs(worldSize, 0);
    MPI_Gather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, EXASIM_COMM_WORLD);
    int total = 0;
    for (int r = 0; r < worldSize; r++) { displs[r] = total; total += counts[r]; }
    std::vector<double> all(worldRank == 0 ? total : 0);
    MPI_Gatherv(mine.data(), n, MPI_DOUBLE, all.data(), counts.data(), displs.data(), MPI_DOUBLE, 0,
                EXASIM_COMM_WORLD);

    CaseResult out;
    if (worldRank != 0) return out;
    out.dist = to_map(all);

    out.g1 = global_faces(m1, m, 0, k);
    out.g2 = global_faces(m2, m, 1, k);
    std::vector<double> ref;
    for (int dir = 0; dir < 2; dir++) {
        const auto& own = dir == 0 ? out.g1 : out.g2;
        const auto& cand = dir == 0 ? out.g2 : out.g1;
        DonorSearchResult<dstype, Int> sr;
        sr.direction = dir;
        own_quadrature_points(sr, own, m.shapfgt.data(), static_cast<Int>(m.ngf));
        donor_search(sr, cand, m.xpf.data(), static_cast<Int>(m.elemtype), static_cast<Int>(m.porder));
        pack(ref, sr);

        // Brute force: the donor distance is the minimum over all faces, and the donor
        // is the smallest global index among the faces within the tie tolerance.
        DonorSearchParams prm;
        dstype gmin[2] = {1e300, 1e300}, gmax[2] = {-1e300, -1e300};
        for (std::size_t i = 0; i < cand.nodes.size(); i++) {
            const int dd = static_cast<int>((i / m.npf) % m.nd);
            gmin[dd] = std::min(gmin[dd], cand.nodes[i]);
            gmax[dd] = std::max(gmax[dd], cand.nodes[i]);
        }
        const dstype tol = prm.tieRelTol * std::hypot(gmax[0] - gmin[0], gmax[1] - gmin[1]);
        for (int p = 0; p < sr.npts; p++) {
            const dstype xp[2] = {sr.x[p], sr.x[p + sr.npts]};
            std::vector<dstype> dm(cand.size());
            dstype dmin = 1e300;
            for (int c = 0; c < cand.size(); c++) {
                dstype xi[2], xb[2];
                ClosestPointOnFace(xi, xb, &dm[c], xp, &cand.nodes[m.npf * m.nd * c], m.xpf.data(), m.nd,
                                   m.npf, m.elemtype, m.porder, prm.maxProjectionIterations,
                                   prm.projectionTolerance);
                dmin = std::min(dmin, dm[c]);
            }
            Int fbest = -1;
            for (int c = 0; c < cand.size(); c++)
                if (dm[c] <= dmin + tol && (fbest < 0 || cand.global[c] < fbest)) fbest = cand.global[c];
            check(sr.f[p] == fbest, name + ": brute-force donor face");
            check(std::abs(sr.dist[p] - dmin) <= tol, name + ": brute-force distance");
        }
    }
    out.serial = to_map(ref);

    check(out.dist.size() == out.serial.size(), name + ": number of points");
    long mismatch = 0;
    for (const auto& kv : out.serial) {
        auto it = out.dist.find(kv.first);
        if (it == out.dist.end() || it->second != kv.second) mismatch++;
    }
    check(mismatch == 0, name + ": distributed tuples equal serial reference (" + std::to_string(mismatch) + " differ)");
    return out;
}

std::vector<dstype> uniform(dstype a, dstype b, int n)
{
    std::vector<dstype> v(n + 1);
    for (int i = 0; i <= n; i++) v[i] = a + (b - a) * dstype(i) / dstype(n);
    v[n] = b;
    return v;
}

void case_matching()
{
    MasterData m = make_master(1, 3);
    ModelMesh m1, m2;
    m1.xs = uniform(0.0, 0.5, 8);  m1.ys = uniform(0.0, 1.0, 16); m1.xiface = 0.5; m1.xfar = 0.0;
    m2.xs = uniform(0.5, 1.0, 8);  m2.ys = uniform(0.0, 1.0, 16); m2.xiface = 0.5; m2.xfar = 1.0;
    CaseResult r = run_case("matching", m1, m2, m);
    if (worldRank != 0) return;

    // Conformal partner by exact matching of the face end points.
    auto partner = [&](const InterfaceFaceSet<dstype, Int>& own, int l, const InterfaceFaceSet<dstype, Int>& other) {
        auto ends = [&](const InterfaceFaceSet<dstype, Int>& fs, int f) {
            dstype lo = 1e300, hi = -1e300;
            for (int j = 0; j < m.npf; j++) {
                lo = std::min(lo, fs.nodes[j + m.npf + m.npf * m.nd * f]);
                hi = std::max(hi, fs.nodes[j + m.npf + m.npf * m.nd * f]);
            }
            return std::make_pair(lo, hi);
        };
        const auto e = ends(own, l);
        Int found = -1;
        for (int c = 0; c < other.size(); c++) {
            const auto o = ends(other, c);
            if (std::abs(o.first - e.first) < 1e-14 && std::abs(o.second - e.second) < 1e-14) found = other.global[c];
        }
        return found;
    };
    for (int dir = 0; dir < 2; dir++) {
        const auto& own = dir == 0 ? r.g1 : r.g2;
        const auto& oth = dir == 0 ? r.g2 : r.g1;
        for (int l = 0; l < own.size(); l++) {
            const Int pf = partner(own, l, oth);
            check(pf >= 0, "matching: partner exists");
            for (int g = 0; g < m.ngf; g++) {
                const auto& t = r.dist.at(Key((long(dir) << 40) + long(own.global[l]), g));
                check(Int(t[5]) == pf, "matching: donor = conformal partner");
                check(t[7] < 1e-12, "matching: dist < 1e-12");
            }
        }
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
        K[std::make_pair(long(t[0]), long(t[1]))].push_back(long(t[5]));
    }
    for (auto& kv : K) {
        std::sort(kv.second.begin(), kv.second.end());
        if (std::unique(kv.second.begin(), kv.second.end()) - kv.second.begin() > 1) multiK++;
    }
    // Points of model 1 above y = 0.9 lie beyond the edge of Gamma^2.
    for (const auto& kv : r.dist)
        if (kv.second[0] == 0 && kv.second[7] > 0.02) beyond++;
    check(off > 0, "nonmatching: some points off their donor face");
    check(multiK > 0, "nonmatching: some faces with |K_l| > 1");
    check(beyond > 0, "nonmatching: points beyond the edge of Gamma^2");
    std::printf("nonmatching: %zu points, %ld off donor face, %ld beyond edge, %ld faces with |K_l|>1\n",
                r.dist.size(), off, beyond, multiK);
}

void case_tie()
{
    MasterData m = make_master(1, 1);
    // y of Gauss point 0 of the first model-1 face (cell row [0, 0.25]).
    dstype ystar = 0.0;
    for (int j = 0; j < m.npf; j++) ystar += m.shapfgt[0 + m.ngf * j] * (j == 0 ? 0.0 : 0.25);
    // The face node order is not assumed: both Gauss heights ystar and 0.25 - ystar
    // become model-2 vertices, so both Gauss points of that face are tied.
    ModelMesh m1, m2;
    m1.xs = uniform(0.0, 0.5, 2); m1.ys = uniform(0.0, 1.0, 4); m1.xiface = 0.5; m1.xfar = 0.0;
    const dstype gap = 0.01;
    m2.xs = uniform(0.5 + gap, 1.0, 2); m2.xiface = 0.5 + gap; m2.xfar = 1.0;
    const dstype y1 = 0.25 - ystar; // the other Gauss point of the same face
    m2.ys = {0.0, std::min(ystar, y1), std::max(ystar, y1), 0.6, 1.0};
    CaseResult r = run_case("tie", m1, m2, m);
    if (worldRank != 0) return;

    // Both Gauss points of the model-1 face on row 0 project onto a model-2 vertex
    // shared by two faces: equidistant, the smaller global index must be chosen.
    int ties = 0;
    for (int l = 0; l < r.g1.size(); l++) {
        if (r.g1.nodes[m.npf + m.npf * m.nd * l] > 0.25 + 1e-12 || r.g1.nodes[1 + m.npf + m.npf * m.nd * l] > 0.25 + 1e-12)
            continue;
        for (int g = 0; g < m.ngf; g++) {
            const auto& t = r.dist.at(Key(long(r.g1.global[l]), g));
            // Faces of model 2 within the tie tolerance of the chosen distance.
            std::vector<Int> tied;
            for (int c = 0; c < r.g2.size(); c++) {
                dstype xi[2], xb[2], d;
                dstype x[2];
                // Owner point coordinates from the face nodes and shapfgt.
                x[0] = 0; x[1] = 0;
                for (int j = 0; j < m.npf; j++) {
                    x[0] += m.shapfgt[g + m.ngf * j] * r.g1.nodes[j + m.npf * m.nd * l];
                    x[1] += m.shapfgt[g + m.ngf * j] * r.g1.nodes[j + m.npf + m.npf * m.nd * l];
                }
                ClosestPointOnFace(xi, xb, &d, x, &r.g2.nodes[m.npf * m.nd * c], m.xpf.data(), m.nd, m.npf,
                                   m.elemtype, m.porder, 100, 1e-13);
                if (std::abs(d - t[7]) <= 1e-12) tied.push_back(r.g2.global[c]);
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
    std::printf("tie: %d tied points resolved to the smaller global face index\n", ties);
}

void case_interfacetype0()
{
    MasterData m = make_master(1, 1);
    ModelMesh mm;
    mm.xs = uniform(0.0, 0.5, 2); mm.ys = uniform(0.0, 1.0, 4); mm.xiface = 0.5; mm.xfar = 0.0;
    RankStructs rs;
    build_rank_structs(rs, mm, m, 0, 1, 0);
    nonmatchingdataT<dstype, Int> nm;
    const bool ran = setup_nonmatching_interface(rs.app, rs.master, rs.mesh, rs.sol,
            static_cast<Int>(worldSize), static_cast<Int>(worldRank), static_cast<Int>(0), nm);
    check(!ran && nm.interfacetype == 0 && nm.dir[0].nownfaces == 0 && nm.dir[0].ownface == nullptr &&
          nm.dir[0].direction == -1 && nm.dir[1].direction == -1, "interfacetype 0: no-op");
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    EXASIM_COMM_WORLD = MPI_COMM_WORLD;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (worldSize < 2 || worldSize % 2 != 0) {
        if (worldRank == 0) std::printf("nonmatching_donorsearch needs an even number of ranks >= 2\n");
        MPI_Finalize();
        return 1;
    }

    case_interfacetype0();
    case_matching();
    case_nonmatching();
    case_tie();

    int total = 0;
    MPI_Allreduce(&failures, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (worldRank == 0) {
        if (total == 0) std::printf("PASS nonmatching_donorsearch (%d ranks, %d per model)\n", worldSize, worldSize / 2);
        else std::printf("FAIL nonmatching_donorsearch: %d check(s)\n", total);
    }
    MPI_Finalize();
    return total == 0 ? 0 : 1;
}
