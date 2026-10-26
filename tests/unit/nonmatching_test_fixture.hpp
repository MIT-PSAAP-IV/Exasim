// Shared synthetic 2D mesh fixture for current non-matching setup tests.
#ifndef EXASIM_TESTS_NONMATCHING_TEST_FIXTURE_HPP
#define EXASIM_TESTS_NONMATCHING_TEST_FIXTURE_HPP

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
#include "../../backend/Discretization/nonmatchingpreprocess.hpp"

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
    const std::string marker = "/tests/unit/nonmatching_test_fixture.hpp";
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
    rs.mesh.szperm = static_cast<Int>(rs.perm.size());
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

unsigned long long checksum(const std::map<Key, std::vector<double>>& values)
{
    unsigned long long hash = 1469598103934665603ULL;
    for (const auto& entry : values)
        for (double value : entry.second) {
            unsigned long long bits;
            std::memcpy(&bits, &value, sizeof(bits));
            for (int byte = 0; byte < 8; ++byte) {
                hash ^= (bits >> (8 * byte)) & 0xffULL;
                hash *= 1099511628211ULL;
            }
        }
    return hash;
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

} // namespace

#endif
