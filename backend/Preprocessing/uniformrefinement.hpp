#ifndef EXASIM_UNIFORMREFINEMENT_HPP
#define EXASIM_UNIFORMREFINEMENT_HPP

/*
 * uniformrefinement.hpp
 *
 * Uniform h-refinement of the input mesh, driven by the pdeapp key
 *
 *     uniformrefinementlevel = k;
 *
 * Each level splits every element into 2^nd children (red refinement for
 * simplices, octree/quadtree split for tensor elements), k times, before
 * boundary faces, partitioning and DG nodes are built. Everything downstream
 * therefore sees an ordinary mesh.
 *
 * Nodal fields read from file (xdgfile, udgfile, vdgfile, wdgfile) are
 * prolongated by evaluating the parent's degree-porder nodal interpolant at
 * each child's DG nodes. That is exact for the polynomial the parent carries,
 * so a curved xdg keeps its geometry and a converged IC keeps its field. When
 * xdg is available, new vertices are also placed on the curved parent map
 * (so boundary expressions still match them); otherwise they are the straight
 * midpoints of the parent.
 *
 * Element order is parent-major: child c of parent e is element e*nchild + c.
 *
 * Requires, from the including translation unit: error(), mkshape(),
 * localbasis() (backend/Preprocessing/makemaster.* or the text2code copy).
 * The MPI driver lives in uniformrefinementpar.hpp.
 */

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace exasim_uref {

// A vertex created by refinement is identified by the sorted ids of the parent
// vertices that support it (2 for an edge midpoint, 4 for a quad/hex face
// centre, nve for a tensor cell centre), padded with -1.
using Key = std::array<int, 8>;

inline std::uint64_t hashKey(const Key& k)
{
    std::uint64_t h = 1469598103934665603ull;          // FNV-1a
    for (int v : k) { h ^= static_cast<std::uint32_t>(v); h *= 1099511628211ull; }
    return h;
}

struct KeyHash {
    std::size_t operator()(const Key& k) const noexcept { return static_cast<std::size_t>(hashKey(k)); }
};

// One level of refinement of the reference element.
struct RefinementTemplate {
    int nd = 0, elemtype = 0, nve = 0;
    int nlat = 0;               // lattice points: parent vertices + new vertices
    int nchild = 0;
    std::vector<double> xlat;   // [nlat x nd]   reference coordinates
    std::vector<int> nsup;      // [nlat]        number of supporting parent vertices
    std::vector<int> sup;       // [nve x nlat]  supporting local parent vertices
    std::vector<int> child;     // [nve x nchild] lattice point of each child vertex
};

inline RefinementTemplate makeRefinementTemplate(int nd, int elemtype, int nve)
{
    RefinementTemplate T;
    T.nd = nd; T.elemtype = elemtype; T.nve = nve;

    std::vector<double> x;
    auto addpt = [&](double a, double b, double c) {
        const double xyz[3] = {a, b, c};
        for (int d = 0; d < nd; d++) x.push_back(xyz[d]);
        T.nlat++;
    };
    auto addchild = [&](std::initializer_list<int> v) { T.child.insert(T.child.end(), v); T.nchild++; };

    if (nd == 1 || elemtype == 1) {
        // Lattice {0, 1/2, 1}^nd, x fastest; vertex ordering follows localbasis
        // (counter-clockwise, and bottom layer before top layer for hexes).
        const int ny = (nd >= 2) ? 3 : 1, nz = (nd == 3) ? 3 : 1;
        for (int k = 0; k < nz; k++)
            for (int j = 0; j < ny; j++)
                for (int i = 0; i < 3; i++) addpt(0.5*i, 0.5*j, 0.5*k);
        auto lat = [](int i, int j, int k) { return i + 3*(j + 3*k); };
        if (nd == 1) {
            addchild({0, 1});
            addchild({1, 2});
        } else if (nd == 2) {
            for (int cy = 0; cy < 2; cy++)
                for (int cx = 0; cx < 2; cx++)
                    addchild({lat(cx, cy, 0), lat(cx+1, cy, 0), lat(cx+1, cy+1, 0), lat(cx, cy+1, 0)});
        } else {
            for (int cz = 0; cz < 2; cz++)
                for (int cy = 0; cy < 2; cy++)
                    for (int cx = 0; cx < 2; cx++)
                        addchild({lat(cx, cy, cz),   lat(cx+1, cy, cz),   lat(cx+1, cy+1, cz),   lat(cx, cy+1, cz),
                                  lat(cx, cy, cz+1), lat(cx+1, cy, cz+1), lat(cx+1, cy+1, cz+1), lat(cx, cy+1, cz+1)});
        }
    } else if (nd == 2) {
        // triangle: v0 v1 v2, then midpoints m01 m12 m20
        addpt(0, 0, 0); addpt(1, 0, 0); addpt(0, 1, 0);
        addpt(0.5, 0, 0); addpt(0.5, 0.5, 0); addpt(0, 0.5, 0);
        addchild({0, 3, 5});
        addchild({3, 1, 4});
        addchild({5, 4, 2});
        addchild({3, 4, 5});
    } else {
        // tetrahedron: v0..v3, then m01 m02 m03 m12 m13 m23; Bey's red refinement
        // (interior octahedron split along m02-m13)
        addpt(0, 0, 0); addpt(1, 0, 0); addpt(0, 1, 0); addpt(0, 0, 1);
        addpt(0.5, 0, 0); addpt(0, 0.5, 0); addpt(0, 0, 0.5);
        addpt(0.5, 0.5, 0); addpt(0.5, 0, 0.5); addpt(0, 0.5, 0.5);
        addchild({0, 4, 5, 6});
        addchild({4, 1, 7, 8});
        addchild({5, 7, 2, 9});
        addchild({6, 8, 9, 3});
        addchild({4, 5, 6, 8});
        addchild({4, 5, 7, 8});
        addchild({5, 6, 8, 9});
        addchild({5, 7, 8, 9});
    }

    if ((int)T.child.size() != nve*T.nchild)
        error("uniformrefinement: unsupported element (nd=" + std::to_string(nd) + ", nve=" +
              std::to_string(nve) + ", elemtype=" + std::to_string(elemtype) + ").\n");

    // Reorder x from point-major to the column-major [nlat x nd] layout.
    T.xlat.resize((std::size_t)T.nlat*nd);
    for (int l = 0; l < T.nlat; l++)
        for (int d = 0; d < nd; d++) T.xlat[l + T.nlat*d] = x[(std::size_t)l*nd + d];

    // A lattice point is supported by the parent vertices whose linear shape
    // function is nonzero there.
    std::vector<double> phi((std::size_t)T.nlat*nve);
    double dummy = 0.0;
    localbasis(phi.data(), &dummy, T.xlat.data(), &dummy, nd, elemtype, T.nlat, 0);
    T.nsup.assign(T.nlat, 0);
    T.sup.assign((std::size_t)nve*T.nlat, -1);
    for (int l = 0; l < T.nlat; l++)
        for (int v = 0; v < nve; v++)
            if (phi[l + T.nlat*v] > 1e-12) T.sup[T.nsup[l]++ + nve*l] = v;

    // Children of a simplex must keep the positive orientation of the master element.
    if (elemtype == 0 && nd >= 2) {
        for (int c = 0; c < T.nchild; c++) {
            const int* cv = &T.child[nve*c];
            double e[3][3] = {{0}};
            for (int a = 1; a <= nd; a++)
                for (int d = 0; d < nd; d++)
                    e[a-1][d] = T.xlat[cv[a] + T.nlat*d] - T.xlat[cv[0] + T.nlat*d];
            const double det = (nd == 2) ? e[0][0]*e[1][1] - e[0][1]*e[1][0]
                : e[0][0]*(e[1][1]*e[2][2] - e[1][2]*e[2][1])
                - e[0][1]*(e[1][0]*e[2][2] - e[1][2]*e[2][0])
                + e[0][2]*(e[1][0]*e[2][1] - e[1][1]*e[2][0]);
            if (det < 0) std::swap(T.child[nve*c + nd - 1], T.child[nve*c + nd]);
        }
    }
    return T;
}

struct RefinementProlongation {
    int npe = 0;
    std::vector<double> P;      // [npe x npe x nchild]: child node i <- parent node a
    std::vector<double> Slat;   // [npe x nlat]: parent nodal basis at the lattice points
};

// xpe is the master element's [npe x nd] node coordinates (Master::xpe).
inline RefinementProlongation makeRefinementProlongation(const RefinementTemplate& T,
                                                         const std::vector<double>& xpe,
                                                         int porder, int npe)
{
    const int nd = T.nd, nve = T.nve, nlat = T.nlat;
    RefinementProlongation R;
    R.npe = npe;

    if ((int)xpe.size() < npe*nd) error("uniformrefinement: master nodes are not initialized.\n");
    std::vector<double> plocal(xpe.begin(), xpe.begin() + (std::size_t)npe*nd);

    std::vector<double> phielem((std::size_t)npe*nve);
    double dummy = 0.0;
    localbasis(phielem.data(), &dummy, plocal.data(), &dummy, nd, T.elemtype, npe, 0);

    // Child c's DG node i sits at reference point sum_v phielem(i,v) * X_c(v) in
    // the parent; the prolongation row is the parent basis evaluated there.
    R.P.assign((std::size_t)npe*npe*T.nchild, 0.0);
    std::vector<double> pts((std::size_t)npe*nd), shap;
    for (int c = 0; c < T.nchild; c++) {
        std::fill(pts.begin(), pts.end(), 0.0);
        for (int d = 0; d < nd; d++)
            for (int v = 0; v < nve; v++) {
                const double xv = T.xlat[T.child[v + nve*c] + nlat*d];
                for (int i = 0; i < npe; i++) pts[i + npe*d] += phielem[i + npe*v]*xv;
            }
        shap.assign((std::size_t)npe*npe*(nd+1), 0.0);
        mkshape(shap, plocal, pts, npe, T.elemtype, porder, nd, npe);
        for (int i = 0; i < npe; i++) {
            double sum = 0.0;
            for (int a = 0; a < npe; a++) {
                R.P[i + npe*(a + npe*c)] = shap[a + npe*i];
                sum += shap[a + npe*i];
            }
            if (std::abs(sum - 1.0) > 1e-8)
                error("uniformrefinement: prolongation is not a partition of unity (row sum " +
                      std::to_string(sum) + ").\n");
        }
    }

    std::vector<double> xlat = T.xlat;
    shap.assign((std::size_t)npe*nlat*(nd+1), 0.0);
    mkshape(shap, plocal, xlat, nlat, T.elemtype, porder, nd, npe);
    R.Slat.assign(shap.begin(), shap.begin() + (std::size_t)npe*nlat);
    return R;
}

// Output of one level of refinement on one process.
struct RefinedLevel {
    std::vector<double> p;      // [nd x (np + nnew)]: old vertices first, then new ones
    std::vector<int> t;         // [nve x ne*nchild]
    std::vector<Key> keys;      // [nnew] sorted (global) ids of supporting parent vertices
    std::vector<char> interior; // [nnew] 1 if the vertex cannot be shared with another element
    int nnew = 0;
};

// Split every element once. gid maps local vertex -> global id (nullptr in
// serial); keys use global ids so ranks agree on shared vertices. xdg, if not
// null, is the [npe x nd x ne] curved geometry used to place new vertices.
inline void refineConnectivity(RefinedLevel& out, const RefinementTemplate& T, const RefinementProlongation& R,
                               const std::vector<double>& p, const std::vector<int>& t, const int* gid,
                               int np, int ne, const double* xdg)
{
    const int nd = T.nd, nve = T.nve, nlat = T.nlat, nchild = T.nchild, npe = R.npe;

    out.p.assign(p.begin(), p.begin() + (std::size_t)nd*np);
    out.t.assign((std::size_t)nve*ne*nchild, -1);
    out.keys.clear();
    out.interior.clear();
    out.nnew = 0;

    std::unordered_map<Key, int, KeyHash> index;
    index.reserve((std::size_t)ne*(nlat - nve));

    std::vector<int> latid(nlat);
    std::vector<std::pair<int, int>> support(nve);   // (global id, local id)
    for (int e = 0; e < ne; e++) {
        const int* te = &t[(std::size_t)nve*e];
        for (int l = 0; l < nlat; l++) {
            const int ns = T.nsup[l];
            if (ns == 1) { latid[l] = te[T.sup[nve*l]]; continue; }

            for (int s = 0; s < ns; s++) {
                const int lv = te[T.sup[s + nve*l]];
                support[s] = {gid ? gid[lv] : lv, lv};
            }
            std::sort(support.begin(), support.begin() + ns);
            Key key; key.fill(-1);
            for (int s = 0; s < ns; s++) key[s] = support[s].first;

            auto ins = index.try_emplace(key, np + out.nnew);
            if (ins.second) {
                for (int d = 0; d < nd; d++) {
                    double xd = 0.0;
                    if (xdg) {
                        const double* xe = &xdg[(std::size_t)npe*(d + (std::size_t)nd*e)];
                        for (int a = 0; a < npe; a++) xd += R.Slat[a + npe*l]*xe[a];
                    } else {
                        // straight midpoint / face or cell centre, summed in global-id
                        // order so every rank computes bit-identical coordinates
                        for (int s = 0; s < ns; s++) xd += p[d + (std::size_t)nd*support[s].second];
                        xd /= ns;
                    }
                    out.p.push_back(xd);
                }
                out.keys.push_back(key);
                out.interior.push_back(ns == nve);
                out.nnew++;
            }
            latid[l] = ins.first->second;
        }
        for (int c = 0; c < nchild; c++)
            for (int v = 0; v < nve; v++)
                out.t[v + (std::size_t)nve*((std::size_t)e*nchild + c)] = latid[T.child[v + nve*c]];
    }
}

// [npe x nc x ne] -> [npe x nc x ne*nchild]
inline void prolongField(std::vector<double>& f, int npe, int ne, int nchild, const std::vector<double>& P,
                         const char* name)
{
    if (f.empty()) return;
    const std::size_t blk = (std::size_t)npe*ne;
    if (blk == 0 || f.size() % blk != 0)
        error(std::string("uniformrefinement: ") + name + " does not have npe*nc*ne entries (npe=" +
              std::to_string(npe) + ", ne=" + std::to_string(ne) + ").\n");
    const std::size_t nc = f.size()/blk;
    std::vector<double> g(f.size()*nchild, 0.0);
    for (std::size_t e = 0; e < (std::size_t)ne; e++)
        for (int c = 0; c < nchild; c++) {
            const double* Pc = &P[(std::size_t)npe*npe*c];
            for (std::size_t m = 0; m < nc; m++) {
                const double* fe = &f[npe*(m + nc*e)];
                double* ge = &g[npe*(m + nc*(e*nchild + c))];
                for (int a = 0; a < npe; a++) {
                    const double fa = fe[a];
                    for (int i = 0; i < npe; i++) ge[i] += Pc[i + npe*a]*fa;
                }
            }
        }
    f.swap(g);
}

inline void repeatPerChild(std::vector<int>& v, int nchild)
{
    std::vector<int> w(v.size()*nchild);
    for (std::size_t e = 0; e < v.size(); e++)
        for (int c = 0; c < nchild; c++) w[e*nchild + c] = v[e];
    v.swap(w);
}

inline void setFieldDims(std::vector<int>& dims, int ne)
{
    if (dims.size() >= 3) dims[2] = ne;
}

inline void checkRefinedSize(long long ne, int nchild, int nlevel)
{
    long long n = ne;
    for (int l = 0; l < nlevel; l++) n *= nchild;
    if (n > INT_MAX)
        error("uniformrefinementlevel = " + std::to_string(nlevel) + " gives " + std::to_string(n) +
              " elements, which overflows the int element index.\n");
}

} // namespace exasim_uref

// Serial driver: refine mesh.p/t in place and prolongate the file-supplied
// nodal fields. Call after initializeMaster and before buildMesh.
template <class MeshT, class PDET, class MasterT>
inline void uniformRefineMesh(MeshT& mesh, PDET& pde, const MasterT& master, int rank = 0)
{
    using namespace exasim_uref;

    const int nlevel = pde.uniformrefinementlevel;
    if (nlevel <= 0) return;
    if (!mesh.uhat.empty())
        error("uniformrefinementlevel > 0 cannot prolongate the face-based uhatfile.\n");

    const RefinementTemplate T = makeRefinementTemplate(mesh.nd, mesh.elemtype, mesh.nve);
    const RefinementProlongation R = makeRefinementProlongation(T, master.xpe, master.porder, master.npe);
    checkRefinedSize(mesh.ne, T.nchild, nlevel);
    if (!mesh.elem2cpu.empty() && (int)mesh.elem2cpu.size() != mesh.ne)
        error("uniformrefinement: partition array does not have one entry per element.\n");

    const int ne0 = mesh.ne, np0 = mesh.np;
    RefinedLevel L;
    for (int level = 0; level < nlevel; level++) {
        refineConnectivity(L, T, R, mesh.p, mesh.t, nullptr, mesh.np, mesh.ne,
                           mesh.xdg.empty() ? nullptr : mesh.xdg.data());
        prolongField(mesh.xdg, R.npe, mesh.ne, T.nchild, R.P, "xdg");
        prolongField(mesh.udg, R.npe, mesh.ne, T.nchild, R.P, "udg");
        prolongField(mesh.vdg, R.npe, mesh.ne, T.nchild, R.P, "vdg");
        prolongField(mesh.wdg, R.npe, mesh.ne, T.nchild, R.P, "wdg");
        if (!mesh.elem2cpu.empty()) repeatPerChild(mesh.elem2cpu, T.nchild);
        mesh.p.swap(L.p);
        mesh.t.swap(L.t);
        mesh.np += L.nnew;
        mesh.ne *= T.nchild;
    }
    setFieldDims(mesh.xdgdims, mesh.ne);
    setFieldDims(mesh.udgdims, mesh.ne);
    setFieldDims(mesh.vdgdims, mesh.ne);
    setFieldDims(mesh.wdgdims, mesh.ne);
    pde.np = mesh.np;
    pde.ne = mesh.ne;

    if (rank == 0)
        std::printf("uniformrefinementlevel = %d: ne %d -> %d, np %d -> %d\n",
                    nlevel, ne0, mesh.ne, np0, mesh.np);
}

#endif
