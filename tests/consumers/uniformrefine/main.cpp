// uniformrefine -- regression for the pdeapp key `uniformrefinementlevel = k`
// (uniformRefineMesh, backend/Preprocessing/uniformrefinement.hpp). For tri, quad,
// tet and hex meshes of the unit square/cube, porder 1..3, k = 1 and 2:
//   - element and vertex counts match the uniformly refined structured mesh, and
//     no two vertices coincide;
//   - the refined mesh is conforming (every face shared by at most two elements,
//     the boundary has the refined number of faces), every element is positively
//     oriented, and the total volume is 1;
//   - fields are prolongated exactly: a straight xdg equals compute_dgnodes on the
//     refined p/t, a degree-porder udg is reproduced at the child nodes, and for a
//     degree-porder curved xdg the new vertices and child nodes lie on the curved map.
// Self-checking: 0 on success, nonzero on failure. No pdeapp.txt.

#include <exasim/operators.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool ok, const char* label, const std::string& what)
{
    if (!ok) { std::printf("[uniformrefine]   %s FAIL: %s\n", label, what.c_str()); failures++; }
}

static double signedVolume(int nd, int elemtype, const std::vector<double>& p, const int* v)
{
    auto x = [&](int a, int d) { return p[d + nd*v[a]]; };
    if (nd == 2 && elemtype == 0)
        return 0.5*((x(1,0)-x(0,0))*(x(2,1)-x(0,1)) - (x(2,0)-x(0,0))*(x(1,1)-x(0,1)));
    if (nd == 2) {  // shoelace over the counter-clockwise quad
        double s = 0.0;
        for (int a = 0; a < 4; a++) s += x(a,0)*x((a+1)%4,1) - x((a+1)%4,0)*x(a,1);
        return 0.5*s;
    }
    const int b = (elemtype == 0) ? 2 : 3, c = (elemtype == 0) ? 3 : 4;  // tet: 1,2,3; hex: 1,3,4
    double e[3][3];
    const int ax[3] = {1, b, c};
    for (int r = 0; r < 3; r++)
        for (int d = 0; d < 3; d++) e[r][d] = x(ax[r], d) - x(0, d);
    const double det = e[0][0]*(e[1][1]*e[2][2] - e[1][2]*e[2][1])
                     - e[0][1]*(e[1][0]*e[2][2] - e[1][2]*e[2][0])
                     + e[0][2]*(e[1][0]*e[2][1] - e[1][1]*e[2][0]);
    return (elemtype == 0) ? det/6.0 : det;   // children of axis-aligned boxes stay boxes
}

// Unit square/cube with n cells per direction, in Exasim's local vertex ordering.
static void boxMesh(int nd, int elemtype, int n, std::vector<double>& p, std::vector<int>& t, int& np, int& nve)
{
    const int nv = n + 1;
    np = (nd == 2) ? nv*nv : nv*nv*nv;
    p.assign((size_t)nd*np, 0.0);
    for (int j = 0; j < np; j++) {
        p[0 + nd*j] = (double)(j % nv)/n;
        p[1 + nd*j] = (double)((j/nv) % nv)/n;
        if (nd == 3) p[2 + nd*j] = (double)(j/(nv*nv))/n;
    }
    t.clear();
    if (nd == 2) {
        nve = (elemtype == 1) ? 4 : 3;
        for (int iy = 0; iy < n; iy++)
            for (int ix = 0; ix < n; ix++) {
                const int a = iy*nv + ix, b = a + 1, c = a + nv + 1, d = a + nv;
                if (elemtype == 1) t.insert(t.end(), {a, b, c, d});
                else               t.insert(t.end(), {a, b, c, a, c, d});
            }
        return;
    }
    nve = (elemtype == 1) ? 8 : 4;
    static const int kuhn[6][4] = {{0,1,2,6},{0,2,3,6},{0,3,7,6},{0,7,4,6},{0,4,5,6},{0,5,1,6}};
    for (int iz = 0; iz < n; iz++)
        for (int iy = 0; iy < n; iy++)
            for (int ix = 0; ix < n; ix++) {
                const int b0 = iz*nv*nv + iy*nv + ix, up = nv*nv;
                const int c[8] = {b0, b0+1, b0+nv+1, b0+nv, b0+up, b0+up+1, b0+up+nv+1, b0+up+nv};
                if (elemtype == 1) { t.insert(t.end(), c, c + 8); continue; }
                for (const auto& k : kuhn) {
                    int v[4] = {c[k[0]], c[k[1]], c[k[2]], c[k[3]]};
                    if (signedVolume(3, 0, p, v) < 0) std::swap(v[2], v[3]);
                    t.insert(t.end(), v, v + 4);
                }
            }
}

static const std::vector<std::vector<int>>& localFaces(int nd, int elemtype)
{
    static const std::vector<std::vector<int>> tri  = {{1,2},{2,0},{0,1}};
    static const std::vector<std::vector<int>> quad = {{0,1},{1,2},{2,3},{3,0}};
    static const std::vector<std::vector<int>> tet  = {{1,2,3},{0,3,2},{0,1,3},{0,2,1}};
    static const std::vector<std::vector<int>> hex  = {{0,3,2,1},{4,5,6,7},{0,1,5,4},{2,3,7,6},{1,2,6,5},{3,0,4,7}};
    if (nd == 2) return elemtype ? quad : tri;
    return elemtype ? hex : tet;
}

static void runCase(int nd, int elemtype, int porder, int k)
{
    char label[64];
    std::snprintf(label, sizeof(label), "%s p%d k%d",
                  nd == 2 ? (elemtype ? "quad" : "tri") : (elemtype ? "hex" : "tet"), porder, k);

    const int n = 2;
    std::vector<double> p0;
    std::vector<int> t0;
    int np0 = 0, nve = 0;
    boxMesh(nd, elemtype, n, p0, t0, np0, nve);
    const int ne0 = (int)t0.size()/nve;

    auto makeMesh = [&](const std::vector<double>& p) {
        Mesh m;
        m.p = p; m.t = t0;
        m.nd = m.dim = nd; m.np = np0; m.ne = ne0; m.nve = nve; m.elemtype = elemtype;
        m.nvf = (nd == 3) ? nd + elemtype : nd;
        m.nfe = nd + (nd - 1)*elemtype + 1;
        return m;
    };

    PDE pde;
    pde.porder = porder;
    pde.pgauss = 2*porder;
    pde.uniformrefinementlevel = k;

    Mesh a = makeMesh(p0);
    Master master = initializeMaster(pde, a);
    const int npe = master.npe;

    auto dgnodes = [&](Mesh& m) {
        std::vector<double> x((size_t)npe*nd*m.ne);
        compute_dgnodes(x.data(), m.p.data(), m.t.data(), master.phielem.data(), npe, nd, m.ne, nve);
        return x;
    };
    // degree-porder polynomials (in both P_p and Q_p)
    auto field = [&](const double* x, int c) {
        const double s = 0.3 + x[0] - 0.7*x[1] + (nd == 3 ? 0.4*x[2] : 0.0);
        return (c == 0) ? std::pow(s, porder) : std::pow(x[0], porder) - std::pow(x[nd-1], porder) + 1.0;
    };
    auto warp = [&](const double* x, double* y) {
        for (int d = 0; d < nd; d++) y[d] = x[d] + 0.05*std::pow(x[(d + 1) % nd], porder);
    };
    auto maxdiff = [](const std::vector<double>& u, const std::vector<double>& v) {
        if (u.size() != v.size()) return 1e300;
        double m = 0.0;
        for (size_t i = 0; i < u.size(); i++) m = std::max(m, std::abs(u[i] - v[i]));
        return m;
    };

    // (a) straight geometry with a polynomial field
    a.xdg = dgnodes(a);
    const std::vector<double> xdg0 = a.xdg;
    a.xdgdims = {npe, nd, ne0};
    a.udg.assign((size_t)npe*2*ne0, 0.0);
    for (int e = 0; e < ne0; e++)
        for (int i = 0; i < npe; i++) {
            double x[3] = {0, 0, 0};
            for (int d = 0; d < nd; d++) x[d] = a.xdg[i + npe*(d + nd*e)];
            for (int c = 0; c < 2; c++) a.udg[i + npe*(c + 2*e)] = field(x, c);
        }
    a.udgdims = {npe, 2, ne0};
    uniformRefineMesh(a, pde, master);

    const int m = n << k;
    const int neExpected = ne0*(1 << (nd*k));
    const int npExpected = (nd == 2) ? (m+1)*(m+1) : (m+1)*(m+1)*(m+1);
    check(a.ne == neExpected, label, "ne " + std::to_string(a.ne) + " != " + std::to_string(neExpected));
    check(a.np == npExpected, label, "np " + std::to_string(a.np) + " != " + std::to_string(npExpected));
    check(pde.ne == a.ne && pde.np == a.np, label, "pde.ne/np not updated");
    check((int)a.t.size() == nve*a.ne && (int)a.p.size() == nd*a.np, label, "p/t sizes");
    check(a.xdgdims[2] == a.ne && a.udgdims[2] == a.ne, label, "field dims not updated");

    std::set<std::vector<long long>> coords;
    for (int j = 0; j < a.np; j++) {
        std::vector<long long> key(nd);
        for (int d = 0; d < nd; d++) key[d] = std::llround(a.p[d + nd*j]*1e9);
        coords.insert(key);
    }
    check((int)coords.size() == a.np, label, "coincident vertices");

    std::map<std::vector<int>, int> faces;
    double volume = 0.0, minvol = 1e300;
    for (int e = 0; e < a.ne; e++) {
        const int* v = &a.t[(size_t)nve*e];
        for (const auto& lf : localFaces(nd, elemtype)) {
            std::vector<int> f;
            for (int q : lf) f.push_back(v[q]);
            std::sort(f.begin(), f.end());
            faces[f]++;
        }
        const double vol = signedVolume(nd, elemtype, a.p, v);
        volume += vol;
        minvol = std::min(minvol, vol);
    }
    int nbou = 0, overshared = 0;
    for (const auto& f : faces) { if (f.second == 1) nbou++; if (f.second > 2) overshared++; }
    const int nbouExpected = (nd == 2) ? 4*m : (elemtype ? 6*m*m : 12*m*m);
    check(overshared == 0, label, std::to_string(overshared) + " faces shared by more than 2 elements");
    check(nbou == nbouExpected, label, "boundary faces " + std::to_string(nbou) + " != " + std::to_string(nbouExpected));
    check(minvol > 0, label, "non-positive element volume " + std::to_string(minvol));
    check(std::abs(volume - 1.0) < 1e-12, label, "total volume " + std::to_string(volume));

    const std::vector<double> xdgR = dgnodes(a);
    const double exdg = maxdiff(a.xdg, xdgR);
    check(exdg < 1e-12, label, "prolongated xdg != compute_dgnodes (" + std::to_string(exdg) + ")");
    double eudg = 0.0;
    for (int e = 0; e < a.ne; e++)
        for (int i = 0; i < npe; i++) {
            double x[3] = {0, 0, 0};
            for (int d = 0; d < nd; d++) x[d] = xdgR[i + npe*(d + nd*e)];
            for (int c = 0; c < 2; c++) eudg = std::max(eudg, std::abs(a.udg[i + npe*(c + 2*e)] - field(x, c)));
        }
    check(eudg < 1e-10, label, "prolongated udg error " + std::to_string(eudg));

    // (b) curved geometry: same topology, vertices and nodes on the warped map
    std::vector<double> pw(p0.size());
    for (int j = 0; j < np0; j++) warp(&p0[(size_t)nd*j], &pw[(size_t)nd*j]);
    Mesh b = makeMesh(pw);
    b.xdg.assign(xdg0.size(), 0.0);
    for (int e = 0; e < ne0; e++)
        for (int i = 0; i < npe; i++) {
            double x[3] = {0, 0, 0}, y[3] = {0, 0, 0};
            for (int d = 0; d < nd; d++) x[d] = xdg0[i + npe*(d + nd*e)];
            warp(x, y);
            for (int d = 0; d < nd; d++) b.xdg[i + npe*(d + nd*e)] = y[d];
        }
    b.xdgdims = {npe, nd, ne0};
    uniformRefineMesh(b, pde, master);

    check(b.t == a.t, label, "curved refinement changed the topology");
    std::vector<double> pwR(a.p.size()), xwR(xdgR.size());
    for (int j = 0; j < a.np; j++) warp(&a.p[(size_t)nd*j], &pwR[(size_t)nd*j]);
    for (int e = 0; e < a.ne; e++)
        for (int i = 0; i < npe; i++) {
            double x[3] = {0, 0, 0}, y[3] = {0, 0, 0};
            for (int d = 0; d < nd; d++) x[d] = xdgR[i + npe*(d + nd*e)];
            warp(x, y);
            for (int d = 0; d < nd; d++) xwR[i + npe*(d + nd*e)] = y[d];
        }
    const double epw = maxdiff(b.p, pwR), exw = maxdiff(b.xdg, xwR);
    check(epw < 1e-12, label, "new vertices off the curved map (" + std::to_string(epw) + ")");
    check(exw < 1e-12, label, "curved xdg not prolongated exactly (" + std::to_string(exw) + ")");

    std::printf("[uniformrefine]   %-12s ne %5d -> %6d  np %5d -> %6d  |dxdg| %.1e  |dudg| %.1e  |dcurved| %.1e\n",
                label, ne0, a.ne, np0, a.np, exdg, eudg, std::max(epw, exw));
}

int main()
{
    for (int nd : {2, 3})
        for (int elemtype : {0, 1})
            for (int porder : {1, 2, 3})
                for (int k : {1, 2})
                    runCase(nd, elemtype, porder, k);

    std::printf("[uniformrefine] %s: uniformrefinementlevel refines tri/quad/tet/hex meshes and "
                "prolongates xdg/udg exactly\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
