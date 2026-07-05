// meshload -- regression for exasim::read_mesh + as_mesh_spec: load a mesh from
// a file into the pure-C++ preprocessing path and prove the resulting
// discretization is identical to the one built directly from in-memory arrays.
// Writes a tiny unit-square quad mesh as a .txt file (1-based connectivity, the
// on-disk convention readMeshFromFile normalizes to 0-based), reads it back, and
// compares topology + allocated sizes against the array-built discretization.
// Self-checking: 0 on success, nonzero on mismatch. No pdeapp.txt.

#include <exasim/operators.hpp>
#include <exasim/export.hpp>
#include "poisson2d.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using Model = Poisson2DT<double>;

static void unitSquareQuadMesh(int n, std::vector<double>& p, std::vector<int>& t, int& np, int& ne)
{
    const int nv = n + 1; np = nv * nv; ne = n * n;
    p.resize((size_t)2 * np); t.resize((size_t)4 * ne);
    for (int iy = 0; iy < nv; ++iy) for (int ix = 0; ix < nv; ++ix) {
        const int j = iy * nv + ix; p[0 + 2 * j] = (double)ix / n; p[1 + 2 * j] = (double)iy / n;
    }
    int e = 0;
    for (int iy = 0; iy < n; ++iy) for (int ix = 0; ix < n; ++ix, ++e) {
        t[0 + 4 * e] = iy * nv + ix;             t[1 + 4 * e] = iy * nv + (ix + 1);
        t[2 + 4 * e] = (iy + 1) * nv + (ix + 1); t[3 + 4 * e] = (iy + 1) * nv + ix;
    }
}

// Write a .txt mesh: "nd np nve ne" then p (column-major) then t (1-based,
// column-major) -- readMeshFromTextFile's format; readMeshFromFile decrements.
static void writeTxtMesh(const std::string& path, const std::vector<double>& p,
                         const std::vector<int>& t, int np, int ne, int nd, int nve)
{
    std::ofstream o(path.c_str());
    o << nd << " " << np << " " << nve << " " << ne << "\n";
    for (int j = 0; j < np; ++j) { for (int i = 0; i < nd; ++i) o << p[i + nd * j] << " "; o << "\n"; }
    for (int e = 0; e < ne; ++e) { for (int i = 0; i < nve; ++i) o << (t[i + nve * e] + 1) << " "; o << "\n"; }
}

static void addBoundaries(exasim::MeshSpec& m)
{
    constexpr double TOL = 1e-8;
    m.add_boundary(1, [](const double* x){ return std::abs(x[1])       < TOL; });
    m.add_boundary(1, [](const double* x){ return std::abs(x[0] - 1.0) < TOL; });
    m.add_boundary(1, [](const double* x){ return std::abs(x[1] - 1.0) < TOL; });
    m.add_boundary(1, [](const double* x){ return std::abs(x[0])       < TOL; });
}

int main()
{
    const int backend = 0;
    int np = 0, ne = 0; std::vector<double> p; std::vector<int> t;
    unitSquareQuadMesh(4, p, t, np, ne);

    PDE pde = exasim::default_pde<Model>();
    pde.porder = 2; pde.pgauss = 4; pde.physicsparam = {1.0};

    // (a) array-built reference discretization
    exasim::MeshSpec ms_arr(p.data(), t.data(), np, ne, /*nve=*/4);
    addBoundaries(ms_arr);
    CDiscretizationT<double, Int> disc_arr(
        exasim::make_preprocessed<Model, double, Int>(pde, ms_arr), backend);

    // (b) file-built discretization via read_mesh + as_mesh_spec
    const std::string mpath = "meshload_unit4.txt";
    writeTxtMesh(mpath, p, t, np, ne, /*nd=*/2, /*nve=*/4);
    Mesh m = exasim::read_mesh(mpath);
    exasim::MeshSpec ms_file = exasim::as_mesh_spec(m);   // m outlives ms_file
    addBoundaries(ms_file);
    CDiscretizationT<double, Int> disc_file(
        exasim::make_preprocessed<Model, double, Int>(pde, ms_file), backend);

    auto& ca = disc_arr.common; auto& cf = disc_file.common;
    struct Cmp { const char* nm; long a, b; };
    const Cmp cmp[] = {
        {"np",        (long)m.np,                (long)np},
        {"ne",        (long)m.ne,                (long)ne},
        {"nve",       (long)m.nve,               4},
        {"ndofuhat",  (long)ca.sizes.ndofuhat,   (long)cf.sizes.ndofuhat},
        {"szudg",     (long)disc_arr.sol.szudg,  (long)disc_file.sol.szudg},
        {"szuh",      (long)disc_arr.sol.szuh,   (long)disc_file.sol.szuh},
        {"ne1",       (long)ca.meshsizes.ne1,    (long)cf.meshsizes.ne1},
        {"nf",        (long)ca.meshsizes.nf,     (long)cf.meshsizes.nf},
    };
    int bad = 0;
    for (const auto& c : cmp) {
        const bool ok = (c.a == c.b);
        std::printf("[meshload]   %-9s array=%ld file=%ld -> %s\n", c.nm, c.a, c.b, ok ? "ok" : "MISMATCH");
        if (!ok) bad = 1;
    }

    // spot-check node coordinates agree (same p in, same xdg out)
    const long ncheck = std::min<long>(64, (long)disc_arr.sol.szxdg);
    long xbad = 0;
    for (long i = 0; i < ncheck; ++i) if (disc_arr.sol.xdg[i] != disc_file.sol.xdg[i]) ++xbad;
    std::printf("[meshload]   xdg[0:%ld] mismatches=%ld\n", ncheck, xbad);
    if (xbad) bad = 1;

    std::printf("[meshload] %s: read_mesh + as_mesh_spec yields the array-built discretization\n",
                bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
