// solio -- round-trip regression for exasim::save_solution / load_solution.
// Builds a tiny in-memory HDG Poisson discretization (no solve), fills the DG
// solve-state arrays (udg/uh) with a deterministic pattern, saves them to a
// self-describing binary, clobbers the live arrays, reloads, and asserts a
// BIT-EXACT round-trip -- for both double and float (the on-disk format stores
// native T, so the reload must reproduce every byte). Self-checking: returns 0
// on success, nonzero on any mismatch. No pdeapp.txt -> run directly by the
// consumer harness with EXASIM_DATA_DIR pointing at the installed node tables.

#include <exasim/operators.hpp>
#include <exasim/export.hpp>
#include "poisson2d.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void unitSquareQuadMesh(int n, std::vector<double>& p, std::vector<int>& t, int& np, int& ne)
{
    const int nv = n + 1; np = nv * nv; ne = n * n;
    p.resize((size_t)2 * np); t.resize((size_t)4 * ne);
    for (int iy = 0; iy < nv; ++iy) for (int ix = 0; ix < nv; ++ix) {
        const int j = iy * nv + ix; p[0 + 2 * j] = (double)ix / n; p[1 + 2 * j] = (double)iy / n;
    }
    int e = 0;
    for (int iy = 0; iy < n; ++iy) for (int ix = 0; ix < n; ++ix, ++e) {
        t[0 + 4 * e] = iy * nv + ix;       t[1 + 4 * e] = iy * nv + (ix + 1);
        t[2 + 4 * e] = (iy + 1) * nv + (ix + 1); t[3 + 4 * e] = (iy + 1) * nv + ix;
    }
}

template <class T>
static int roundtrip(const char* tag)
{
    using Model = Poisson2DT<T>;
    const int backend = 0;
    constexpr double TOL = 1e-8;

    int np = 0, ne = 0; std::vector<double> p; std::vector<int> t;
    unitSquareQuadMesh(4, p, t, np, ne);

    PDE pde = exasim::default_pde<Model>();
    pde.porder = 2; pde.pgauss = 4; pde.physicsparam = {1.0};

    exasim::MeshSpec mesh(p.data(), t.data(), np, ne, /*nve=*/4);
    mesh.add_boundary(1, [](const double* x){ return std::abs(x[1])       < TOL; });
    mesh.add_boundary(1, [](const double* x){ return std::abs(x[0] - 1.0) < TOL; });
    mesh.add_boundary(1, [](const double* x){ return std::abs(x[1] - 1.0) < TOL; });
    mesh.add_boundary(1, [](const double* x){ return std::abs(x[0])       < TOL; });

    CDiscretizationT<T, Int> disc(exasim::make_preprocessed<Model, T, Int>(pde, mesh), backend);

    // Deterministic fill of the two always-present arrays (Poisson: ncw=nco=0).
    auto fill = [](T* a, long n, unsigned base){
        for (long i = 0; i < n; ++i)
            a[i] = (T)((int)base) + (T)((unsigned)(i * 2654435761u) % 997u) / (T)997;
    };
    fill(disc.sol.udg, (long)disc.sol.szudg, 1u);
    fill(disc.sol.uh,  (long)disc.sol.szuh,  2u);
    if (disc.sol.szwdg > 0) fill(disc.sol.wdg, (long)disc.sol.szwdg, 3u);
    if (disc.sol.szodg > 0) fill(disc.sol.odg, (long)disc.sol.szodg, 4u);

    std::vector<T> eudg(disc.sol.udg, disc.sol.udg + disc.sol.szudg);
    std::vector<T> euh (disc.sol.uh,  disc.sol.uh  + disc.sol.szuh);

    const std::string path = std::string("solio_") + tag + ".bin";
    exasim::save_solution<T, Int>(disc, path);

    std::memset(disc.sol.udg, 0, (size_t)disc.sol.szudg * sizeof(T));
    std::memset(disc.sol.uh,  0, (size_t)disc.sol.szuh  * sizeof(T));

    exasim::load_solution<T, Int>(disc, path);

    long bad = 0;
    for (long i = 0; i < (long)disc.sol.szudg; ++i) if (disc.sol.udg[i] != eudg[i]) { ++bad; }
    for (long i = 0; i < (long)disc.sol.szuh;  ++i) if (disc.sol.uh[i]  != euh[i])  { ++bad; }
    std::printf("[solio]   (%s) szudg=%ld szuh=%ld  mismatches=%ld -> %s\n",
                tag, (long)disc.sol.szudg, (long)disc.sol.szuh, bad, bad ? "FAIL" : "bit-exact");
    return bad ? 1 : 0;
}

int main()
{
    int bad = 0;
    bad |= roundtrip<double>("f64");
    bad |= roundtrip<float>("f32");
    std::printf("[solio] %s: save_solution/load_solution bit-exact round-trip (double + float), no rebuild\n",
                bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
