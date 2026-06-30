// Operator-export consumer (P1a) — exercises the IN-MEMORY discretization +
// operator path that an external driver (e.g. PETSc) builds against, with NO
// Exasim solver, NO datain files, and NO text2code kernels.
//
// What it proves:
//   1. <exasim/operators.hpp> compiles for a concrete hand-written Model
//      (Poisson2D) — the templated FEM aggregation + preprocessing + the
//      in-memory CDiscretization(Preprocessed&&) ctor.
//   2. The never-before-run in-memory construction path actually works:
//      ExasimSolver<M> curated PDE defaults  ->  meshFromArrays  ->
//      CPreprocessing::take()  ->  CDiscretization(Preprocessed&&)  ->
//      CResidual<M> / CAssembler<M> / CPreconditioner<M>.
//   3. The HDG residual operator R(uh) assembles to a finite vector of the
//      expected length N = common.sizes.ndofuhat (the PETSc state dimension).
//
// This is the recipe a PETSc app will reuse: build the operators in memory,
// then drive SNES/KSP/PCShell matrix-free against them (P1b).

#include <exasim/operators.hpp>     // FEM aggregation + preprocessing + in-memory ctor
#include <exasim/solver_facade.hpp> // ExasimSolver<M>: curated PDE defaults + add_boundary

#include "my_model.hpp"             // Poisson2D (hand-written, no codegen)

#include <cmath>
#include <cstdio>
#include <vector>

// Build a uniform n x n quad mesh of the unit square [0,1]^2.
//   p : nd(=2) x np column-major vertex coords
//   t : nve(=4) x ne column-major, 0-based, CCW corners
static void unitSquareQuadMesh(int n, std::vector<double>& p, std::vector<int>& t,
                               int& np, int& ne)
{
    const int nv = n + 1;
    np = nv * nv;
    ne = n * n;
    p.resize((size_t)2 * np);
    t.resize((size_t)4 * ne);

    for (int iy = 0; iy < nv; ++iy)
        for (int ix = 0; ix < nv; ++ix) {
            const int j = iy * nv + ix;
            p[0 + 2 * j] = (double)ix / n;
            p[1 + 2 * j] = (double)iy / n;
        }

    int e = 0;
    for (int iy = 0; iy < n; ++iy)
        for (int ix = 0; ix < n; ++ix, ++e) {
            const int v0 = iy * nv + ix;
            const int v1 = iy * nv + (ix + 1);
            const int v2 = (iy + 1) * nv + (ix + 1);
            const int v3 = (iy + 1) * nv + ix;
            t[0 + 4 * e] = v0;
            t[1 + 4 * e] = v1;
            t[2 + 4 * e] = v2;
            t[3 + 4 * e] = v3;
        }
}

int main()
{
    const int backend = 0;            // serial CPU
    const Int mpiprocs = 1, mpirank = 0, fileoffset = 0, omprank = 0;
    constexpr double TOL = 1e-8;

    // ---- 1. Mesh (in memory) ------------------------------------------------
    int np = 0, ne = 0;
    std::vector<double> p;
    std::vector<int>    t;
    unitSquareQuadMesh(/*n=*/8, p, t, np, ne);
    std::printf("[operators] mesh: %d verts, %d quad elements\n", np, ne);

    // ---- 2. Curated PDE defaults + boundary conditions ----------------------
    // Reuse ExasimSolver<M> purely for the (verified) PDE-default setup and the
    // boundary-predicate plumbing; we do NOT call solve() (no Exasim solver).
    exasim::ExasimSolver<Poisson2D> solver;
    solver.set_polynomial_order(3);   // matches apps/poisson/poisson2d (porder=3)
    solver.set_quadrature_order(6);
    solver.set_physics_params({1.0}); // mu = 1
    // All four sides Dirichlet (tag 1), mirroring the poisson2d pdeapp.
    solver.add_boundary(1, [](const double* x){ return std::abs(x[1])       < TOL; }); // y=0
    solver.add_boundary(1, [](const double* x){ return std::abs(x[0] - 1.0) < TOL; }); // x=1
    solver.add_boundary(1, [](const double* x){ return std::abs(x[1] - 1.0) < TOL; }); // y=1
    solver.add_boundary(1, [](const double* x){ return std::abs(x[0])       < TOL; }); // x=0

    // ---- 3. Preprocess into an in-memory Preprocessed bundle -----------------
    // CPreprocessing's programmatic ctor consumes the curated pde/params/spec;
    // meshFromArrays replaces the on-disk grid; take() runs master/DMD/struct
    // assembly with NO datain binaries written.
    CPreprocessing preproc(solver.pde(), solver.params(), solver.spec(), mpirank, mpiprocs);
    preproc.mesh = meshFromArrays(p.data(), t.data(), np, ne, /*nve=*/4, Poisson2D::nd,
                                  preproc.params, preproc.pde);
    exasim::Preprocessed pre = preproc.take();
    pre.save_outputs = false;
    std::printf("[operators] preprocessing done (in-memory Preprocessed built)\n");

    // ---- 4. In-memory discretization + operators ----------------------------
    CDiscretization disc(std::move(pre), /*fileout=*/"", solver.pde().exasimpath,
                         mpiprocs, mpirank, fileoffset, omprank, backend,
                         solver.pde().builtinmodelID);
    std::printf("[operators] CDiscretization(Preprocessed&&) constructed\n");

    CResidual<Poisson2D>      residual(disc);
    CAssembler<Poisson2D>     assembler(disc);
    CPreconditioner<Poisson2D> prec(disc, backend, ExasimExecutionMode::Solve);

    // Initialize the operator's own state (model initial condition, then recover
    // q/uh from u) -- the bits CSolution's ctor runs before any solve.
    residual.initializeSolution();
    residual.recoverInitialState(backend, /*postprocessOnly=*/false);
    std::printf("[operators] operators built; initial state recovered\n");

    // ---- 5. Smoke test: assemble the HDG residual R(uh) ---------------------
    const Int N = disc.common.sizes.ndofuhat;   // PETSc state dimension
    if (N <= 0) { std::printf("[operators] FAIL: ndofuhat = %lld\n", (long long)N); return 1; }

    auto l2norm = [](const std::vector<dstype>& v) {
        double s = 0.0; for (double x : v) s += x * x; return std::sqrt(s);
    };
    auto allFinite = [](const std::vector<dstype>& v) {
        for (double x : v) if (!std::isfinite(x)) return false; return true;
    };

    if (disc.common.components.ncq > 0)
        hdgGetQ(disc.sol.udg, disc.sol.uh, disc.sol, disc.res, disc.mesh, disc.tmp, disc.common, backend);

    // (a) Raw trace residual R_h(uh) = face numerical-flux balance. At the zero
    //     initial state this is identically zero (the source enters the trace
    //     only through static condensation, exercised in (b)).
    std::vector<dstype> rh((size_t)N, 0.0);
    assembler.hdgAssembleResidual(rh.data(), backend);
    if (!allFinite(rh)) { std::printf("[operators] FAIL: R_h not finite\n"); return 1; }

    // (b) Condensed linear system  H * duh = b  (b = statically-condensed
    //     residual). This is the operator PETSc's SNES FormFunction/FormJacobian
    //     will drive: at uh0=0 it is source-driven and must be NONZERO and finite.
    std::vector<dstype> b((size_t)N, 0.0);
    assembler.hdgAssembleLinearSystem(b.data(), backend);
    if (!allFinite(b)) { std::printf("[operators] FAIL: condensed residual not finite\n"); return 1; }

    const double nrm_rh = l2norm(rh);
    const double nrm_b   = l2norm(b);
    std::printf("[operators] N = %lld\n", (long long)N);
    std::printf("[operators]   ||R_h(uh0)||            = %.6e  (raw trace flux balance; 0 expected)\n", nrm_rh);
    std::printf("[operators]   ||b(uh0)|| (condensed)  = %.6e  (source-driven; must be > 0)\n", nrm_b);

    if (!(nrm_b > 0.0)) {
        std::printf("[operators] FAIL: condensed residual is zero (source not assembled)\n");
        return 1;
    }
    std::printf("[operators] PASS\n");
    return 0;
}
