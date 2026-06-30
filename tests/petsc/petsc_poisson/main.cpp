// P1b — PETSc drives Exasim's exported HDG operators (steady Poisson).
//
// Goal: PETSc owns the nonlinear/linear solve (SNES + KSP=GMRES) and never
// touches Exasim internals -- it calls back into the exported operators only
// through opaque shims:
//   MatShell MatMult  -> CAssembler::evalMatVec     (apply res.H, the HDG Jacobian)
//   PCShell  PCApply   -> CPreconditioner::ApplyPreconditioner (apply res.K)
//   SNES     FormFunction/FormJacobian -> the condensed residual G(uh)=H*uh-b0
//
// Steady Poisson is LINEAR, so the condensed trace residual is affine:
//   G(uh) = H*uh - b0,   b0 = condensed RHS assembled at uh=0.
// SNES therefore converges in one Newton step; this de-risks the full PETSc
// plumbing on the real operators before heat + TS (P2).
//
// Validation: solve the IDENTICAL exported system (res.H, res.K, b0) two ways --
// PETSc (SNES/KSP + shims) and Exasim-native (CSolver::gmres) -- and check the
// two traces agree. Same operators, two Krylov drivers => must match to KSP tol.

#include <petscsnes.h>

#include <exasim/operators.hpp>      // FEM aggregation + preprocessing + in-memory ctor
#include <exasim/solver_facade.hpp>  // ExasimSolver<M>: curated PDE defaults + add_boundary

#include "my_model.hpp"              // Poisson2D

#include <cmath>
#include <cstdio>
#include <vector>

// ---- opaque operator context handed to every PETSc shim ----------------------
struct OpCtx {
    CDiscretization*            disc = nullptr;
    CAssembler<Poisson2D>*      asmb = nullptr;
    CPreconditioner<Poisson2D>* prec = nullptr;
    sysstruct*                  sys  = nullptr;   // workspace (sys.u/sys.b unused by HDG matvec)
    Int                         N    = 0;
    Int                         backend = 0;
    std::vector<dstype>         b0;               // condensed RHS at uh=0  (the affine const)
};

// JV = H * V   (HDG Jacobian apply via res.H; u/Ru args ignored for HDG)
static PetscErrorCode MatMult_Exasim(Mat J, Vec V, Vec JV)
{
    OpCtx* c = nullptr;
    PetscCall(MatShellGetContext(J, &c));
    const PetscScalar* v; PetscScalar* jv;
    PetscCall(VecGetArrayRead(V, &v));
    PetscCall(VecGetArray(JV, &jv));
    c->asmb->evalMatVec(jv, const_cast<dstype*>(v), c->sys->u, c->sys->b, /*spatialScheme=*/1, c->backend);
    PetscCall(VecRestoreArrayRead(V, &v));
    PetscCall(VecRestoreArray(JV, &jv));
    return PETSC_SUCCESS;
}

// PV = K^{-1} * V   (ApplyPreconditioner is in-place, so copy first)
static PetscErrorCode PCApply_Exasim(PC pc, Vec V, Vec PV)
{
    OpCtx* c = nullptr;
    PetscCall(PCShellGetContext(pc, &c));
    PetscCall(VecCopy(V, PV));
    PetscScalar* pv;
    PetscCall(VecGetArray(PV, &pv));
    c->prec->ApplyPreconditioner(pv, *c->sys, *c->disc, /*spatialScheme=*/1, c->backend);
    PetscCall(VecRestoreArray(PV, &pv));
    return PETSC_SUCCESS;
}

// F = G(U) = H*U - b0   (affine: steady Poisson is linear)
static PetscErrorCode FormFunction_Exasim(SNES, Vec U, Vec F, void* ctx)
{
    OpCtx* c = static_cast<OpCtx*>(ctx);
    const PetscScalar* u; PetscScalar* f;
    PetscCall(VecGetArrayRead(U, &u));
    PetscCall(VecGetArray(F, &f));
    c->asmb->evalMatVec(f, const_cast<dstype*>(u), c->sys->u, c->sys->b, 1, c->backend); // f = H*u
    for (Int i = 0; i < c->N; ++i) f[i] -= c->b0[(size_t)i];                              // f -= b0
    PetscCall(VecRestoreArrayRead(U, &u));
    PetscCall(VecRestoreArray(F, &f));
    return PETSC_SUCCESS;
}

// Jacobian is the constant MatShell J (res.H already assembled); nothing to recompute.
static PetscErrorCode FormJacobian_Exasim(SNES, Vec, Mat, Mat, void*) { return PETSC_SUCCESS; }

// Uniform n x n quad mesh of [0,1]^2 (column-major p: nd x np, t: nve x ne, 0-based CCW).
static void unitSquareQuadMesh(int n, std::vector<double>& p, std::vector<int>& t, int& np, int& ne)
{
    const int nv = n + 1; np = nv * nv; ne = n * n;
    p.resize((size_t)2 * np); t.resize((size_t)4 * ne);
    for (int iy = 0; iy < nv; ++iy)
        for (int ix = 0; ix < nv; ++ix) {
            const int j = iy * nv + ix;
            p[0 + 2*j] = (double)ix / n; p[1 + 2*j] = (double)iy / n;
        }
    int e = 0;
    for (int iy = 0; iy < n; ++iy)
        for (int ix = 0; ix < n; ++ix, ++e) {
            const int nvv = nv;
            t[0+4*e] = iy*nvv+ix;       t[1+4*e] = iy*nvv+(ix+1);
            t[2+4*e] = (iy+1)*nvv+(ix+1); t[3+4*e] = (iy+1)*nvv+ix;
        }
}

int main(int argc, char** argv)
{
    PetscCall(PetscInitialize(&argc, &argv, nullptr, "Exasim operators driven by PETSc (steady Poisson)\n"));
    Kokkos::initialize(argc, argv);
    int rc = 0;
    {
        const int   backend = 0;
        const Int   mpiprocs = 1, mpirank = 0, fileoffset = 0, omprank = 0;
        constexpr double TOL = 1e-8;

        // ---- in-memory mesh + curated PDE defaults + boundaries ----
        int np = 0, ne = 0; std::vector<double> p; std::vector<int> t;
        unitSquareQuadMesh(8, p, t, np, ne);

        exasim::ExasimSolver<Poisson2D> solver;
        solver.set_polynomial_order(3);
        solver.set_quadrature_order(6);
        solver.set_physics_params({1.0});
        solver.add_boundary(1, [](const double* x){ return std::abs(x[1])       < TOL; });
        solver.add_boundary(1, [](const double* x){ return std::abs(x[0] - 1.0) < TOL; });
        solver.add_boundary(1, [](const double* x){ return std::abs(x[1] - 1.0) < TOL; });
        solver.add_boundary(1, [](const double* x){ return std::abs(x[0])       < TOL; });

        CPreprocessing preproc(solver.pde(), solver.params(), solver.spec(), mpirank, mpiprocs);
        preproc.mesh = meshFromArrays(p.data(), t.data(), np, ne, 4, Poisson2D::nd,
                                      preproc.params, preproc.pde);
        exasim::Preprocessed pre = preproc.take();
        pre.save_outputs = false;

        // ---- exported discretization + operators (no Exasim solver loop) ----
        CDiscretization disc(std::move(pre), "", solver.pde().exasimpath,
                             mpiprocs, mpirank, fileoffset, omprank, backend, solver.pde().builtinmodelID);
        CResidual<Poisson2D>       residual(disc);
        CAssembler<Poisson2D>      assembler(disc);
        CPreconditioner<Poisson2D> prec(disc, backend, ExasimExecutionMode::Solve);
        CSolver<Poisson2D>         solv(disc, backend, ExasimExecutionMode::Solve); // owns sys + native gmres
        residual.initializeSolution();
        residual.recoverInitialState(backend, /*postprocessOnly=*/false);

        const Int N = disc.common.sizes.ndofuhat;
        std::printf("[petsc] in-memory operators built; N (trace dofs) = %lld\n", (long long)N);

        // ---- assemble the condensed system at uh0=0:  H, K, b0 ----
        if (disc.common.components.ncq > 0)
            hdgGetQ(disc.sol.udg, disc.sol.uh, disc.sol, disc.res, disc.mesh, disc.tmp, disc.common, backend);
        assembler.hdgAssembleLinearSystem(solv.sys.b, backend);   // -> res.H/res.Rh/res.Rq + sys.b
        prec.ComputeHDGPreconditioner(disc, backend);             // -> res.K

        std::vector<dstype> b0((size_t)N);
        for (Int i = 0; i < N; ++i) b0[(size_t)i] = solv.sys.b[i];

        OpCtx ctx; ctx.disc=&disc; ctx.asmb=&assembler; ctx.prec=&prec; ctx.sys=&solv.sys;
        ctx.N=N; ctx.backend=backend; ctx.b0=b0;

        // ================= PETSc SNES + KSP(GMRES) + MatShell + PCShell =================
        Mat J;
        PetscCall(MatCreateShell(PETSC_COMM_SELF, N, N, N, N, &ctx, &J));
        PetscCall(MatShellSetOperation(J, MATOP_MULT, (void(*)(void))MatMult_Exasim));

        Vec U, F;
        PetscCall(MatCreateVecs(J, &U, &F));
        PetscCall(VecSet(U, 0.0));

        SNES snes;
        PetscCall(SNESCreate(PETSC_COMM_SELF, &snes));
        PetscCall(SNESSetFunction(snes, F, FormFunction_Exasim, &ctx));
        PetscCall(SNESSetJacobian(snes, J, J, FormJacobian_Exasim, &ctx));

        KSP ksp; PC pc;
        PetscCall(SNESGetKSP(snes, &ksp));
        PetscCall(KSPSetType(ksp, KSPGMRES));
        PetscCall(KSPGetPC(ksp, &pc));
        PetscCall(PCSetType(pc, PCSHELL));
        PetscCall(PCShellSetContext(pc, &ctx));
        PetscCall(PCShellSetApply(pc, PCApply_Exasim));
        PetscCall(KSPSetTolerances(ksp, 1e-10, 1e-12, PETSC_DEFAULT, 500));
        PetscCall(SNESSetTolerances(snes, 1e-10, 1e-12, 1e-12, 50, 1000));
        PetscCall(SNESSetFromOptions(snes));

        PetscCall(SNESSolve(snes, nullptr, U));
        SNESConvergedReason reason; PetscInt its;
        PetscCall(SNESGetConvergedReason(snes, &reason));
        PetscCall(SNESGetIterationNumber(snes, &its));
        std::printf("[petsc] SNES converged reason = %d, Newton iters = %lld\n", (int)reason, (long long)its);

        std::vector<dstype> uh_petsc((size_t)N);
        { const PetscScalar* u; PetscCall(VecGetArrayRead(U, &u));
          for (Int i=0;i<N;++i) uh_petsc[(size_t)i]=u[i]; PetscCall(VecRestoreArrayRead(U,&u)); }

        // ================= Exasim-native reference: CSolver::gmres on the SAME H/K/b0 ====
        for (Int i = 0; i < N; ++i) { solv.sys.u[i] = 0.0; solv.sys.x[i] = 0.0; solv.sys.b[i] = b0[(size_t)i]; }
        std::ofstream devnull;
        solv.gmres(assembler, disc, prec, N, /*spatialScheme=*/1, backend);   // solves H*x = b -> sys.x
        std::vector<dstype> uh_native((size_t)N);
        for (Int i = 0; i < N; ++i) uh_native[(size_t)i] = solv.sys.x[i];

        // ---- compare ----
        double dnum = 0.0, dden = 0.0, np_ = 0.0, nn_ = 0.0;
        bool finite = true;
        for (Int i = 0; i < N; ++i) {
            if (!std::isfinite(uh_petsc[(size_t)i]) || !std::isfinite(uh_native[(size_t)i])) finite = false;
            const double d = uh_petsc[(size_t)i] - uh_native[(size_t)i];
            dnum += d*d; dden += uh_native[(size_t)i]*uh_native[(size_t)i];
            np_ += uh_petsc[(size_t)i]*uh_petsc[(size_t)i]; nn_ += uh_native[(size_t)i]*uh_native[(size_t)i];
        }
        const double relerr = std::sqrt(dnum) / (std::sqrt(dden) + 1e-300);
        std::printf("[petsc] ||uh_petsc||  = %.8e\n", std::sqrt(np_));
        std::printf("[petsc] ||uh_native|| = %.8e\n", std::sqrt(nn_));
        std::printf("[petsc] rel || uh_petsc - uh_native || = %.3e\n", relerr);

        if (!finite)            { std::printf("[petsc] FAIL: non-finite trace\n"); rc = 1; }
        else if (reason < 0)    { std::printf("[petsc] FAIL: SNES did not converge\n"); rc = 1; }
        else if (relerr > 1e-6) { std::printf("[petsc] FAIL: PETSc and native traces disagree\n"); rc = 1; }
        else                     std::printf("[petsc] PASS: PETSc drives the exported operators to the native solution\n");

        PetscCall(VecDestroy(&U)); PetscCall(VecDestroy(&F));
        PetscCall(MatDestroy(&J)); PetscCall(SNESDestroy(&snes));
    }
    Kokkos::finalize();
    PetscCall(PetscFinalize());
    return rc;
}
