// P2 (increment 2a) — TRANSIENT operator export: one backward-Euler step of the
// heat equation, solved by PETSc KSP(GMRES)+MatShell+PCShell on Exasim's
// TIME-AUGMENTED HDG operators, validated against Exasim-native GMRES on the
// identical operators.
//
// Heat = the SAME Poisson2D spatial model (NO model change): Poisson2D inherits
// ModelDefaults::tdfunc -> m=1 (identity mass = du/dt coeff 1), and dtcoef_u=1.
// Transient is pure CONFIG: tdep=1, torder=1 (backward Euler = DIRK(1,1)), dt.
// The time term auto-folds into the residual/Jacobian via TdfuncDriver<M> +
// UpdateSource (which sets dtfactor = DIRKcoeff/dt and the history source sdg).
//
// This increment proves the time-augmented operator export works in memory and
// that PETSc drives it identically to Exasim-native. The full PETSc TS multi-step
// loop is a WIP (see main_ts_wip.cpp.txt + .claude/PLAN.md): PETSc TS owns the
// loop and the per-step operator + TS integration are validated (a single BE step
// and the large-dt -> steady limit both match native), but the inter-step udg
// recovery still needs the exact Exasim native-DIRK state sequence.

#include <petscksp.h>

#include <exasim/operators.hpp>
#include <exasim/solver_facade.hpp>
#include "my_model.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

struct OpCtx {
    CAssembler<Poisson2D>*      asmb = nullptr;
    CPreconditioner<Poisson2D>* prec = nullptr;
    CDiscretization*            disc = nullptr;
    sysstruct*                  sys  = nullptr;
    Int                         backend = 0;
};

static PetscErrorCode MatMult_Exasim(Mat J, Vec V, Vec JV)
{
    OpCtx* c; PetscCall(MatShellGetContext(J, &c));
    const PetscScalar* v; PetscScalar* jv;
    PetscCall(VecGetArrayRead(V, &v)); PetscCall(VecGetArray(JV, &jv));
    c->asmb->evalMatVec(jv, const_cast<dstype*>(v), c->sys->u, c->sys->b, 1, c->backend);
    PetscCall(VecRestoreArrayRead(V, &v)); PetscCall(VecRestoreArray(JV, &jv));
    return PETSC_SUCCESS;
}
static PetscErrorCode PCApply_Exasim(PC pc, Vec V, Vec PV)
{
    OpCtx* c; PetscCall(PCShellGetContext(pc, &c));
    PetscCall(VecCopy(V, PV));
    PetscScalar* pv; PetscCall(VecGetArray(PV, &pv));
    c->prec->ApplyPreconditioner(pv, *c->sys, *c->disc, 1, c->backend);
    PetscCall(VecRestoreArray(PV, &pv));
    return PETSC_SUCCESS;
}

static void unitSquareQuadMesh(int n, std::vector<double>& p, std::vector<int>& t, int& np, int& ne)
{
    const int nv=n+1; np=nv*nv; ne=n*n; p.resize((size_t)2*np); t.resize((size_t)4*ne);
    for (int iy=0; iy<nv; ++iy) for (int ix=0; ix<nv; ++ix) {
        const int j=iy*nv+ix; p[0+2*j]=(double)ix/n; p[1+2*j]=(double)iy/n; }
    int e=0;
    for (int iy=0; iy<n; ++iy) for (int ix=0; ix<n; ++ix,++e) {
        t[0+4*e]=iy*nv+ix; t[1+4*e]=iy*nv+(ix+1); t[2+4*e]=(iy+1)*nv+(ix+1); t[3+4*e]=(iy+1)*nv+ix; }
}

int main(int argc, char** argv)
{
    PetscCall(PetscInitialize(&argc, &argv, nullptr, "Exasim transient operators via PETSc (heat, 1 BE step)\n"));
    Kokkos::initialize(argc, argv);
    int rc = 0;
    {
        const int backend = 0;
        const Int mpiprocs=1, mpirank=0, fileoffset=0, omprank=0;
        constexpr double TOL = 1e-8;
        const double dt = 0.1;

        int np=0, ne=0; std::vector<double> p; std::vector<int> t;
        unitSquareQuadMesh(8, p, t, np, ne);

        exasim::ExasimSolver<Poisson2D> solver;
        solver.set_polynomial_order(3);
        solver.set_quadrature_order(6);
        solver.set_physics_params({1.0});
        solver.add_boundary(1, [](const double* x){ return std::abs(x[1])       < TOL; });
        solver.add_boundary(1, [](const double* x){ return std::abs(x[0] - 1.0) < TOL; });
        solver.add_boundary(1, [](const double* x){ return std::abs(x[1] - 1.0) < TOL; });
        solver.add_boundary(1, [](const double* x){ return std::abs(x[0])       < TOL; });

        // --- TRANSIENT CONFIG: backward Euler (DIRK(1,1)) ---
        solver.pde().tdep   = 1;
        solver.pde().torder = 1;
        solver.pde().nstage = 1;
        solver.pde().dt     = { dt };

        CPreprocessing preproc(solver.pde(), solver.params(), solver.spec(), mpirank, mpiprocs);
        preproc.mesh = meshFromArrays(p.data(), t.data(), np, ne, 4, Poisson2D::nd,
                                      preproc.params, preproc.pde);
        exasim::Preprocessed pre = preproc.take();
        pre.save_outputs = false;

        CDiscretization disc(std::move(pre), "", solver.pde().exasimpath,
                             mpiprocs, mpirank, fileoffset, omprank, backend, solver.pde().builtinmodelID);
        CResidual<Poisson2D>       residual(disc);
        CAssembler<Poisson2D>      assembler(disc);
        CPreconditioner<Poisson2D> prec(disc, backend, ExasimExecutionMode::Solve);
        CSolver<Poisson2D>         solv(disc, backend, ExasimExecutionMode::Solve);
        residual.initializeSolution();
        residual.recoverInitialState(backend, false);

        const Int N = disc.common.sizes.ndofuhat;
        std::printf("[heat] N=%lld, tdep=%d, ncs=%lld, dt=%g\n", (long long)N,
                    (int)disc.common.timeparams.tdep, (long long)disc.common.components.ncs, dt);

        TimestepCoefficents(disc.common);   // BE: DIRKcoeff_d[0]=1

        // --- one BE step from the initial state ---
        disc.common.timestate.currentstep  = 0;
        PreviousSolutions(disc.sol, solv.sys, disc.common, backend);     // save udg^0
        disc.common.timestate.currentstage = 0;
        disc.common.timestate.time         = dt * disc.common.DIRKcoeff_t[0];
        UpdateSource(disc.sol, solv.sys, disc.app, disc.driver_abi, disc.res, disc.common, backend);
        std::printf("[heat] UpdateSource done; dtfactor=%g (expect 1/dt=%g)\n",
                    disc.common.timestate.dtfactor, 1.0/dt);

        if (disc.common.components.ncq > 0)
            hdgGetQ(disc.sol.udg, disc.sol.uh, disc.sol, disc.res, disc.mesh, disc.tmp, disc.common, backend);
        assembler.hdgAssembleLinearSystem(solv.sys.b, backend);   // TIME-AUGMENTED H + b
        prec.ComputeHDGPreconditioner(disc, backend);             // TIME-AUGMENTED K

        std::vector<dstype> b0((size_t)N);
        for (Int i=0;i<N;++i) b0[(size_t)i]=solv.sys.b[i];

        OpCtx ctx; ctx.asmb=&assembler; ctx.prec=&prec; ctx.disc=&disc; ctx.sys=&solv.sys; ctx.backend=backend;

        // ---- PETSc KSP(GMRES)+MatShell+PCShell on the time-augmented operator ----
        Mat J; PetscCall(MatCreateShell(PETSC_COMM_SELF, N, N, N, N, &ctx, &J));
        PetscCall(MatShellSetOperation(J, MATOP_MULT, (void(*)(void))MatMult_Exasim));
        Vec X, B; PetscCall(MatCreateVecs(J, &X, &B));
        for (Int i=0;i<N;++i) { PetscCall(VecSetValue(B, i, b0[(size_t)i], INSERT_VALUES)); }
        PetscCall(VecAssemblyBegin(B)); PetscCall(VecAssemblyEnd(B));
        PetscCall(VecSet(X, 0.0));
        KSP ksp; PC pc;
        PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
        PetscCall(KSPSetOperators(ksp, J, J));
        PetscCall(KSPSetType(ksp, KSPGMRES));
        PetscCall(KSPGetPC(ksp, &pc));
        PetscCall(PCSetType(pc, PCSHELL));
        PetscCall(PCShellSetContext(pc, &ctx));
        PetscCall(PCShellSetApply(pc, PCApply_Exasim));
        PetscCall(KSPSetTolerances(ksp, 1e-12, 1e-14, PETSC_DEFAULT, 500));
        PetscCall(KSPSolve(ksp, B, X));
        KSPConvergedReason reason; PetscInt its;
        PetscCall(KSPGetConvergedReason(ksp, &reason));
        PetscCall(KSPGetIterationNumber(ksp, &its));
        std::printf("[heat] KSP reason=%d, iters=%lld\n", (int)reason, (long long)its);

        std::vector<dstype> uh_petsc((size_t)N);
        { const PetscScalar* x; PetscCall(VecGetArrayRead(X,&x));
          for (Int i=0;i<N;++i) uh_petsc[(size_t)i]=x[i]; PetscCall(VecRestoreArrayRead(X,&x)); }

        // ---- Exasim-native GMRES on the SAME time-augmented operator ----
        for (Int i=0;i<N;++i){ solv.sys.u[i]=0.0; solv.sys.x[i]=0.0; solv.sys.b[i]=b0[(size_t)i]; }
        solv.gmres(assembler, disc, prec, N, 1, backend);
        std::vector<dstype> uh_native((size_t)N);
        for (Int i=0;i<N;++i) uh_native[(size_t)i]=solv.sys.x[i];

        double dnum=0, dden=0, npn=0, nnn=0; bool finite=true;
        for (Int i=0;i<N;++i){
            if(!std::isfinite(uh_petsc[(size_t)i])||!std::isfinite(uh_native[(size_t)i])) finite=false;
            double d=uh_petsc[(size_t)i]-uh_native[(size_t)i];
            dnum+=d*d; dden+=uh_native[(size_t)i]*uh_native[(size_t)i];
            npn+=uh_petsc[(size_t)i]*uh_petsc[(size_t)i]; nnn+=uh_native[(size_t)i]*uh_native[(size_t)i]; }
        double relerr=std::sqrt(dnum)/(std::sqrt(dden)+1e-300);
        std::printf("[heat] ||uh_petsc||=%.8e  ||uh_native||=%.8e\n", std::sqrt(npn), std::sqrt(nnn));
        std::printf("[heat] rel ||uh_petsc - uh_native|| = %.3e\n", relerr);

        if(!finite){ std::printf("[heat] FAIL: non-finite\n"); rc=1; }
        else if(reason<0){ std::printf("[heat] FAIL: KSP diverged\n"); rc=1; }
        else if(!(std::sqrt(nnn)>0)){ std::printf("[heat] FAIL: native trace is zero (time term not assembled)\n"); rc=1; }
        else if(relerr>1e-6){ std::printf("[heat] FAIL: PETSc vs native disagree\n"); rc=1; }
        else std::printf("[heat] PASS: PETSc drives the time-augmented operator to the native BE step\n");

        PetscCall(VecDestroy(&X)); PetscCall(VecDestroy(&B));
        PetscCall(MatDestroy(&J)); PetscCall(KSPDestroy(&ksp));
    }
    Kokkos::finalize();
    PetscCall(PetscFinalize());
    return rc;
}
