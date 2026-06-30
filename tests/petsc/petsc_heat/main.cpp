// P2 — PETSc TS owns the time loop for the heat equation, driving Exasim's
// exported TIME-AUGMENTED HDG operators (Level A "loop-driver").
//
// Heat = the SAME Poisson2D spatial model (NO model change): Poisson2D inherits
// ModelDefaults::tdfunc -> m=1 (identity mass = du/dt coeff 1), dtcoef_u=1.
// Transient is pure CONFIG: tdep=1, torder=1 (backward Euler = DIRK(1,1)), dt.
//
// Why "loop-driver" and not PETSc-native M*udot-R: in HDG the global unknown is
// the trace uh, which is ALGEBRAIC (no time derivative; du/dt lives on the volume
// udg, folded into the source + statically condensed). Condensation makes the
// trace Jacobian non-affine in the time shift, so there is no separable trace
// mass for PETSc's native form. Instead PETSc TS owns step control/convergence
// while Exasim supplies the per-step already-time-folded condensed residual +
// Jacobian. Since the Exasim residual F(U)=H*U-b has NO explicit Udot dependence
// (the time term is inside H and b via UpdateSource/fc_u), dF/dUdot=0 and the
// consistent Jacobian is simply J=H -> PETSc TS reduces to a clean loop driver,
// with the time history threaded through PreStep/PostStep.
//
// Per step (PreStep): PreviousSolutions (save udg^n) -> UpdateSource (dtfactor=1/dt,
// history sdg) -> hdgAssembleLinearSystem (time-augmented H,b) -> ComputeHDGPreconditioner.
// IFunction: F = H*U - b.  IJacobian: J = MatShell(res.H).  PostStep: recover udg^{n+1}.
//
// Validation: run the IDENTICAL per-step sequence with Exasim-native gmres (a hand BE
// loop) on a SEPARATE discretization, then PETSc TS on the main one; compare the final
// trace. Result: rel ||uh_petsc - uh_native|| ~ 1.9e-15 over 10 backward-Euler steps.
//
// Two bugs were needed to make the multi-step loop correct (see prepareStep / the
// separate-disc reference): (1) after UpdateSource the nodal history source sol.sdg must
// be interpolated to Gauss points sol.sdgg (the assembly reads sdgg) — SteadyProblem does
// this; without it the time term miscancels and the trajectory freezes after step 1.
// (2) the native reference and the PETSc run must use separate disc instances — sharing
// one bleeds time-history state (initializeSolution does not fully reset it after N steps).

#include <petscts.h>

#include <exasim/operators.hpp>
#include <exasim/export.hpp>
#include "my_model.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

struct OpCtx {
    CDiscretization*            disc = nullptr;
    CResidual<Poisson2D>*       res  = nullptr;
    CAssembler<Poisson2D>*      asmb = nullptr;
    CPreconditioner<Poisson2D>* prec = nullptr;
    sysstruct*                  sys  = nullptr;
    Int                         N = 0, backend = 0;
    double                      dt = 0.0;
    std::vector<dstype>         b;          // current step's condensed RHS (refreshed in PreStep)
    std::vector<dstype>         uh_n;       // step-start trace (linearization origin for IFunction)
    std::vector<dstype>         work;       // scratch for (U - uh_n)
};

// ---- shared backend pieces -------------------------------------------------

// Assemble the time-augmented operator for one backward-Euler step at the current
// state (sol.udg/sol.uh = step-start values). Sets res.H/res.K and writes the
// condensed RHS into out_b. Mirrors the native DIRK stage setup.
static void prepareStep(OpCtx& c)
{
    auto& disc = *c.disc;
    disc.common.timestate.currentstep  = 0;          // fixed dt -> dt[0]; history is in sol/sys
    PreviousSolutions(disc.sol, *c.sys, disc.common, c.backend);          // udgprev <- udg^n
    disc.common.timestate.currentstage = 0;
    UpdateSource(disc.sol, *c.sys, disc.app, disc.driver_abi, disc.res, disc.common, c.backend);
    // Interpolate the NODAL history source sol.sdg -> GAUSS-point sol.sdgg. The HDG
    // assembly (uequation.hpp) reads sdgg, not sdg; SteadyProblem (solution.cpp) does
    // exactly this Node2Gauss step before its Newton solve. Without it sdgg stays stale
    // (zero), the time term miscancels the spatial residual, and the trajectory freezes.
    if (disc.common.components.ncs > 0) {
        for (Int j = 0; j < disc.common.meshsizes.nbe; j++) {
            Int e1 = disc.common.eblks[3*j] - 1;
            Int e2 = disc.common.eblks[3*j+1];
            GetElemNodes(disc.tmp.tempn, disc.sol.sdg, disc.common.grid.npe,
                         disc.common.components.ncs, 0, disc.common.components.ncs, e1, e2);
            Node2Gauss(disc.common.cublasHandle,
                       &disc.sol.sdgg[disc.common.grid.nge*disc.common.components.ncs*e1],
                       disc.tmp.tempn, disc.master.shapegt, disc.common.grid.nge,
                       disc.common.grid.npe, (e2-e1)*disc.common.components.ncs, c.backend);
        }
    }
    if (disc.common.components.ncq > 0)
        hdgGetQ(disc.sol.udg, disc.sol.uh, disc.sol, disc.res, disc.mesh, disc.tmp, disc.common, c.backend);
    c.asmb->hdgAssembleLinearSystem(c.sys->b, c.backend);   // time-augmented H + b
    c.prec->ComputeHDGPreconditioner(disc, c.backend);      // time-augmented K
    for (Int i = 0; i < c.N; ++i) { c.b[(size_t)i] = c.sys->b[i]; c.uh_n[(size_t)i] = disc.sol.uh[i]; }
}

// Recover the volume udg^{n+1} from the converged trace uh and accumulate the
// stage (UpdateSolution). After this sol.udg/sol.uh are the step-end state.
static void recoverStep(OpCtx& c, const dstype* uh)
{
    auto& disc = *c.disc;
    exasim::recover_volume(disc, uh, c.sys->x, c.backend);   // hdgGetDUDG + UpdateUDG + hdgGetQ
    UpdateSolution(disc.sol, *c.sys, disc.app, disc.driver_abi, disc.res, disc.tmp, disc.common, c.backend);
}

// ---- PETSc shims -----------------------------------------------------------

static PetscErrorCode MatMult_Exasim(Mat J, Vec V, Vec JV)
{
    OpCtx* c; PetscCall(MatShellGetContext(J, &c));
    const PetscScalar* v; PetscScalar* jv;
    PetscCall(VecGetArrayRead(V,&v)); PetscCall(VecGetArray(JV,&jv));
    c->asmb->evalMatVec(jv, const_cast<dstype*>(v), c->sys->u, c->sys->b, 1, c->backend);
    PetscCall(VecRestoreArrayRead(V,&v)); PetscCall(VecRestoreArray(JV,&jv));
    return PETSC_SUCCESS;
}
static PetscErrorCode PCApply_Exasim(PC pc, Vec V, Vec PV)
{
    OpCtx* c; PetscCall(PCShellGetContext(pc,&c));
    PetscCall(VecCopy(V, PV));
    PetscScalar* pv; PetscCall(VecGetArray(PV,&pv));
    c->prec->ApplyPreconditioner(pv, *c->sys, *c->disc, 1, c->backend);
    PetscCall(VecRestoreArray(PV,&pv));
    return PETSC_SUCCESS;
}
// Time-augmented condensed residual at trace U:  F(U) = H*(U - uh_n) - b.
// Exasim's per-step Newton solves H*duh=b for the INCREMENT (uh^{n+1}=uh^n+duh),
// so the residual is linearized about the step-start trace uh_n (not the origin).
// The time term is baked into H,b via UpdateSource/fc_u; no explicit Udot dependence.
static PetscErrorCode IFunction_Exasim(TS, PetscReal, Vec U, Vec /*Udot*/, Vec F, void* vctx)
{
    OpCtx* c = static_cast<OpCtx*>(vctx);
    const PetscScalar* u; PetscScalar* f;
    PetscCall(VecGetArrayRead(U,&u)); PetscCall(VecGetArray(F,&f));
    for (Int i=0;i<c->N;++i) c->work[(size_t)i] = u[i] - c->uh_n[(size_t)i];   // U - uh_n
    c->asmb->evalMatVec(f, c->work.data(), c->sys->u, c->sys->b, 1, c->backend); // H*(U-uh_n)
    for (Int i=0;i<c->N;++i) f[i] -= c->b[(size_t)i];                            // - b
    PetscCall(VecRestoreArrayRead(U,&u)); PetscCall(VecRestoreArray(F,&f));
    return PETSC_SUCCESS;
}
// J = H (constant MatShell, refreshed in PreStep); dF/dUdot=0 so no shift term.
// PETSc still expects the Jacobian matrix flagged assembled each call.
static PetscErrorCode IJacobian_Exasim(TS, PetscReal, Vec, Vec, PetscReal, Mat J, Mat P, void*)
{
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
    if (P != J) { PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY)); }
    return PETSC_SUCCESS;
}

static PetscErrorCode PreStep_Exasim(TS ts)
{
    OpCtx* c; PetscCall(TSGetApplicationContext(ts, &c));
    prepareStep(*c);   // refresh res.H/res.K and the step RHS c->b from the current state
    return PETSC_SUCCESS;
}
static PetscErrorCode PostStep_Exasim(TS ts)
{
    OpCtx* c; PetscCall(TSGetApplicationContext(ts, &c));
    Vec U; PetscCall(TSGetSolution(ts, &U));
    const PetscScalar* u; PetscCall(VecGetArrayRead(U,&u));
    { double n=0; for(Int i=0;i<c->N;++i) n+=u[i]*u[i]; PetscInt st; TSGetStepNumber(ts,&st);
      std::printf("[petsc] step %lld ||U||=%.10e\n",(long long)st,std::sqrt(n)); }
    recoverStep(*c, u);
    PetscCall(VecRestoreArrayRead(U,&u));
    return PETSC_SUCCESS;
}

static void unitSquareQuadMesh(int n, std::vector<double>& p, std::vector<int>& t, int& np, int& ne)
{
    const int nv=n+1; np=nv*nv; ne=n*n; p.resize((size_t)2*np); t.resize((size_t)4*ne);
    for(int iy=0;iy<nv;++iy)for(int ix=0;ix<nv;++ix){int j=iy*nv+ix;p[0+2*j]=(double)ix/n;p[1+2*j]=(double)iy/n;}
    int e=0; for(int iy=0;iy<n;++iy)for(int ix=0;ix<n;++ix,++e){
        t[0+4*e]=iy*nv+ix;t[1+4*e]=iy*nv+(ix+1);t[2+4*e]=(iy+1)*nv+(ix+1);t[3+4*e]=(iy+1)*nv+ix;}
}

int main(int argc, char** argv)
{
    PetscCall(PetscInitialize(&argc, &argv, nullptr, "Exasim heat: PETSc TS owns the loop\n"));
    Kokkos::initialize(argc, argv);
    int rc = 0;
    {
        const int backend=0; const Int mpiprocs=1,mpirank=0,fileoffset=0,omprank=0;
        constexpr double TOL=1e-8;
        const double dt=0.05; const int nsteps=10;

        int np=0,ne=0; std::vector<double> p; std::vector<int> t;
        unitSquareQuadMesh(8,p,t,np,ne);

        // PDE config + mesh/boundaries via the export helpers (no solver facade). Backward Euler.
        PDE pde = exasim::default_pde<Poisson2D>();
        pde.porder=3; pde.pgauss=6; pde.physicsparam={1.0};
        pde.tdep=1; pde.torder=1; pde.nstage=1; pde.dt={dt};
        exasim::MeshSpec mesh(p.data(),t.data(),np,ne,4);
        mesh.add_boundary(1,[](const double* x){return std::abs(x[1])      <TOL;});
        mesh.add_boundary(1,[](const double* x){return std::abs(x[0]-1.0)  <TOL;});
        mesh.add_boundary(1,[](const double* x){return std::abs(x[1]-1.0)  <TOL;});
        mesh.add_boundary(1,[](const double* x){return std::abs(x[0])      <TOL;});

        // A fresh in-memory Preprocessed from the same config. Used twice (once per disc)
        // so the Exasim-native reference and the PETSc run never share state.
        auto freshPre = [&]() { return exasim::make_preprocessed<Poisson2D>(pde, mesh, mpirank, mpiprocs); };

        CDiscretization disc(freshPre(), backend);
        CResidual<Poisson2D> residual(disc);
        CAssembler<Poisson2D> assembler(disc);
        CPreconditioner<Poisson2D> prec(disc,backend,ExasimExecutionMode::Solve);
        CSolver<Poisson2D> solv(disc,backend,ExasimExecutionMode::Solve);
        residual.initializeSolution(); residual.recoverInitialState(backend,false);
        TimestepCoefficents(disc.common);
        // Tighten Exasim-native GMRES (default 1e-8) to match PETSc's KSP rtol so the
        // per-step solves agree to machine precision and the trajectories don't drift.
        disc.common.solverparams.linearSolverTol = 1e-12;

        const Int N = disc.common.sizes.ndofuhat;
        std::printf("[heat-ts] N=%lld, tdep=%d, ncs=%lld, dt=%g, nsteps=%d\n",
                    (long long)N,(int)disc.common.timeparams.tdep,(long long)disc.common.components.ncs,dt,nsteps);

        OpCtx ctx; ctx.disc=&disc; ctx.res=&residual; ctx.asmb=&assembler; ctx.prec=&prec;
        ctx.sys=&solv.sys; ctx.N=N; ctx.backend=backend; ctx.dt=dt;
        ctx.b.assign((size_t)N,0.0); ctx.uh_n.assign((size_t)N,0.0); ctx.work.assign((size_t)N,0.0);

        // ================= Exasim-native BE reference (its OWN disc — no state shared with PETSc) =====
        // Same per-step operator as the PETSc path but solved with Exasim's own GMRES end-to-end, on
        // a separate discretization so the two runs cannot bleed time-history state into each other.
        std::vector<dstype> uh_native((size_t)N);
        {
            CDiscretization nd(freshPre(), backend);
            CResidual<Poisson2D> nr(nd); CAssembler<Poisson2D> na(nd);
            CPreconditioner<Poisson2D> npc(nd,backend,ExasimExecutionMode::Solve);
            CSolver<Poisson2D> ns(nd,backend,ExasimExecutionMode::Solve);
            nr.initializeSolution(); nr.recoverInitialState(backend,false);
            TimestepCoefficents(nd.common); nd.common.solverparams.linearSolverTol = 1e-12;
            OpCtx nc; nc.disc=&nd; nc.res=&nr; nc.asmb=&na; nc.prec=&npc; nc.sys=&ns.sys;
            nc.N=N; nc.backend=backend; nc.dt=dt;
            nc.b.assign((size_t)N,0.0); nc.uh_n.assign((size_t)N,0.0); nc.work.assign((size_t)N,0.0);
            for (int istep=0; istep<nsteps; ++istep) {
                prepareStep(nc);                                  // PreviousSolutions+UpdateSource+sdgg+assemble
                for (Int i=0;i<N;++i){ ns.sys.u[i]=nd.sol.uh[i]; ns.sys.x[i]=0.0; ns.sys.b[i]=nc.b[(size_t)i]; }
                ns.gmres(na, nd, npc, N, 1, backend);             // H*x=b (duh); uh^{n+1}=uh^n+x
                std::vector<dstype> uh_new((size_t)N);
                for (Int i=0;i<N;++i) uh_new[(size_t)i]=nd.sol.uh[i]+ns.sys.x[i];
                recoverStep(nc, uh_new.data());
            }
            for (Int i=0;i<N;++i) uh_native[(size_t)i]=nd.sol.uh[i];
        }

        // ================= PETSc TS owns the loop (on the main disc, fresh) =================
        Mat J; PetscCall(MatCreateShell(PETSC_COMM_SELF,N,N,N,N,&ctx,&J));
        PetscCall(MatShellSetOperation(J, MATOP_MULT,(void(*)(void))MatMult_Exasim));
        Vec U; PetscCall(MatCreateVecs(J,&U,nullptr));
        for (Int i=0;i<N;++i){ PetscCall(VecSetValue(U,i,disc.sol.uh[i],INSERT_VALUES)); }
        PetscCall(VecAssemblyBegin(U)); PetscCall(VecAssemblyEnd(U));

        TS ts; PetscCall(TSCreate(PETSC_COMM_SELF,&ts));
        PetscCall(TSSetApplicationContext(ts,&ctx));
        PetscCall(TSSetProblemType(ts,TS_NONLINEAR));
        PetscCall(TSSetType(ts,TSBEULER));
        PetscCall(TSSetIFunction(ts,nullptr,IFunction_Exasim,&ctx));
        PetscCall(TSSetIJacobian(ts,J,J,IJacobian_Exasim,&ctx));
        PetscCall(TSSetPreStep(ts,PreStep_Exasim));
        PetscCall(TSSetPostStep(ts,PostStep_Exasim));
        PetscCall(TSSetTimeStep(ts,dt));
        PetscCall(TSSetMaxSteps(ts,nsteps));
        PetscCall(TSSetMaxTime(ts,dt*nsteps));
        PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
        // linear per step -> single KSP solve, GMRES + our PCShell
        SNES snes; KSP ksp; PC pc;
        PetscCall(TSGetSNES(ts,&snes));
        PetscCall(SNESSetType(snes,SNESKSPONLY));
        PetscCall(SNESGetKSP(snes,&ksp));
        PetscCall(KSPSetType(ksp,KSPGMRES));
        PetscCall(KSPGetPC(ksp,&pc));
        PetscCall(PCSetType(pc,PCSHELL));
        PetscCall(PCShellSetContext(pc,&ctx));
        PetscCall(PCShellSetApply(pc,PCApply_Exasim));
        PetscCall(KSPSetTolerances(ksp,1e-12,1e-14,PETSC_DEFAULT,500));
        PetscCall(TSSetFromOptions(ts));

        PetscCall(TSSolve(ts,U));
        PetscInt steps; PetscReal ftime;
        PetscCall(TSGetStepNumber(ts,&steps)); PetscCall(TSGetTime(ts,&ftime));
        std::printf("[heat-ts] PETSc TS done: %lld steps, final t=%g\n",(long long)steps,(double)ftime);

        std::vector<dstype> uh_petsc((size_t)N);
        { const PetscScalar* u; PetscCall(VecGetArrayRead(U,&u));
          for (Int i=0;i<N;++i) uh_petsc[(size_t)i]=u[i]; PetscCall(VecRestoreArrayRead(U,&u)); }

        double dnum=0,dden=0,npn=0,nnn=0; bool finite=true;
        for (Int i=0;i<N;++i){
            if(!std::isfinite(uh_petsc[(size_t)i])||!std::isfinite(uh_native[(size_t)i])) finite=false;
            double d=uh_petsc[(size_t)i]-uh_native[(size_t)i];
            dnum+=d*d; dden+=uh_native[(size_t)i]*uh_native[(size_t)i];
            npn+=uh_petsc[(size_t)i]*uh_petsc[(size_t)i]; nnn+=uh_native[(size_t)i]*uh_native[(size_t)i]; }
        double relerr=std::sqrt(dnum)/(std::sqrt(dden)+1e-300);
        std::printf("[heat-ts] final ||uh_petsc||=%.8e  ||uh_native||=%.8e\n",std::sqrt(npn),std::sqrt(nnn));
        std::printf("[heat-ts] rel ||uh_petsc - uh_native|| = %.3e\n",relerr);

        if(!finite){ std::printf("[heat-ts] FAIL: non-finite\n"); rc=1; }
        else if((long long)steps!=nsteps){ std::printf("[heat-ts] FAIL: TS took %lld steps != %d\n",(long long)steps,nsteps); rc=1; }
        else if(!(std::sqrt(nnn)>0)){ std::printf("[heat-ts] FAIL: native trace zero\n"); rc=1; }
        else if(relerr>1e-6){ std::printf("[heat-ts] FAIL: PETSc TS vs native disagree\n"); rc=1; }
        else std::printf("[heat-ts] PASS: PETSc TS owns the loop and matches Exasim-native BE\n");

        PetscCall(VecDestroy(&U)); PetscCall(MatDestroy(&J)); PetscCall(TSDestroy(&ts));
    }
    Kokkos::finalize();
    PetscCall(PetscFinalize());
    return rc;
}
