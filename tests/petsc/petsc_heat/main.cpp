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
// No Exasim solver is constructed (no CSolver) -- PETSc owns every solve; the only Exasim
// objects are the exported operators + a standalone setsysstruct workspace. Validation is
// self-contained: the heat eq with this Poisson source has exact solution
// u = (1 - e^{-2 pi^2 t}) sin(pi x) sin(pi y), and eval_qoi gives the quadrature L2 distance
// to it (~5e-4 at t=0.5, the backward-Euler time-discretization error at dt=0.05).
//
// One subtlety made the multi-step loop correct (see prepareStep): after UpdateSource the
// nodal history source sol.sdg must be interpolated to Gauss points sol.sdgg (the assembly
// reads sdgg) -- SteadyProblem does this; without it the time term miscancels the spatial
// residual and the trajectory freezes after step 1. (During development this was confirmed
// against an Exasim-native BE loop to rel ~1.9e-15; that solver reference has been removed
// for the clean self-contained example.)

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
    exasim::recover_volume(disc, uh, c.sys->x);   // hdgGetDUDG + UpdateUDG + hdgGetQ
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
        pde.nsca=2; pde.nvec=1;   // Poisson2D vis fields: [u, ux+uy] + flux q  (for write_vtk)
        pde.nvqoi=2;              // qoi_volume = [ (u-u_exact)^2, u ]  (for the self-contained L2 check)
        exasim::MeshSpec mesh(p.data(),t.data(),np,ne,4);
        mesh.add_boundary(1,[](const double* x){return std::abs(x[1])      <TOL;});
        mesh.add_boundary(1,[](const double* x){return std::abs(x[0]-1.0)  <TOL;});
        mesh.add_boundary(1,[](const double* x){return std::abs(x[1]-1.0)  <TOL;});
        mesh.add_boundary(1,[](const double* x){return std::abs(x[0])      <TOL;});

        // A fresh in-memory Preprocessed from the same config. Used twice (once per disc)
        // so the Exasim-native reference and the PETSc run never share state.
        auto freshPre = [&]() { return exasim::make_preprocessed<Poisson2D>(pde, mesh, mpirank, mpiprocs); };

        // PETSc-driven path: the exported operators + a standalone operator-apply workspace
        // (setsysstruct, NOT a CSolver). PETSc owns every solve.
        CDiscretization disc(freshPre(), backend);
        CResidual<Poisson2D> residual(disc);
        CAssembler<Poisson2D> assembler(disc);
        CPreconditioner<Poisson2D> prec(disc,backend,ExasimExecutionMode::Solve);
        residual.initializeSolution(); residual.recoverInitialState(backend,false);
        TimestepCoefficents(disc.common);
        sysstruct sys;   // transient workspace (sys.udgprev etc.); NOT a solver
        setsysstruct(sys, disc.common, disc.res, disc.mesh, disc.tmp, backend);

        const Int N = disc.common.sizes.ndofuhat;
        std::printf("[heat-ts] N=%lld, tdep=%d, ncs=%lld, dt=%g, nsteps=%d\n",
                    (long long)N,(int)disc.common.timeparams.tdep,(long long)disc.common.components.ncs,dt,nsteps);

        OpCtx ctx; ctx.disc=&disc; ctx.res=&residual; ctx.asmb=&assembler; ctx.prec=&prec;
        ctx.sys=&sys; ctx.N=N; ctx.backend=backend; ctx.dt=dt;
        ctx.b.assign((size_t)N,0.0); ctx.uh_n.assign((size_t)N,0.0); ctx.work.assign((size_t)N,0.0);

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

        // write the final-time heat field to ParaView (PostStep already recovered udg)
        exasim::write_vtk<Poisson2D::Vis>(disc, "heat_petsc_final");

        // ================= self-contained verification (no Exasim solver) =================
        // The heat equation du/dt = mu*Lap(u) + f with this Poisson source (f = 2*pi^2 sin sin,
        // mu=1) has the exact solution u(x,y,t) = (1 - e^{-2 pi^2 t}) sin(pi x) sin(pi y): the
        // single mode sin(pi x) sin(pi y) (Laplacian eigenvalue -2 pi^2) relaxes to the steady
        // Poisson solution. By t = nsteps*dt = 0.5 the amplitude a(t)=1-e^{-2 pi^2 t} ~ 1 - 5e-5,
        // so the field has all but reached steady; eval_qoi gives the quadrature L2 distance to
        // that exact field (qoi_volume[0] = (u - u_exact)^2).
        std::vector<dstype> uh_petsc((size_t)N);
        { const PetscScalar* u; PetscCall(VecGetArrayRead(U,&u));
          for (Int i=0;i<N;++i) uh_petsc[(size_t)i]=u[i]; PetscCall(VecRestoreArrayRead(U,&u)); }
        bool finite=true; double nrm=0;
        for (Int i=0;i<N;++i){ if(!std::isfinite(uh_petsc[(size_t)i])) finite=false; nrm+=uh_petsc[(size_t)i]*uh_petsc[(size_t)i]; }
        nrm=std::sqrt(nrm);

        std::vector<dstype> qoi = exasim::eval_qoi<Poisson2D::QoI>(disc);   // qoi[0] = integral (u-u_exact)^2
        const double l2err = std::sqrt(qoi.empty() ? 1.0 : qoi[0]);
        const double tfinal = dt*nsteps;
        std::printf("[heat-ts] final ||uh|| = %.8e (steady ~ 1.131e+01)\n", nrm);
        std::printf("[heat-ts] L2 ||u - u_exact|| = %.3e at t=%.2f (heat warmed to near-steady)\n", l2err, tfinal);

        if(!finite){ std::printf("[heat-ts] FAIL: non-finite\n"); rc=1; }
        else if((long long)steps!=nsteps){ std::printf("[heat-ts] FAIL: TS took %lld steps != %d\n",(long long)steps,nsteps); rc=1; }
        else if(l2err>1e-3){ std::printf("[heat-ts] FAIL: solution far from exact (%.3e)\n", l2err); rc=1; }
        else std::printf("[heat-ts] PASS: PETSc TS owns the heat time loop on the exported operators\n");

        PetscCall(VecDestroy(&U)); PetscCall(MatDestroy(&J)); PetscCall(TSDestroy(&ts));
    }
    Kokkos::finalize();
    PetscCall(PetscFinalize());
    return rc;
}
