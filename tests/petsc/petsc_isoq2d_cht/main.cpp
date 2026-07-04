// PETSc version of the isoq2d CHT SOLID transient — time stepping on Exasim's side.
//
// Interpretation (see docs/internals/petsc-isoq2d-cht.md): reading (B). PETSc TS
// owns the transient time loop; Exasim owns the HDG spatial discretization + the
// backward-Euler time term (rho*cp mass folded into H,b via UpdateSource/tdfunc)
// and exports the matrix-free operators (res.H MatShell + res.K PCShell). This is
// the isoq2d CHT SOLID thermal step (rho*cp dT/dt = div(k grad T)), which the
// legacy app runs on Summit; here it runs IN-PROCESS on Exasim's operators driven
// by the SAME summit-dev conda PETSc. The compressible-NS fluid + the live
// Dirichlet-Neumann interface coupling are deferred (documented in the design doc);
// this first increment does the transient solid on a slab sized to the isoq wall,
// with the fluid->solid heat-flux transfer represented by the ib==2 interface
// boundary (exercised here with a stubbed constant flux).
//
// Structure mirrors tests/petsc/petsc_heat exactly; the deltas are the HeatSolid2D
// model (real solid material props + interface-flux BC) and the verification, which
// additionally checks the rho*cp transient RATE (not just the steady state).

#include <petscts.h>

#include <exasim/operators.hpp>
#include <exasim/export.hpp>
#include <exasim/petsc.hpp>
#include "heatsolid2d.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using Model = HeatSolid2D;

struct OpCtx {
    CDiscretization*         disc = nullptr;
    CResidual<Model>*        res  = nullptr;
    CAssembler<Model>*       asmb = nullptr;
    CPreconditioner<Model>*  prec = nullptr;
    sysstruct*               sys  = nullptr;
    exasim::petsc::Operator<Model>* op = nullptr;
    Int                      N = 0, backend = 0;
    double                   dt = 0.0;
    std::vector<dstype>      b, uh_n;
    std::vector<double>      stepNorm;      // ||U|| recorded per PostStep (transient check)
    Vec                      bVec = nullptr, workVec = nullptr;
};

// Assemble the time-augmented backward-Euler operator for one step at the current state.
static void prepareStep(OpCtx& c)
{
    auto& disc = *c.disc;
    disc.common.timestate.currentstep  = 0;
    PreviousSolutions(disc.sol, *c.sys, disc.common, c.backend);
    disc.common.timestate.currentstage = 0;
    UpdateSource(disc.sol, *c.sys, disc.app, disc.driver_abi, disc.res, disc.common, c.backend);
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
    c.asmb->hdgAssembleLinearSystem(c.sys->b, c.backend);
    c.prec->ComputeHDGPreconditioner(disc, c.backend);
    for (Int i = 0; i < c.N; ++i) { c.b[(size_t)i] = c.sys->b[i]; c.uh_n[(size_t)i] = disc.sol.uh[i]; }
}

static void recoverStep(OpCtx& c, const dstype* uh)
{
    auto& disc = *c.disc;
    exasim::recover_volume(disc, uh, c.sys->x);
    UpdateSolution(disc.sol, *c.sys, disc.app, disc.driver_abi, disc.res, disc.tmp, disc.common, c.backend);
}

static PetscErrorCode IFunction_Exasim(TS, PetscReal, Vec U, Vec, Vec F, void* vctx)
{
    OpCtx* c = static_cast<OpCtx*>(vctx);
    const PetscScalar* u; PetscScalar* w;
    PetscCall(VecGetArrayRead(U, &u));
    PetscCall(VecGetArray(c->workVec, &w));
    for (Int i=0;i<c->N;++i) w[i] = u[i] - c->uh_n[(size_t)i];
    PetscCall(VecRestoreArray(c->workVec, &w));
    PetscCall(VecRestoreArrayRead(U, &u));
    PetscCall(MatMult(c->op->mat(), c->workVec, F));
    PetscCall(VecAXPY(F, -1.0, c->bVec));
    return PETSC_SUCCESS;
}
static PetscErrorCode IJacobian_Exasim(TS, PetscReal, Vec, Vec, PetscReal, Mat J, Mat P, void*)
{
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
    if (P != J) { PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY)); }
    return PETSC_SUCCESS;
}
static PetscErrorCode PreStep_Exasim(TS ts)
{
    OpCtx* c; PetscCall(TSGetApplicationContext(ts, &c));
    prepareStep(*c);
    PetscScalar* bv; PetscCall(VecGetArray(c->bVec, &bv));
    for (Int i=0;i<c->N;++i) bv[i] = c->b[(size_t)i];
    PetscCall(VecRestoreArray(c->bVec, &bv));
    return PETSC_SUCCESS;
}
static PetscErrorCode PostStep_Exasim(TS ts)
{
    OpCtx* c; PetscCall(TSGetApplicationContext(ts, &c));
    Vec U; PetscCall(TSGetSolution(ts, &U));
    const PetscScalar* u; PetscCall(VecGetArrayRead(U,&u));
    double n=0; for(Int i=0;i<c->N;++i) n+=u[i]*u[i]; n=std::sqrt(n);
    PetscInt st; TSGetStepNumber(ts,&st);
    c->stepNorm.push_back(n);
    std::printf("[isoq-cht] step %2lld  ||U||=%.8e\n",(long long)st,n);
    recoverStep(*c, u);
    PetscCall(VecRestoreArrayRead(U,&u));
    return PETSC_SUCCESS;
}

static void slabQuadMesh(int nx, int ny, double Lx, double Ly,
                         std::vector<double>& p, std::vector<int>& t, int& np, int& ne)
{
    const int nvx=nx+1, nvy=ny+1; np=nvx*nvy; ne=nx*ny;
    p.resize((size_t)2*np); t.resize((size_t)4*ne);
    for(int iy=0;iy<nvy;++iy)for(int ix=0;ix<nvx;++ix){
        int j=iy*nvx+ix; p[0+2*j]=Lx*ix/nx; p[1+2*j]=Ly*iy/ny; }
    int e=0; for(int iy=0;iy<ny;++iy)for(int ix=0;ix<nx;++ix,++e){
        t[0+4*e]=iy*nvx+ix; t[1+4*e]=iy*nvx+(ix+1);
        t[2+4*e]=(iy+1)*nvx+(ix+1); t[3+4*e]=(iy+1)*nvx+ix; }
}

// Build a fresh Preprocessed for the slab. `interfaceBottom` tags the bottom edge
// with ib==2 (interface heat-flux BC); otherwise all edges are ib==1 (Dirichlet).
static PDE makeConfig(double Lx, double Ly, double dt, double k, double rho, double cp,
                      double qn, double srcScale)
{
    PDE pde = exasim::default_pde<Model>();
    pde.porder=3; pde.pgauss=6;
    pde.physicsparam = {k, rho, cp, qn, Lx, Ly, srcScale};
    pde.tdep=1; pde.torder=1; pde.nstage=1; pde.dt={dt};
    pde.nsca=2; pde.nvec=1; pde.nvqoi=2;
    return pde;
}

int main(int argc, char** argv)
{
    PetscCall(PetscInitialize(&argc, &argv, nullptr,
        "PETSc-driven isoq2d CHT solid transient (Exasim owns the discretization)\n"));
    Kokkos::initialize(argc, argv);
    int rc = 0;
    {
        const int backend=0; const Int mpiprocs=1, mpirank=0;
        constexpr double TOL=1e-9;

        // isoq2d CHT solid material (thermal.yaml) + slab sized to the isoq wall bbox.
        const double k=0.5, rho=280.0, cp=2000.0;
        const double Lx=0.06, Ly=0.051;
        const double dt=60.0; const int nsteps=25;
        const int nx=12, ny=10;

        const double gamma  = M_PI*M_PI*(1.0/(Lx*Lx) + 1.0/(Ly*Ly));
        const double lambda = k*gamma/(rho*cp);       // continuous decay rate
        std::printf("[isoq-cht] material k=%.3g rho=%.1f cp=%.1f | slab %.3gx%.3g\n", k,rho,cp,Lx,Ly);
        std::printf("[isoq-cht] gamma=%.4g lambda=k*gamma/(rho*cp)=%.6g (tau=%.1fs), dt=%g nsteps=%d\n",
                    gamma,lambda,1.0/lambda,dt,nsteps);

        int np=0,ne=0; std::vector<double> p; std::vector<int> t;
        slabQuadMesh(nx,ny,Lx,Ly,p,t,np,ne);

        // ============ MMS run: PETSc TS drives the solid transient, all-Dirichlet ============
        PDE pde = makeConfig(Lx,Ly,dt,k,rho,cp,/*qn=*/0.0,/*srcScale=*/1.0);
        exasim::MeshSpec mesh(p.data(),t.data(),np,ne,4);
        mesh.add_boundary(1,[&](const double* x){return std::abs(x[1])   <TOL;}); // bottom  ib=1
        mesh.add_boundary(1,[&](const double* x){return std::abs(x[0]-Lx)<TOL;}); // right   ib=1
        mesh.add_boundary(1,[&](const double* x){return std::abs(x[1]-Ly)<TOL;}); // top     ib=1
        mesh.add_boundary(1,[&](const double* x){return std::abs(x[0])   <TOL;}); // left    ib=1

        CDiscretization disc(exasim::make_preprocessed<Model>(pde, mesh, mpirank, mpiprocs), backend);
        CResidual<Model> residual(disc);
        CAssembler<Model> assembler(disc);
        CPreconditioner<Model> prec(disc,backend,ExasimExecutionMode::Solve);
        residual.initializeSolution(); residual.recoverInitialState(backend,false);
        TimestepCoefficents(disc.common);
        sysstruct sys; setsysstruct(sys, disc.common, disc.res, disc.mesh, disc.tmp, backend);

        const Int N = disc.common.sizes.ndofuhat;
        std::printf("[isoq-cht] N=%lld ncs=%lld\n",(long long)N,(long long)disc.common.components.ncs);

        OpCtx ctx; ctx.disc=&disc; ctx.res=&residual; ctx.asmb=&assembler; ctx.prec=&prec;
        ctx.sys=&sys; ctx.N=N; ctx.backend=backend; ctx.dt=dt;
        ctx.b.assign((size_t)N,0.0); ctx.uh_n.assign((size_t)N,0.0);

        exasim::petsc::Operator<Model> op(disc, assembler, prec, sys, PETSC_COMM_SELF);
        ctx.op=&op; ctx.bVec=op.make_vec(); ctx.workVec=op.make_vec();

        Vec U = op.make_vec();
        for (Int i=0;i<N;++i) PetscCall(VecSetValue(U,i,disc.sol.uh[i],INSERT_VALUES));
        PetscCall(VecAssemblyBegin(U)); PetscCall(VecAssemblyEnd(U));

        TS ts; PetscCall(TSCreate(PETSC_COMM_SELF,&ts));
        PetscCall(TSSetApplicationContext(ts,&ctx));
        PetscCall(TSSetProblemType(ts,TS_NONLINEAR));
        PetscCall(TSSetType(ts,TSBEULER));
        PetscCall(TSSetIFunction(ts,nullptr,IFunction_Exasim,&ctx));
        PetscCall(TSSetIJacobian(ts,op.mat(),op.mat(),IJacobian_Exasim,&ctx));
        PetscCall(TSSetPreStep(ts,PreStep_Exasim));
        PetscCall(TSSetPostStep(ts,PostStep_Exasim));
        PetscCall(TSSetTimeStep(ts,dt));
        PetscCall(TSSetMaxSteps(ts,nsteps));
        PetscCall(TSSetMaxTime(ts,dt*nsteps));
        PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
        SNES snes; KSP ksp; PC pc;
        PetscCall(TSGetSNES(ts,&snes)); PetscCall(SNESSetType(snes,SNESKSPONLY));
        PetscCall(SNESGetKSP(snes,&ksp)); PetscCall(KSPSetType(ksp,KSPGMRES));
        PetscCall(KSPGetPC(ksp,&pc)); op.configure_pc(pc);
        PetscCall(KSPSetTolerances(ksp,1e-12,1e-14,PETSC_DEFAULT,500));
        PetscCall(TSSetFromOptions(ts));

        PetscCall(TSSolve(ts,U));
        PetscInt steps; PetscReal ftime;
        PetscCall(TSGetStepNumber(ts,&steps)); PetscCall(TSGetTime(ts,&ftime));
        std::printf("[isoq-cht] PETSc TS done: %lld steps, final t=%g\n",(long long)steps,(double)ftime);

        exasim::write_vtk<Model::Vis>(disc, "isoq2d_cht_solid_final");

        // ---- verification ----
        std::vector<dstype> uh_p((size_t)N);
        { const PetscScalar* u; PetscCall(VecGetArrayRead(U,&u));
          for (Int i=0;i<N;++i) uh_p[(size_t)i]=u[i]; PetscCall(VecRestoreArrayRead(U,&u)); }
        bool finite=true; double nrm=0;
        for (Int i=0;i<N;++i){ if(!std::isfinite(uh_p[(size_t)i])) finite=false; nrm+=uh_p[(size_t)i]*uh_p[(size_t)i]; }
        nrm=std::sqrt(nrm);

        // (1) steady L2: at near-steady time T ~ phi (manufactured steady solution)
        std::vector<dstype> qoi = exasim::eval_qoi<Model::QoI>(disc);
        const double l2err = std::sqrt(qoi.empty() ? 1.0 : qoi[0]);
        std::printf("[isoq-cht] (1) steady L2 ||T - phi|| = %.3e  (near-steady MMS)\n", l2err);

        // (2) transient/mass check: ||U_k||/||U_final|| must track the backward-Euler
        //     amplitude a_k = 1-(1+lambda*dt)^{-k}, lambda = k*gamma/(rho*cp). A wrong
        //     rho*cp mass would change lambda by orders of magnitude (steady jump at step 1).
        double transErr = 0.0; int kcheck = 5; double meas=0, expc=0;
        if ((int)ctx.stepNorm.size() >= nsteps && nsteps > kcheck) {
            auto ampl = [&](int kk){ return 1.0 - std::pow(1.0+lambda*dt, -(double)kk); };
            meas = ctx.stepNorm[(size_t)kcheck-1] / ctx.stepNorm[(size_t)nsteps-1];
            expc = ampl(kcheck) / ampl(nsteps);
            transErr = std::abs(meas - expc);
            std::printf("[isoq-cht] (2) transient a_%d ratio: measured=%.5f expected=%.5f |d|=%.4f\n",
                        kcheck, meas, expc, transErr);
        }

        // (3) exported-matrix consistency: assembled MATAIJ vs matrix-free op.mat()
        double asm_diff=0.0;
        {
            Vec v=op.make_vec(), y1=op.make_vec(), y2=op.make_vec();
            for (Int i=0;i<N;++i) PetscCall(VecSetValue(v,i,std::sin(0.7*i+0.1),INSERT_VALUES));
            PetscCall(VecAssemblyBegin(v)); PetscCall(VecAssemblyEnd(v));
            Mat A = exasim::petsc::assemble_matrix<Model>(disc, PETSC_COMM_SELF);
            PetscReal d,nn; PetscCall(MatMult(op.mat(),v,y1)); PetscCall(MatMult(A,v,y2));
            PetscCall(VecAXPY(y2,-1.0,y1)); PetscCall(VecNorm(y2,NORM_2,&d)); PetscCall(VecNorm(y1,NORM_2,&nn));
            asm_diff=(double)d/((double)nn+1e-300);
            PetscCall(MatDestroy(&A)); PetscCall(VecDestroy(&v)); PetscCall(VecDestroy(&y1)); PetscCall(VecDestroy(&y2));
            std::printf("[isoq-cht] (3) assembled MATAIJ vs matrix-free = %.3e\n", asm_diff);
        }

        if(!finite){ std::printf("[isoq-cht] FAIL: non-finite\n"); rc=1; }
        else if((long long)steps!=nsteps){ std::printf("[isoq-cht] FAIL: %lld steps != %d\n",(long long)steps,nsteps); rc=1; }
        else if(l2err>1e-3){ std::printf("[isoq-cht] FAIL: steady L2 too large (%.3e)\n", l2err); rc=1; }
        else if(!(meas>0.5 && meas<0.95)){ std::printf("[isoq-cht] FAIL: step-%d not mid-transient (%.3f)\n",kcheck,meas); rc=1; }
        else if(transErr>0.06){ std::printf("[isoq-cht] FAIL: transient rate off (mass term?) |d|=%.4f\n", transErr); rc=1; }
        else if(asm_diff>1e-10){ std::printf("[isoq-cht] FAIL: MATAIJ != matrix-free (%.3e)\n", asm_diff); rc=1; }
        else std::printf("[isoq-cht] PASS: PETSc TS drives the isoq2d CHT solid transient on exported operators\n");

        PetscCall(VecDestroy(&U)); PetscCall(VecDestroy(&ctx.bVec)); PetscCall(VecDestroy(&ctx.workVec)); PetscCall(TSDestroy(&ts));

        // ============ CHT-interface coupling HOOK demo (ib==2 interface heat flux) ============
        // Behavioral, not MMS: the fluid->solid transfer is stubbed as a CONSTANT wall flux qn
        // on the bottom edge (ib==2); source off. Verify the interface BC path runs, stays finite,
        // and drives a nonzero thermal response (the coupling entry point works end-to-end).
        {
            const double qn = -2000.0;   // outward-flux BC value (<0 injects heat)
            const int ndemo = 8;
            PDE pde2 = makeConfig(Lx,Ly,dt,k,rho,cp,qn,/*srcScale=*/0.0);
            exasim::MeshSpec mesh2(p.data(),t.data(),np,ne,4);
            mesh2.add_boundary(2,[&](const double* x){return std::abs(x[1])   <TOL;}); // bottom = INTERFACE ib=2
            mesh2.add_boundary(1,[&](const double* x){return std::abs(x[0]-Lx)<TOL;}); // right  ib=1
            mesh2.add_boundary(1,[&](const double* x){return std::abs(x[1]-Ly)<TOL;}); // top    ib=1
            mesh2.add_boundary(1,[&](const double* x){return std::abs(x[0])   <TOL;}); // left   ib=1

            CDiscretization disc2(exasim::make_preprocessed<Model>(pde2, mesh2, mpirank, mpiprocs), backend);
            CResidual<Model> res2(disc2); CAssembler<Model> asm2(disc2);
            CPreconditioner<Model> prec2(disc2,backend,ExasimExecutionMode::Solve);
            res2.initializeSolution(); res2.recoverInitialState(backend,false);
            TimestepCoefficents(disc2.common);
            sysstruct sys2; setsysstruct(sys2, disc2.common, disc2.res, disc2.mesh, disc2.tmp, backend);
            const Int N2 = disc2.common.sizes.ndofuhat;

            OpCtx c2; c2.disc=&disc2; c2.res=&res2; c2.asmb=&asm2; c2.prec=&prec2;
            c2.sys=&sys2; c2.N=N2; c2.backend=backend; c2.dt=dt;
            c2.b.assign((size_t)N2,0.0); c2.uh_n.assign((size_t)N2,0.0);
            exasim::petsc::Operator<Model> op2(disc2, asm2, prec2, sys2, PETSC_COMM_SELF);
            c2.op=&op2; c2.bVec=op2.make_vec(); c2.workVec=op2.make_vec();

            Vec U2=op2.make_vec();
            for (Int i=0;i<N2;++i) PetscCall(VecSetValue(U2,i,disc2.sol.uh[i],INSERT_VALUES));
            PetscCall(VecAssemblyBegin(U2)); PetscCall(VecAssemblyEnd(U2));

            TS ts2; PetscCall(TSCreate(PETSC_COMM_SELF,&ts2));
            PetscCall(TSSetApplicationContext(ts2,&c2));
            PetscCall(TSSetProblemType(ts2,TS_NONLINEAR)); PetscCall(TSSetType(ts2,TSBEULER));
            PetscCall(TSSetIFunction(ts2,nullptr,IFunction_Exasim,&c2));
            PetscCall(TSSetIJacobian(ts2,op2.mat(),op2.mat(),IJacobian_Exasim,&c2));
            PetscCall(TSSetPreStep(ts2,PreStep_Exasim)); PetscCall(TSSetPostStep(ts2,PostStep_Exasim));
            PetscCall(TSSetTimeStep(ts2,dt)); PetscCall(TSSetMaxSteps(ts2,ndemo));
            PetscCall(TSSetMaxTime(ts2,dt*ndemo)); PetscCall(TSSetExactFinalTime(ts2,TS_EXACTFINALTIME_STEPOVER));
            SNES sn2; KSP kp2; PC pc2;
            PetscCall(TSGetSNES(ts2,&sn2)); PetscCall(SNESSetType(sn2,SNESKSPONLY));
            PetscCall(SNESGetKSP(sn2,&kp2)); PetscCall(KSPSetType(kp2,KSPGMRES));
            PetscCall(KSPGetPC(kp2,&pc2)); op2.configure_pc(pc2);
            PetscCall(KSPSetTolerances(kp2,1e-12,1e-14,PETSC_DEFAULT,500));

            std::printf("[isoq-cht] --- interface coupling-hook demo: ib==2 wall flux qn=%.1f ---\n", qn);
            PetscCall(TSSolve(ts2,U2));

            std::vector<dstype> u2((size_t)N2);
            { const PetscScalar* u; PetscCall(VecGetArrayRead(U2,&u));
              for (Int i=0;i<N2;++i) u2[(size_t)i]=u[i]; PetscCall(VecRestoreArrayRead(U2,&u)); }
            bool f2=true; for (Int i=0;i<N2;++i) if(!std::isfinite(u2[(size_t)i])) f2=false;
            std::vector<dstype> q2 = exasim::eval_qoi<Model::QoI>(disc2);
            const double meanT = q2.size()>1 ? q2[1] : 0.0;   // integral of T over the solid
            std::printf("[isoq-cht] interface demo: finite=%d  integral(T)=%.6e  (nonzero => flux drove heating)\n",
                        (int)f2, meanT);
            if(!f2){ std::printf("[isoq-cht] FAIL: interface demo non-finite\n"); rc=1; }
            else if(std::abs(meanT)<1e-8){ std::printf("[isoq-cht] FAIL: interface flux drove no response\n"); rc=1; }
            else std::printf("[isoq-cht] PASS: CHT interface heat-flux BC (ib==2) hook works end-to-end\n");

            PetscCall(VecDestroy(&U2)); PetscCall(VecDestroy(&c2.bVec)); PetscCall(VecDestroy(&c2.workVec)); PetscCall(TSDestroy(&ts2));
        }
    }
    Kokkos::finalize();
    PetscCall(PetscFinalize());
    return rc;
}
