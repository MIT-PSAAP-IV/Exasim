// PETSc drives Exasim's exported HDG operators IN PARALLEL (MPI) — steady Poisson.
//
// Same recipe as petsc_poisson, but distributed:
//   * The mesh is built in a SCALABLE way -- each rank fabricates ONLY its slice of the
//     global n x n quad mesh (a contiguous block of global nodes + a contiguous block of
//     global elements, with GLOBAL node indices). No rank ever holds the whole mesh.
//     exasim::make_preprocessed_distributed runs meshFromArraysDistributed + takeParallel,
//     which calls ParMETIS to REPARTITION for locality and builds the per-rank DMD.
//   * PETSc uses parallel Vec/Mat (PETSC_COMM_WORLD). Each rank owns N = ndofuhat trace
//     dofs. The MatMult shim is UNCHANGED from the serial case: CAssembler::evalMatVec ->
//     hdgMatVec does the inter-rank halo exchange (non-blocking send/recv over the DMD)
//     internally, so PETSc sees a globally consistent operator while only ever handling
//     each rank's owned dofs. The preconditioner is block-Jacobi (each rank's local res.K).
//
// NO Exasim solver is constructed; PETSc owns the solve. Verification is self-contained:
// recover the volume field, integrate the QoI (u - u_exact)^2 per rank, MPI_Allreduce to
// the global quadrature L2 error.

#include <petscsnes.h>
#include <mpi.h>

#include <exasim/operators.hpp>      // FEM aggregation + preprocessing + in-memory ctor
#include <exasim/export.hpp>         // default_pde / MeshSpecDistributed / make_preprocessed_distributed

#include "poisson2d.hpp"             // Poisson2D

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct OpCtx {
    CDiscretization*            disc = nullptr;
    CAssembler<Poisson2D>*      asmb = nullptr;
    CPreconditioner<Poisson2D>* prec = nullptr;
    sysstruct*                  sys  = nullptr;
    Vec                         B0   = nullptr;
    Int                         backend = 0;
};

// JV = H * V. evalMatVec -> hdgMatVec performs the inter-rank halo exchange internally,
// so this is identical to the serial shim even though V/JV are parallel vectors.
static PetscErrorCode MatMult_Exasim(Mat J, Vec V, Vec JV)
{
    OpCtx* c; PetscCall(MatShellGetContext(J, &c));
    const PetscScalar* v; PetscScalar* jv; PetscMemType mtv, mtj;
    PetscCall(VecGetArrayReadAndMemType(V, &v, &mtv));
    PetscCall(VecGetArrayWriteAndMemType(JV, &jv, &mtj));
    c->asmb->evalMatVec(jv, const_cast<dstype*>(v), c->sys->u, c->sys->b, /*spatialScheme=*/1, c->backend);
    PetscCall(VecRestoreArrayWriteAndMemType(JV, &jv));
    PetscCall(VecRestoreArrayReadAndMemType(V, &v));
    return PETSC_SUCCESS;
}

// PV = K^{-1} V  (block-Jacobi: each rank applies its local res.K, in place).
static PetscErrorCode PCApply_Exasim(PC pc, Vec V, Vec PV)
{
    OpCtx* c; PetscCall(PCShellGetContext(pc, &c));
    PetscCall(VecCopy(V, PV));
    PetscScalar* pv; PetscMemType mt; PetscCall(VecGetArrayAndMemType(PV, &pv, &mt));
    c->prec->ApplyPreconditioner(pv, *c->sys, *c->disc, /*spatialScheme=*/1, c->backend);
    PetscCall(VecRestoreArrayAndMemType(PV, &pv));
    return PETSC_SUCCESS;
}

// F = G(U) = H*U - b0  (steady Poisson is linear).
static PetscErrorCode FormFunction_Exasim(SNES, Vec U, Vec F, void* ctx)
{
    OpCtx* c = static_cast<OpCtx*>(ctx);
    const PetscScalar* u; PetscScalar* f; PetscMemType mtu, mtf;
    PetscCall(VecGetArrayReadAndMemType(U, &u, &mtu));
    PetscCall(VecGetArrayWriteAndMemType(F, &f, &mtf));
    c->asmb->evalMatVec(f, const_cast<dstype*>(u), c->sys->u, c->sys->b, 1, c->backend);
    PetscCall(VecRestoreArrayWriteAndMemType(F, &f));
    PetscCall(VecRestoreArrayReadAndMemType(U, &u));
    PetscCall(VecAXPY(F, -1.0, c->B0));
    return PETSC_SUCCESS;
}
static PetscErrorCode FormJacobian_Exasim(SNES, Vec, Mat, Mat, void*) { return PETSC_SUCCESS; }

int main(int argc, char** argv)
{
    PetscCall(PetscInitialize(&argc, &argv, nullptr,
              "Exasim HDG operators driven by PETSc IN PARALLEL (steady Poisson)\n"));
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &nprocs);
    // Exasim's inter-rank halo exchange (hdgMatVec) communicates over EXASIM_COMM_LOCAL;
    // point both Exasim communicators at PETSc's world so the operator matvec is consistent.
    EXASIM_COMM_WORLD = PETSC_COMM_WORLD;
    EXASIM_COMM_LOCAL = PETSC_COMM_WORLD;
    const int backend = std::getenv("EXASIM_BACKEND") ? std::atoi(std::getenv("EXASIM_BACKEND")) : 0;
    // Bind one GPU per rank BEFORE Kokkos initializes so each rank drives its own device
    // (single node: device id = rank). Harmless for the CPU (Serial) backend.
    { Kokkos::InitializationSettings ks;
      if (backend >= 2) ks.set_device_id(rank);
      Kokkos::initialize(ks); }
    int rc = 0;
    {
        constexpr double TOL = 1e-8;

        // ----- scalable distributed mesh: each rank builds ONLY its slice -----
        const int n = 32;                                   // global 32 x 32 quad mesh
        const int np_global = (n+1)*(n+1), ne_global = n*n;
        const long np0 = (long)rank*np_global/nprocs, np1 = (long)(rank+1)*np_global/nprocs;
        const long e0  = (long)rank*ne_global/nprocs, e1  = (long)(rank+1)*ne_global/nprocs;
        const int np_local = (int)(np1-np0), ne_local = (int)(e1-e0);

        std::vector<double> p_local((size_t)2*np_local);    // this rank's node coords
        for (long j=np0; j<np1; ++j) {
            const int ix=(int)(j%(n+1)), iy=(int)(j/(n+1)); const long k=j-np0;
            p_local[2*k+0]=(double)ix/n; p_local[2*k+1]=(double)iy/n;
        }
        std::vector<int> t_local((size_t)4*ne_local);       // this rank's elements (GLOBAL node ids)
        for (long e=e0; e<e1; ++e) {
            const int ex=(int)(e%n), ey=(int)(e/n); const long k=e-e0;
            t_local[4*k+0]=ey*(n+1)+ex; t_local[4*k+1]=ey*(n+1)+ex+1;
            t_local[4*k+2]=(ey+1)*(n+1)+ex+1; t_local[4*k+3]=(ey+1)*(n+1)+ex;
        }
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "[mpi] %d ranks; global %d nodes / %d elems; rank-local slice ~ %d nodes / %d elems\n",
            nprocs, np_global, ne_global, np_local, ne_local));

        PDE pde = exasim::default_pde<Poisson2D>();
        pde.porder=3; pde.pgauss=6; pde.physicsparam={1.0};
        pde.nvqoi=2;  pde.nsca=2; pde.nvec=1;

        exasim::MeshSpecDistributed mesh(p_local.data(), t_local.data(), np_local, ne_local,
                                         np_global, ne_global, /*nve=*/4);
        mesh.add_boundary(1, [](const double* x){ return std::abs(x[1])      < TOL; });
        mesh.add_boundary(1, [](const double* x){ return std::abs(x[0]-1.0)  < TOL; });
        mesh.add_boundary(1, [](const double* x){ return std::abs(x[1]-1.0)  < TOL; });
        mesh.add_boundary(1, [](const double* x){ return std::abs(x[0])      < TOL; });

        // make_preprocessed_distributed -> ParMETIS repartition + per-rank DMD; MPI ctor.
        CDiscretization disc(exasim::make_preprocessed_distributed<Poisson2D>(pde, mesh, PETSC_COMM_WORLD),
                             backend, nprocs, rank);
        CResidual<Poisson2D>       residual(disc);
        CAssembler<Poisson2D>      assembler(disc);
        CPreconditioner<Poisson2D> prec(disc, backend, ExasimExecutionMode::Solve);
        residual.initializeSolution();
        residual.recoverInitialState(backend, /*postprocessOnly=*/false);

        sysstruct sys;
        setsysstruct(sys, disc.common, disc.res, disc.mesh, disc.tmp, backend);

        const Int N = disc.common.sizes.ndofuhat;           // this rank's owned trace dofs
        long Nglob=0, Nl=(long)N; MPI_Allreduce(&Nl,&Nglob,1,MPI_LONG,MPI_SUM,PETSC_COMM_WORLD);
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[mpi] after ParMETIS: global trace dofs N = %ld\n", Nglob));

        if (disc.common.components.ncq > 0)
            hdgGetQ(disc.sol.udg, disc.sol.uh, disc.sol, disc.res, disc.mesh, disc.tmp, disc.common, backend);
        assembler.hdgAssembleLinearSystem(sys.b, backend);
        prec.ComputeHDGPreconditioner(disc, backend);

        // parallel RHS Vec aliasing sys.b (local part). sys.b is a DEVICE buffer when gpu, so
        // wrap it with the CUDA variant; both are zero-copy (PETSc + the operators share it).
        const bool gpu = (backend >= 2);
        Vec B0;
        if (gpu) PetscCall(VecCreateMPICUDAWithArray(PETSC_COMM_WORLD, 1, N, PETSC_DECIDE, sys.b, &B0));
        else     PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, N, PETSC_DECIDE, sys.b, &B0));
        OpCtx ctx; ctx.disc=&disc; ctx.asmb=&assembler; ctx.prec=&prec; ctx.sys=&sys; ctx.B0=B0; ctx.backend=backend;

        Mat J; PetscCall(MatCreateShell(PETSC_COMM_WORLD, N, N, PETSC_DETERMINE, PETSC_DETERMINE, &ctx, &J));
        PetscCall(MatShellSetOperation(J, MATOP_MULT, (void(*)(void))MatMult_Exasim));
        if (gpu) PetscCall(MatShellSetVecType(J, VECCUDA));
        Vec U, Fr; PetscCall(VecDuplicate(B0, &U)); PetscCall(VecDuplicate(B0, &Fr));
        PetscCall(VecSet(U, 0.0));

        SNES snes; PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
        PetscCall(SNESSetFunction(snes, Fr, FormFunction_Exasim, &ctx));
        PetscCall(SNESSetJacobian(snes, J, J, FormJacobian_Exasim, &ctx));
        KSP ksp; PC pc;
        PetscCall(SNESGetKSP(snes, &ksp));
        PetscCall(KSPSetType(ksp, KSPGMRES));
        PetscCall(KSPGetPC(ksp, &pc));
        PetscCall(PCSetType(pc, PCSHELL));
        PetscCall(PCShellSetContext(pc, &ctx));
        PetscCall(PCShellSetApply(pc, PCApply_Exasim));
        PetscCall(KSPSetTolerances(ksp, 1e-10, 1e-12, PETSC_DEFAULT, 1000));
        PetscCall(SNESSetTolerances(snes, 1e-10, 1e-12, 1e-12, 50, 2000));
        PetscCall(SNESSetFromOptions(snes));

        PetscCall(SNESSolve(snes, nullptr, U));
        SNESConvergedReason reason; PetscInt its; PetscReal fnorm;
        PetscCall(SNESGetConvergedReason(snes, &reason));
        PetscCall(SNESGetIterationNumber(snes, &its));
        PetscCall(SNESGetFunctionNorm(snes, &fnorm));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "[mpi] SNES reason=%d, Newton iters=%lld, ||H*uh - b0|| = %.3e\n",
            (int)reason, (long long)its, (double)fnorm));

        // ----- verification: recover volume, integrate QoI per rank, reduce to global L2 -----
        { const PetscScalar* u; PetscMemType mt; PetscCall(VecGetArrayReadAndMemType(U, &u, &mt));
          exasim::recover_volume(disc, const_cast<dstype*>(u), sys.x);
          PetscCall(VecRestoreArrayReadAndMemType(U, &u)); }
        std::vector<dstype> qoi = exasim::eval_qoi<Poisson2D::QoI>(disc);  // local integral (u-u_exact)^2
        double local = qoi.empty() ? 0.0 : (double)qoi[0], global = 0.0;
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
        const double l2err = std::sqrt(global);

        // parallel ParaView output (per-rank .vtu + .pvtu)
        exasim::write_vtk<Poisson2D::Vis>(disc, "poisson_petsc_mpi");

        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "[mpi] global L2 error ||u - u_exact|| = %.3e (quadrature QoI, summed over ranks)\n", l2err));

        bool finite = std::isfinite((double)fnorm) && std::isfinite(l2err);
        int bad = (!finite) || (reason < 0) || (fnorm > 1e-6) || (l2err > 1e-3);
        if (rank == 0) {
            if (bad) std::printf("[mpi] FAIL\n");
            else     std::printf("[mpi] PASS: PETSc solved the exported HDG operators across %d ranks "
                                 "(scalable mesh + ParMETIS partition)\n", nprocs);
        }
        rc = bad ? 1 : 0;

        PetscCall(VecDestroy(&U)); PetscCall(VecDestroy(&Fr)); PetscCall(VecDestroy(&B0));
        PetscCall(MatDestroy(&J)); PetscCall(SNESDestroy(&snes));
    }
    Kokkos::finalize();
    PetscCall(PetscFinalize());
    return rc;
}
