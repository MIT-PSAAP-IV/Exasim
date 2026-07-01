// PETSc drives Exasim's exported HDG operators — steady Poisson, CPU or GPU.
//
// This is the recipe a real external PETSc app writes. Exasim provides ONLY the
// discretization + operators; PETSc owns the entire solve (SNES + KSP=GMRES) and
// never touches Exasim internals — it calls back through three opaque shims:
//   MatShell MatMult -> CAssembler::evalMatVec              (apply res.H, HDG Jacobian)
//   PCShell  PCApply  -> CPreconditioner::ApplyPreconditioner (apply res.K)
//   SNES FormFunction -> the condensed residual G(uh) = H*uh - b0
// Steady Poisson is LINEAR so G is affine (b0 = condensed RHS at uh=0) and SNES
// converges in one Newton step.
//
// NO Exasim solver is constructed (no CSolver / CSolution / native GMRES). The
// only Exasim objects are the exported operators + a standalone sysstruct (the
// legacy operator-apply API takes a workspace struct; allocated directly with
// setsysstruct, NOT via the solver).
//
// CPU or GPU is chosen by EXASIM_BACKEND (0 = serial CPU, 2 = CUDA). The shims are
// backend-portable: PETSc's VecGetArray*AndMemType returns a host pointer for a
// VECSEQ vector and a *device* pointer for a VECCUDA vector, and the Exasim
// operators run on the matching backend — so on GPU the entire solve (matvec,
// preconditioner, Krylov vectors) stays on the device with no host staging.

#include <petscsnes.h>

#include <exasim/operators.hpp>      // FEM aggregation + preprocessing + in-memory ctor
#include <exasim/export.hpp>         // default_pde<M> / MeshSpec / make_preprocessed (no solver facade)
#include <exasim/petsc.hpp>          // exasim::petsc::Operator (MatShell/PCShell/Vec/SNES glue)

#include "poisson2d.hpp"              // Poisson2D

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void unitSquareQuadMesh(int n, std::vector<double>& p, std::vector<int>& t, int& np, int& ne)
{
    const int nv = n + 1; np = nv*nv; ne = n*n;
    p.resize((size_t)2*np); t.resize((size_t)4*ne);
    for (int iy=0; iy<nv; ++iy) for (int ix=0; ix<nv; ++ix) {
        const int j=iy*nv+ix; p[0+2*j]=(double)ix/n; p[1+2*j]=(double)iy/n; }
    int e=0;
    for (int iy=0; iy<n; ++iy) for (int ix=0; ix<n; ++ix,++e) {
        t[0+4*e]=iy*nv+ix; t[1+4*e]=iy*nv+(ix+1); t[2+4*e]=(iy+1)*nv+(ix+1); t[3+4*e]=(iy+1)*nv+ix; }
}

int main(int argc, char** argv)
{
    PetscCall(PetscInitialize(&argc, &argv, nullptr, "Exasim operators driven by PETSc (steady Poisson, CPU/GPU)\n"));
    Kokkos::initialize(argc, argv);
    int rc = 0;
    {
        // EXASIM_BACKEND: 0 = serial CPU (VECSEQ), 2 = CUDA (VECCUDA).
        const int backend = std::getenv("EXASIM_BACKEND") ? std::atoi(std::getenv("EXASIM_BACKEND")) : 0;
        const bool gpu = (backend >= 2);
        const Int mpiprocs=1, mpirank=0, fileoffset=0, omprank=0;
        constexpr double TOL = 1e-8, PI = 3.141592653589793;
        std::printf("[petsc] backend=%d (%s)\n", backend, gpu ? "CUDA" : "CPU");

        int np=0, ne=0; std::vector<double> p; std::vector<int> t;
        unitSquareQuadMesh(8, p, t, np, ne);

        // PDE config for the model (HDG defaults; override the few fields we care about).
        PDE pde = exasim::default_pde<Poisson2D>();
        pde.porder      = 3;
        pde.pgauss      = 6;
        pde.physicsparam = {1.0};
        pde.nvqoi       = 2;          // Poisson2D::qoi_volume outputs [ (u-u_exact)^2, u ]
        pde.nsca        = 2;          // Poisson2D::vis_scalars outputs [ u, ux+uy ]
        pde.nvec        = 1;          // Poisson2D::vis_vectors outputs the flux q

        // Mesh + boundary tags (all four sides Dirichlet, tag 1).
        exasim::MeshSpec mesh(p.data(), t.data(), np, ne, /*nve=*/4);
        mesh.add_boundary(1, [](const double* x){ return std::abs(x[1])      < TOL; });
        mesh.add_boundary(1, [](const double* x){ return std::abs(x[0]-1.0)  < TOL; });
        mesh.add_boundary(1, [](const double* x){ return std::abs(x[1]-1.0)  < TOL; });
        mesh.add_boundary(1, [](const double* x){ return std::abs(x[0])      < TOL; });

        // the ONLY Exasim objects: discretization + exported operators (on `backend`)
        CDiscretization disc(exasim::make_preprocessed<Poisson2D>(pde, mesh), backend);
        CResidual<Poisson2D>       residual(disc);
        CAssembler<Poisson2D>      assembler(disc);
        CPreconditioner<Poisson2D> prec(disc, backend, ExasimExecutionMode::Solve);
        residual.initializeSolution();
        residual.recoverInitialState(backend, /*postprocessOnly=*/false);

        sysstruct sys;   // operator-apply workspace (device buffers when backend>=2), NOT a CSolver
        setsysstruct(sys, disc.common, disc.res, disc.mesh, disc.tmp, backend);

        const Int N = disc.common.sizes.ndofuhat;
        std::printf("[petsc] in-memory operators built; N (trace dofs) = %lld\n", (long long)N);

        // assemble the condensed system at uh0=0:  res.H, res.K, sys.b
        if (disc.common.components.ncq > 0)
            hdgGetQ(disc.sol.udg, disc.sol.uh, disc.sol, disc.res, disc.mesh, disc.tmp, disc.common, backend);
        assembler.hdgAssembleLinearSystem(sys.b, backend);
        prec.ComputeHDGPreconditioner(disc, backend);

        // The PETSc glue: MatShell(res.H) + PCShell(res.K) + a zero-copy RHS Vec aliasing sys.b,
        // all backend/comm-aware. PETSc still owns the solve; we just set type/tolerances.
        exasim::petsc::Operator<Poisson2D> op(disc, assembler, prec, sys, PETSC_COMM_SELF);
        Vec Fr = op.make_vec();
        SNES snes = op.make_snes(Fr);
        KSP ksp; PetscCall(SNESGetKSP(snes, &ksp));
        PetscCall(KSPSetType(ksp, KSPGMRES));
        PetscCall(KSPSetTolerances(ksp, 1e-12, 1e-14, PETSC_DEFAULT, 500));
        PetscCall(SNESSetTolerances(snes, 1e-12, 1e-14, 1e-14, 50, 1000));
        PetscCall(SNESSetFromOptions(snes));

        Vec U = op.make_vec(); PetscCall(VecSet(U, 0.0));
        PetscCall(SNESSolve(snes, nullptr, U));
        SNESConvergedReason reason; PetscInt its;
        PetscCall(SNESGetConvergedReason(snes, &reason));
        PetscCall(SNESGetIterationNumber(snes, &its));
        std::printf("[petsc] SNES converged reason=%d, Newton iters=%lld\n", (int)reason, (long long)its);

        // ================= verification =================
        // (A) PETSc already drove the residual to tolerance -- read its own ||G(U)|| = ||H*uh - b0||.
        PetscReal fnorm; PetscCall(SNESGetFunctionNorm(snes, &fnorm));

        // (B) discretization error vs the exact solution, as a proper QUADRATURE norm via the
        //     model's exported volume QoI (qoi_volume[0] = (u - u_exact)^2). Recover the volume
        //     field from the trace first (uh stays in its own memory space -- device ptr on GPU).
        op.recover(U);
        std::vector<dstype> qoi = exasim::eval_qoi<Poisson2D::QoI>(disc);   // qoi[0] = integral (u-u_exact)^2
        const double l2err = std::sqrt(qoi.empty() ? 1.0 : qoi[0]);   // L2 error of (u - u_exact)

        // write the solution to ParaView (uses Exasim's vis pipeline; CG geometry built in-memory)
        exasim::write_vtk<Poisson2D::Vis>(disc, "poisson_petsc");

        std::printf("[petsc] (A) PETSc residual ||H*uh - b0||  = %.3e   (PETSc solved the exported system)\n", (double)fnorm);
        std::printf("[petsc] (B) L2 error ||u - u_exact||      = %.3e   (quadrature QoI)\n", l2err);

        bool finite = std::isfinite((double)fnorm) && std::isfinite(l2err);
        if (!finite)            { std::printf("[petsc] FAIL: non-finite result\n"); rc=1; }
        else if (reason < 0)    { std::printf("[petsc] FAIL: SNES did not converge\n"); rc=1; }
        else if (fnorm > 1e-8)  { std::printf("[petsc] FAIL: residual not driven to zero\n"); rc=1; }
        else if (l2err > 1e-3)  { std::printf("[petsc] FAIL: solution far from exact (%.3e)\n", l2err); rc=1; }
        else  std::printf("[petsc] PASS: PETSc solved the exported HDG Poisson operators on %s\n", gpu?"GPU":"CPU");

        PetscCall(VecDestroy(&U)); PetscCall(VecDestroy(&Fr)); PetscCall(SNESDestroy(&snes));
        // op destructs here -- destroys its MatShell + the RHS Vec.
    }
    Kokkos::finalize();
    PetscCall(PetscFinalize());
    return rc;
}
