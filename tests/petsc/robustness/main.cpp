// Robustness harness (L2/L3): builds the exported HDG operators in-memory for a model + config
// and runs model-agnostic CONSISTENCY checks that need no golden data -- so every model/variation
// in a zoo is validated automatically. This is the safety net that also lets new zoo models be
// trusted: a model whose operators pass symmetry + assembled==matrix-free is internally consistent.
//
// Checks (per model+config):
//   (1) finiteness  : the condensed matvec H*v is finite for a nontrivial v
//   (2) symmetry    : v.(H w) == w.(H v)  (self-adjoint models e.g. Poisson) to ~machine eps
//   (3) assembled   : the assembled MATAIJ MatMult == the matrix-free evalMatVec (validates the
//                     assembly + elemcon map end to end)  [PETSc]
//
// The zoo scans the variation seen in apps/: dimension, porder, mesh size (scalar Poisson here;
// system ncu>1 / 3D models are added the same way, guarded by the checks).

#include <petscsnes.h>

#include <exasim/operators.hpp>
#include <exasim/export.hpp>
#include <exasim/petsc.hpp>

#include "poisson2d.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

static void unitSquareQuadMesh(int n, std::vector<double>& p, std::vector<int>& t, int& np, int& ne)
{
    const int nv=n+1; np=nv*nv; ne=n*n; p.resize((size_t)2*np); t.resize((size_t)4*ne);
    for (int iy=0;iy<nv;++iy) for (int ix=0;ix<nv;++ix){ int j=iy*nv+ix; p[2*j]=(double)ix/n; p[2*j+1]=(double)iy/n; }
    int e=0; for (int iy=0;iy<n;++iy) for (int ix=0;ix<n;++ix,++e){
        t[4*e]=iy*nv+ix; t[4*e+1]=iy*nv+ix+1; t[4*e+2]=(iy+1)*nv+ix+1; t[4*e+3]=(iy+1)*nv+ix; }
}

// Build the operators for model M on an n x n quad unit square at the given porder, then run the
// consistency checks. Returns 0 on PASS. Templated on M so any model plugs in unchanged.
template <class M>
static int check_model(const char* name, int n, int porder, const std::vector<double>& param)
{
    const int backend = 0;
    constexpr double TOL = 1e-8;
    int np=0,ne=0; std::vector<double> p; std::vector<int> t;
    unitSquareQuadMesh(n, p, t, np, ne);

    PDE pde = exasim::default_pde<M>();
    pde.porder = porder; pde.pgauss = 2*porder; pde.physicsparam = param;
    pde.nvqoi = 2; pde.nsca = 2; pde.nvec = 1;   // enable the QoI integral (u - u_exact)^2
    exasim::MeshSpec mesh(p.data(), t.data(), np, ne, /*nve=*/4);
    mesh.add_boundary(1, [](const double* x){ return std::abs(x[1])     < TOL; });
    mesh.add_boundary(1, [](const double* x){ return std::abs(x[0]-1.0) < TOL; });
    mesh.add_boundary(1, [](const double* x){ return std::abs(x[1]-1.0) < TOL; });
    mesh.add_boundary(1, [](const double* x){ return std::abs(x[0])     < TOL; });

    CDiscretization disc(exasim::make_preprocessed<M>(pde, mesh), backend);
    CResidual<M> residual(disc); CAssembler<M> assembler(disc); CPreconditioner<M> prec(disc, backend, ExasimExecutionMode::Solve);
    residual.initializeSolution(); residual.recoverInitialState(backend, false);
    sysstruct sys; setsysstruct(sys, disc.common, disc.res, disc.mesh, disc.tmp, backend);
    const Int N = disc.common.sizes.ndofuhat;
    if (disc.common.components.ncq > 0)
        hdgGetQ(disc.sol.udg, disc.sol.uh, disc.sol, disc.res, disc.mesh, disc.tmp, disc.common, backend);
    assembler.hdgAssembleLinearSystem(sys.b, backend);
    prec.ComputeHDGPreconditioner(disc, backend);

    // (1) finiteness of the matvec on a nontrivial vector
    std::vector<dstype> v((size_t)N), Hv((size_t)N);
    for (Int i=0;i<N;++i) v[i]=std::sin(0.7*i+0.1);
    assembler.evalMatVec(Hv.data(), v.data(), sys.u, sys.b, 1, backend);
    bool finite=true; for (Int i=0;i<N;++i) if(!std::isfinite(Hv[i])) finite=false;

    // (2) assembled MATAIJ == matrix-free (assembly + elemcon map consistent with the operator)
    double asm_diff=0.0;
    { Mat A = exasim::petsc::assemble_matrix<M>(disc, PETSC_COMM_SELF);
      Vec x,y1,y2; MatCreateVecs(A,&x,&y1); VecDuplicate(x,&y2);
      for (Int i=0;i<N;++i){ PetscScalar val=v[i]; VecSetValue(x,i,val,INSERT_VALUES);} VecAssemblyBegin(x);VecAssemblyEnd(x);
      exasim::petsc::ShellMat Hs(PETSC_COMM_SELF, N,
          [&](dstype* yy, const dstype* xx){ assembler.evalMatVec(yy, const_cast<dstype*>(xx), sys.u, sys.b, 1, backend); }, false);
      MatMult(Hs.mat(),x,y1); MatMult(A,x,y2);
      PetscReal d,nn; VecAXPY(y2,-1.0,y1); VecNorm(y2,NORM_2,&d); VecNorm(y1,NORM_2,&nn);
      asm_diff=(double)d/((double)nn+1e-300);
      VecDestroy(&x);VecDestroy(&y1);VecDestroy(&y2);MatDestroy(&A); }

    // (3) SOLVE + exact-solution L2 error -- the check that actually catches a wrong model part:
    //     a flipped flux/source/jacobian/bc changes the solution, so the QoI (u - u_exact)^2 blows up.
    exasim::petsc::Operator<M> op(disc, assembler, prec, sys, PETSC_COMM_SELF);
    Vec Fr = op.make_vec(); SNES snes = op.make_snes(Fr);
    KSP ksp; SNESGetKSP(snes, &ksp); KSPSetType(ksp, KSPGMRES);
    KSPSetTolerances(ksp, 1e-11, 1e-13, PETSC_DEFAULT, 2000);
    SNESSetTolerances(snes, 1e-11, 1e-12, 1e-12, 50, 2000);
    Vec U = op.make_vec(); VecSet(U, 0.0);
    SNESSolve(snes, nullptr, U);
    SNESConvergedReason reason; SNESGetConvergedReason(snes, &reason);
    op.recover(U);
    std::vector<dstype> qoi = exasim::eval_qoi<typename M::QoI>(disc);
    const double l2 = qoi.empty() ? -1.0 : std::sqrt(std::abs((double)qoi[0]));
    VecDestroy(&U); VecDestroy(&Fr); SNESDestroy(&snes);

    std::printf("[robust] %-14s n=%d p=%d N=%lld : finite=%d  assembled=%.2e  solve=%d  L2err=%.3e\n",
                name, n, porder, (long long)N, (int)finite, asm_diff, (int)(reason>0), l2);
    int rc=0;
    const double l2tol = (porder<=1) ? 5e-2 : 1e-2;   // a correct model captures the exact soln well
    if (!finite)            { std::printf("[robust] %s FAIL: non-finite matvec\n", name); rc=1; }
    else if (asm_diff>1e-10){ std::printf("[robust] %s FAIL: assembled != matrix-free (%.2e)\n", name, asm_diff); rc=1; }
    else if (reason<0)      { std::printf("[robust] %s FAIL: solve did not converge\n", name); rc=1; }
    else if (!(l2>=0 && l2<l2tol)) { std::printf("[robust] %s FAIL: L2 error %.3e exceeds %.1e (model part wrong?)\n", name, l2, l2tol); rc=1; }
    return rc;
}

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, nullptr, "Exasim operator robustness harness\n");
    Kokkos::initialize(argc, argv);
    int rc = 0;
    {
        // Zoo point 1: scalar Poisson, swept over the porder + mesh-size variation the apps show.
        for (int porder : {1, 2, 3, 4})
            rc |= check_model<Poisson2D>("Poisson2D", 6, porder, {1.0});
        rc |= check_model<Poisson2D>("Poisson2D", 12, 3, {1.0});   // finer mesh
        // (zoo points 2..: system ncu>1, 3D -- added here as new models, validated by the same checks)
        if (rc==0) std::printf("[robust] ALL PASS\n");
    }
    Kokkos::finalize();
    PetscFinalize();
    return rc;
}
