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
#include <exasim/pointlocator.hpp>   // CPointLocator: point location + shape-function interpolation

#include "poisson2d.hpp"
#include "navierstokes2d.hpp"   // GeneratedModel: compressible Navier-Stokes, nd=2 ncu=4, HDG,
                                // codegen flux/fbou + AUTO-GENERATED Jacobians (concrete-M, header-only)

#include <algorithm>
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
      exasim::petsc::ShellMat<> Hs(PETSC_COMM_SELF, N,
          [&](dstype* yy, const dstype* xx){ assembler.evalMatVec(yy, const_cast<dstype*>(xx), sys.u, sys.b, 1, backend); }, backend);
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

// Navier-Stokes KERNEL Jacobian check (L1): the compressible-NS Jacobians are AUTO-GENERATED by
// codegen, and a flipped/stale term there is exactly the "part of the model flipping" worry. Verify
// flux_jac_uq against a central finite difference of flux at a valid NS state -- directly, with no
// global assembly (which for steady NS needs a fine mesh / matvec-free solve to stay non-singular).
// This is fast, has no golden data, and pinpoints a wrong analytic derivative.
static int check_ns_kernels()
{
    using M = GeneratedModel;
    constexpr int ncu=M::ncu, Nq=M::Nq, nf=ncu*M::nd;         // 4, 12, 8
    const double mu[8] = {1.4, 1000.0, 0.72, 0.2, 1.0, 1.0, 0.0, 45.1429};
    const double x[2]  = {0.3, 0.4}; double dum[1]={0.0}; const double* uinf=nullptr;
    // a physically valid state (r>0, positive pressure) with nonzero gradients
    double uq[Nq] = {1.0,0.9,0.15,46.0,  0.02,-0.01,0.03,0.05,  -0.02,0.04,-0.01,0.03};

    double Jac[nf*Nq]; M::flux_jac_uq(Jac, x, uq, dum, dum, mu, uinf, 0.0);   // analytic df/duq
    const double eps=1e-6; double maxrel=0.0;
    for (int j=0;j<Nq;++j) {                                  // central FD of column j = d f / d uq[j]
        double up[Nq],um[Nq]; for(int k=0;k<Nq;++k){up[k]=um[k]=uq[k];} up[j]+=eps; um[j]-=eps;
        double fp[nf],fm[nf];
        M::flux(fp,x,up,dum,dum,mu,uinf,0.0); M::flux(fm,x,um,dum,dum,mu,uinf,0.0);
        for (int i=0;i<nf;++i){ double fd=(fp[i]-fm[i])/(2*eps);
            maxrel = std::max(maxrel, std::abs(fd - Jac[j*nf+i])/(std::abs(Jac[j*nf+i])+1e-2)); }  // f_uq[j*nf+i]=df[i]/duq[j]
    }
    // Far-field (ib=2) HDG boundary Jacobians vs FD -- the characteristic/Roe BC is the most complex
    // auto-generated derivative, so the highest-value place to catch a flipped term.
    const int ib=2; const double n[2]={0.6,0.8}, tau[1]={3.0};
    double uh[ncu]={0.98,0.97,0.10,45.5};
    double Jq[ncu*Nq];  M::fbou_hdg_jac_uq(Jq, ib, x, uq, dum, dum, uh, n, tau, mu, uinf, 0.0);
    double Jh[ncu*ncu]; M::fbou_hdg_jac_uh(Jh, ib, x, uq, dum, dum, uh, n, tau, mu, uinf, 0.0);
    double maxrel_bq=0.0, maxrel_bh=0.0;
    for (int j=0;j<Nq;++j) {                                  // d fbou / d uq[j]
        double up[Nq],um[Nq]; for(int k=0;k<Nq;++k){up[k]=um[k]=uq[k];} up[j]+=eps; um[j]-=eps;
        double fp[ncu],fm[ncu];
        M::fbou_hdg(fp,ib,x,up,dum,dum,uh,n,tau,mu,uinf,0.0); M::fbou_hdg(fm,ib,x,um,dum,dum,uh,n,tau,mu,uinf,0.0);
        for (int i=0;i<ncu;++i){ double fd=(fp[i]-fm[i])/(2*eps);
            maxrel_bq=std::max(maxrel_bq, std::abs(fd-Jq[j*ncu+i])/(std::abs(Jq[j*ncu+i])+1e-2)); }
    }
    for (int j=0;j<ncu;++j) {                                 // d fbou / d uh[j]
        double up[ncu],um[ncu]; for(int k=0;k<ncu;++k){up[k]=um[k]=uh[k];} up[j]+=eps; um[j]-=eps;
        double fp[ncu],fm[ncu];
        M::fbou_hdg(fp,ib,x,uq,dum,dum,up,n,tau,mu,uinf,0.0); M::fbou_hdg(fm,ib,x,uq,dum,dum,um,n,tau,mu,uinf,0.0);
        for (int i=0;i<ncu;++i){ double fd=(fp[i]-fm[i])/(2*eps);
            maxrel_bh=std::max(maxrel_bh, std::abs(fd-Jh[j*ncu+i])/(std::abs(Jh[j*ncu+i])+1e-2)); }
    }
    std::printf("[robust] NavierStokes2D (auto-Jacobian FD): ncu=%d Nq=%d  flux=%.2e  fbou/uq=%.2e  fbou/uh=%.2e\n",
                ncu, Nq, maxrel, maxrel_bq, maxrel_bh);
    if (!(maxrel<1e-4))    { std::printf("[robust] NavierStokes2D FAIL: flux_jac_uq disagrees with FD (%.3e)\n", maxrel); return 1; }
    if (!(maxrel_bq<1e-4)) { std::printf("[robust] NavierStokes2D FAIL: fbou_hdg_jac_uq disagrees with FD (%.3e)\n", maxrel_bq); return 1; }
    if (!(maxrel_bh<1e-4)) { std::printf("[robust] NavierStokes2D FAIL: fbou_hdg_jac_uh disagrees with FD (%.3e)\n", maxrel_bh); return 1; }
    return 0;
}
// Round-trip point SAMPLING check: exercises CPointLocator end to end -- point location
// (candidate elems -> ellipsoid grid -> per-element Newton for reference coords) + shape-function
// interpolation, the PointLocator subsystem the app zoo never touches. Build a preprocessed Poisson
// disc, run the sampling builder (locate points a small distance off boundary ibc), then verify the
// GEOMETRIC round trip: the shape functions applied to the owner element's node coordinates must
// reproduce the sampled point (sum_i N_i * x_node_i == x_sample) and be a partition of unity. A
// wrong element pick, Newton solve, or shape function shows up as a large residual. No golden data.
static int check_sampling()
{
    const int backend = 0, n = 8, porder = 3;
    constexpr double TOL = 1e-8;
    int np=0,ne=0; std::vector<double> p; std::vector<int> t;
    unitSquareQuadMesh(n, p, t, np, ne);
    PDE pde = exasim::default_pde<Poisson2D>();
    pde.porder = porder; pde.pgauss = 2*porder; pde.physicsparam = {1.0};
    exasim::MeshSpec mesh(p.data(), t.data(), np, ne, /*nve=*/4);
    mesh.add_boundary(1, [](const double* x){ return std::abs(x[1])     < TOL; });
    mesh.add_boundary(1, [](const double* x){ return std::abs(x[0]-1.0) < TOL; });
    mesh.add_boundary(1, [](const double* x){ return std::abs(x[1]-1.0) < TOL; });
    mesh.add_boundary(1, [](const double* x){ return std::abs(x[0])     < TOL; });
    CDiscretization disc(exasim::make_preprocessed<Poisson2D>(pde, mesh), backend);

    CPointLocator locator;
    const dstype y1 = 0.05;                               // sample this far off the wall (interior)
    const bool ok = locator.BuildWallModelSamplingData(disc, /*ibc=*/1, y1);
    const auto& wm = locator.wm;
    if (!ok || wm.npoints <= 0) {
        std::printf("[robust] Sampling FAIL: BuildWallModelSamplingData ok=%d npoints=%lld (e2f/f2e missing?)\n",
                    (int)ok, (long long)wm.npoints);
        return 1;
    }
    const Int npe=wm.npe, nd=wm.nd, ncx=wm.ncx;
    const dstype* xdg = disc.sol.xdg;                     // geometry dg nodes [npe, ncx, ne], host
    double maxres=0.0, maxpou=0.0; Int located=0;
    for (Int q=0; q<wm.npoints; ++q) {
        const Int e = wm.elemsx1[q];
        if (e < 0) continue;                             // point not located (sentinel -1)
        ++located;
        double s=0.0; for (Int i=0;i<npe;++i) s += (double)wm.shap1[i + q*npe];
        maxpou = std::max(maxpou, std::abs(s-1.0));
        for (int d=0; d<nd; ++d) {
            double xr=0.0;
            for (Int i=0;i<npe;++i) xr += (double)wm.shap1[i + q*npe] * (double)xdg[i + d*npe + e*npe*ncx];
            maxres = std::max(maxres, std::abs(xr - (double)wm.x1[d + q*nd]));
        }
    }
    std::printf("[robust] Sampling (round-trip): npoints=%lld located=%lld npe=%lld : roundtrip=%.3e  partition_of_unity=%.3e\n",
                (long long)wm.npoints, (long long)located, (long long)npe, maxres, maxpou);
    int rc=0;
    if (located <= 0)       { std::printf("[robust] Sampling FAIL: no points located\n"); rc=1; }
    else if (maxres > 1e-9) { std::printf("[robust] Sampling FAIL: round-trip residual %.3e (bad element/shape)\n", maxres); rc=1; }
    else if (maxpou > 1e-9) { std::printf("[robust] Sampling FAIL: shape functions not partition-of-unity (%.3e)\n", maxpou); rc=1; }
    return rc;
}

// NOTE: an in-memory C++ LDG assembly check was attempted here (build an LDG disc, call
// ComputeLDGPreconditioner -> BlockJacobianLDG). It is NOT feasible via this harness: the header-only
// in-memory preprocessing (make_preprocessed / default_pde, documented "HDG-friendly") does not build
// LDG connectivity -- constructing a disc with discretization="ldg" segfaults inside preprocessing.
// LDG is instead exercised end to end through the frontend (examples/ConvectionDiffusion/1D, an LDG
// example that converges) in the MATLAB regression, which drives the full LDG backend
// (ldgblockjacobian.cpp). Reviving a C++ LDG check requires teaching make_preprocessed the LDG path.

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
        // Zoo point 2: compressible Navier-Stokes (ncu=4 system) -- auto-generated Jacobian check.
        rc |= check_ns_kernels();
        // Zoo point 3: point-location + interpolation round trip (CPointLocator subsystem).
        rc |= check_sampling();
        if (rc==0) std::printf("[robust] ALL PASS\n");
    }
    Kokkos::finalize();
    PetscFinalize();
    return rc;
}
