/*
    CAssembler -- HDG global linear-system assembly, extracted from CDiscretization (Stage 3).
    The method bodies are unchanged from the former CDiscretization::hdgAssemble* methods;
    the struct members they used (sol/res/app/master/mesh/tmp/common) are bound here as
    references into the discretization, so the body reads exactly as before.
*/
#ifndef __ASSEMBLER
#define __ASSEMBLER

#include "assembler.h"

template <class M, class T, class I>
void CAssembler<M, T, I>::hdgAssembleLinearSystem(dstype *b, Int backend)
{
    auto& sol = disc.sol; auto& res = disc.res; auto& app = disc.app;
    auto& master = disc.master; auto& mesh = disc.mesh; auto& tmp = disc.tmp;
    auto& common = disc.common;

    int n = common.grid.npe*common.components.ncu;
    int m = common.grid.npf*common.meshsizes.nfe*common.components.ncu;
    int ne = common.meshsizes.ne1;

    ArraySetValue(res.H, zero, m*m*ne);
    ArraySetValue(res.Rh, zero, m*ne);
    ArraySetValue(res.Ru, zero, n*ne);
    ArraySetValue(res.F, zero, n*m*ne);

#ifdef HAVE_MPI
    hdgAssembleLinearSystemMPI<M>(b, sol, res, app, master, mesh, tmp, common, common.cublasHandle, backend);
#else
    uEquationHDG<M>(sol, res, app, master, mesh, tmp, common, common.cublasHandle, backend);
    hdgAssembleRHS(b, res.Rh, mesh, common);
#endif
    // (building the HDG preconditioner matrix K from the assembled H is a preconditioner
    //  concern -- moved to CPreconditioner::ComputeHDGPreconditioner, called after assembly)
}

template <class M, class T, class I>
void CAssembler<M, T, I>::hdgAssembleResidual(dstype *b, Int backend)
{
    auto& sol = disc.sol; auto& res = disc.res; auto& app = disc.app;
    auto& master = disc.master; auto& mesh = disc.mesh; auto& tmp = disc.tmp;
    auto& common = disc.common;

    int n = common.grid.npe*common.components.ncu;
    int m = common.grid.npf*common.meshsizes.nfe*common.components.ncu;
    int ne = common.meshsizes.ne1;
    ArraySetValue(res.Rh, zero, m*ne);
    ArraySetValue(res.Ru, zero, n*ne);

#ifdef HAVE_MPI
    hdgAssembleResidualMPI<M>(b, sol, res, app, master, mesh, tmp, common, common.cublasHandle, backend);
#else
    // b, K, H, F, Ru
    ResidualHDG<M>(sol, res, app, master, mesh, tmp, common, common.cublasHandle, backend);
    //uEquationHDG<M>(sol, res, app, master, mesh, tmp, common, common.cublasHandle, backend);
    hdgAssembleRHS(b, res.Rh, mesh, common);
#endif
}

// Assemble the LDG block operator at base state u (M1): run the per-element block-Jacobian
// assembly and capture the UN-inverted element diagonal into res.Adiag (ldgMatVec applies it).
// Scope for M1: serial CPU/GPU single-rank, and AV must be frozen -- the analytic LDG Jacobian
// does not differentiate artificial viscosity, so with live AV the assembled operator would not
// match the FD reference. The off-diagonal neighbor blocks are M2; the MPI halo is M3.
template <class M, class T, class I>
void CAssembler<M, T, I>::ldgAssembleLinearSystem(dstype* u, Int backend)
{
    auto& res = disc.res; auto& common = disc.common;

    if (common.mpiProcs > 1)
        error("ldgAssembleLinearSystem: assembled LDG operator is serial-only in M1 (MPI is M3)");
    if (common.solverparams.preconditioner != 1)
        error("ldgAssembleLinearSystem: needs the LDG block-Jacobian arena (run with preconditioner==1)");
    if (res.Adiag == nullptr)
        error("ldgAssembleLinearSystem: res.Adiag is not allocated");
    if (common.physicsparams.ncAV > 0 && common.physicsparams.frozenAVflag == 0)
        error("ldgAssembleLinearSystem: M1 requires frozenAVflag==1 when ncAV>0 (AV differentiation absent)");

    // Reuse the existing per-element assembly; res.K receives the inverted block-Jacobi
    // preconditioner (a harmless side effect) while res.Adiag receives the un-inverted diagonal
    // and res.Aoff the off-diagonal neighbour blocks (res.Anbr the neighbour-element map).
    BlockJacobianLDG(res.K, u, disc.sol, res, disc.app, disc.driver_abi, disc.master, disc.mesh,
            disc.tmp, common, common.cublasHandle, backend, res.Adiag, res.Aoff, res.Anbr);
}

// matrix-vector product Jv = J(u)*v
template <class M, class T, class I>
void CAssembler<M, T, I>::evalMatVec(dstype* Jv, dstype* v, dstype* u, dstype* Ru, Int backend)
{
    auto& sol = disc.sol; auto& res = disc.res; auto& app = disc.app;
    auto& master = disc.master; auto& mesh = disc.mesh; auto& tmp = disc.tmp;
    auto& common = disc.common;
    MatVec<M>(Jv, sol, res, app, master, mesh, tmp, common, common.cublasHandle, v, u, Ru, backend);
}

// matrix-vector product Jv = J(u)*v (LDG matrix-free FD, or HDG apply of the assembled res.H)
template <class M, class T, class I>
void CAssembler<M, T, I>::evalMatVec(dstype* Jv, dstype* v, dstype* u, dstype* Ru, Int spatialScheme, Int backend)
{
    auto& sol = disc.sol; auto& res = disc.res; auto& app = disc.app;
    auto& master = disc.master; auto& mesh = disc.mesh; auto& tmp = disc.tmp;
    auto& common = disc.common;
    if (spatialScheme == 0) {// LDG
      if (common.solverparams.ldgAssembledOperator) // apply the pre-assembled block operator (diagonal + off-diagonal)
        ldgMatVec(Jv, res.Adiag, res.Aoff, res.Anbr, res.Avnbr, v, common, common.cublasHandle, backend);
      else // matrix-free finite-difference matvec (default fallback + validation reference)
        MatVec<M>(Jv, sol, res, app, master, mesh, tmp, common, common.cublasHandle, v, u, Ru, backend);
    }
    else if (spatialScheme == 1) { // HDG
      hdgMatVec(Jv, res.H, v, res.Rh, res.Rq, res, app, mesh, common, tmp, common.cublasHandle, backend);
    }
}

#endif
