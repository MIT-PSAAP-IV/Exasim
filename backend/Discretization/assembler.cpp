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
      MatVec<M>(Jv, sol, res, app, master, mesh, tmp, common, common.cublasHandle, v, u, Ru, backend);
    }
    else if (spatialScheme == 1) { // HDG
      hdgMatVec(Jv, res.H, v, res.Rh, res.Rq, res, app, mesh, common, tmp, common.cublasHandle, backend);
    }
}

#endif
