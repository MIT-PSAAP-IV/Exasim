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

    if constexpr (std::is_same_v<T,double> && std::is_same_v<I,::Int>) {
      if (common.composite) fieldsplit::freeze(*common.composite,sol,app,common);
    }
    const size_t hs=common.composite?fieldsplit::HS:size_t(m)*m;
    const size_t fs=common.composite?fieldsplit::FS:size_t(n)*m;
    fieldsplit::setStorage(res.H,T(0),hs*ne);
    fieldsplit::setStorage(res.F,T(0),fs*ne);
    ArraySetValue(res.Rh, zero, m*ne);
    ArraySetValue(res.Ru, zero, n*ne);

#ifdef HAVE_MPI
    hdgAssembleLinearSystemMPI<M>(b, sol, res, app, master, mesh, tmp, common,
                                  disc.getNonmatchingInteractions(), common.cublasHandle, backend); // [nonmatching]
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
    hdgAssembleResidualMPI<M>(b, sol, res, app, master, mesh, tmp, common,
                              disc.getNonmatchingInteractions(), common.cublasHandle, backend); // [nonmatching]
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
    evalMatVecWithNorm(Jv, v, u, Ru, dstype(-1), backend);
}

template <class M, class T, class I>
void CAssembler<M, T, I>::evalMatVecWithNorm(dstype* Jv, dstype* v, dstype* u, dstype* Ru, dstype directionNorm, Int backend)
{
    auto& sol = disc.sol; auto& res = disc.res; auto& app = disc.app;
    auto& master = disc.master; auto& mesh = disc.mesh; auto& tmp = disc.tmp;
    auto& common = disc.common;
    MatVec<M>(Jv, sol, res, app, master, mesh, tmp, common, common.cublasHandle, v, u, Ru, backend, directionNorm);
}

// matrix-vector product Jv = J(u)*v (LDG matrix-free FD, or HDG apply of the assembled res.H)
template <class M, class T, class I>
void CAssembler<M, T, I>::evalMatVec(dstype* Jv, dstype* v, dstype* u, dstype* Ru, Int spatialScheme, Int backend)
{
    evalMatVecWithNorm(Jv, v, u, Ru, dstype(-1), spatialScheme, backend);
}

template <class M, class T, class I>
void CAssembler<M, T, I>::evalMatVecWithNorm(dstype* Jv, dstype* v, dstype* u, dstype* Ru, dstype directionNorm, Int spatialScheme, Int backend)
{
    auto& sol = disc.sol; auto& res = disc.res; auto& app = disc.app;
    auto& master = disc.master; auto& mesh = disc.mesh; auto& tmp = disc.tmp;
    auto& common = disc.common;
    if (spatialScheme == 0) {// LDG
      MatVec<M>(Jv, sol, res, app, master, mesh, tmp, common, common.cublasHandle, v, u, Ru, backend, directionNorm);
    }
    else if (spatialScheme == 1) { // HDG
      hdgMatVec(Jv, res.H, v, res.Rh, res.Rq, res, app, mesh, common, tmp, common.cublasHandle, backend,
                disc.getNonmatchingInteractions()); // [nonmatching]
    }
}

#endif
