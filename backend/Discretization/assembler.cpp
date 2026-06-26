/*
    CAssembler -- HDG global linear-system assembly, extracted from CDiscretization (Stage 3).
    The method bodies are unchanged from the former CDiscretization::hdgAssemble* methods;
    the struct members they used (sol/res/app/master/mesh/tmp/common) are bound here as
    references into the discretization, so the body reads exactly as before.
*/
#ifndef __ASSEMBLER
#define __ASSEMBLER

#include "assembler.h"

void CAssembler::hdgAssembleLinearSystem(dstype *b, Int backend)
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
    hdgAssembleLinearSystemMPI<exasim::detail::AbiAdapter>(b, sol, res, app, master, mesh, tmp, common, common.cublasHandle, backend);
#else
    uEquationHDG<exasim::detail::AbiAdapter>(sol, res, app, master, mesh, tmp, common, common.cublasHandle, backend);
    hdgAssembleRHS<exasim::detail::AbiAdapter>(b, res.Rh, mesh, common);
#endif

    if (common.solverparams.preconditioner==0) {
      // fix bug here: tmp.tempn is not enough memory to store ncu*npf*ncu*npf*nf
      hdgBlockJacobi<exasim::detail::AbiAdapter>(res.K, res.H, res, mesh, tmp, common, common.cublasHandle, backend);
    }
    else if (common.solverparams.preconditioner==1) {
      hdgElementalAdditiveSchwarz<exasim::detail::AbiAdapter>(res.K, res.H, res, mesh, tmp, common, common.cublasHandle, backend);
    }
    else if (common.solverparams.preconditioner==2) {
      hdgBlockILU0<exasim::detail::AbiAdapter>(res.K, res.H, res, mesh, tmp, common, common.cublasHandle, backend);
    }
}

void CAssembler::hdgAssembleResidual(dstype *b, Int backend)
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
    hdgAssembleResidualMPI<exasim::detail::AbiAdapter>(b, sol, res, app, master, mesh, tmp, common, common.cublasHandle, backend);
#else
    // b, K, H, F, Ru
    ResidualHDG<exasim::detail::AbiAdapter>(sol, res, app, master, mesh, tmp, common, common.cublasHandle, backend);
    //uEquationHDG<exasim::detail::AbiAdapter>(sol, res, app, master, mesh, tmp, common, common.cublasHandle, backend);
    hdgAssembleRHS<exasim::detail::AbiAdapter>(b, res.Rh, mesh, common);
#endif
}

// matrix-vector product Jv = J(u)*v
void CAssembler::evalMatVec(dstype* Jv, dstype* v, dstype* u, dstype* Ru, Int backend)
{
    auto& sol = disc.sol; auto& res = disc.res; auto& app = disc.app;
    auto& master = disc.master; auto& mesh = disc.mesh; auto& tmp = disc.tmp;
    auto& common = disc.common;
    MatVec<exasim::detail::AbiAdapter>(Jv, sol, res, app, master, mesh, tmp, common, common.cublasHandle, v, u, Ru, backend);
}

// matrix-vector product Jv = J(u)*v (LDG matrix-free FD, or HDG apply of the assembled res.H)
void CAssembler::evalMatVec(dstype* Jv, dstype* v, dstype* u, dstype* Ru, Int spatialScheme, Int backend)
{
    auto& sol = disc.sol; auto& res = disc.res; auto& app = disc.app;
    auto& master = disc.master; auto& mesh = disc.mesh; auto& tmp = disc.tmp;
    auto& common = disc.common;
    if (spatialScheme == 0) {// LDG
      MatVec<exasim::detail::AbiAdapter>(Jv, sol, res, app, master, mesh, tmp, common, common.cublasHandle, v, u, Ru, backend);
    }
    else if (spatialScheme == 1) { // HDG
      hdgMatVec<exasim::detail::AbiAdapter>(Jv, res.H, v, res.Rh, res.Rq, res, app, mesh, common, tmp, common.cublasHandle, backend);
    }
}

#endif
