/*
    precondstages.hpp -- replaceable stages of the LDG block-Jacobian preconditioner build.

    BlockJacobianLDG (one rank) and mpiBlockJacobianLDG (several ranks) keep their orchestration; two of their
    per-block stages go through a registered PrecondStageTable when one is set:

        elem(pctx, jth)              uEquationElemBlock for element block jth: res.D, res.B <- -(volume terms)
        elemface(pctx, jth)          uEquationElemFaceBlockLDG: res.D, res.B += face terms; res.F <- face terms
        schur(ctx, jth)              uEquationSchurBlockLDG for element block jth: res.D <- the block's Schur complement
        cross(pctx, K)               RuFaceCrossDerivOptimized: K += the cross-face q-derivative terms (before the inverse)
        apply(ctx, x)                CPreconditioner::ApplyPreconditioner (LDG block-Jacobian): x <- K x, per element block
        cgs(handle, V, H, ...)       GMRES classical Gram-Schmidt step (gmres.cpp CGS): orthogonalize V[m] against V[0..m)
        inverse(ctx, A, n, batch)    Inverse: A (batch column-major n x n blocks) <- their inverses; res.H is scratch

    elem/elemface receive a PrecondStageContext (the residual context plus the model driver ABI, which the face stage's
    boundary Jacobians need).

    Every entry is optional (nullptr selects the production stage), and without a registered table the build is the
    unchanged production code. Stage tables are case-side code; they receive the discretization structs through the
    same ResidualStageContext as the residual stages (residualstages.hpp).
*/
#ifndef __PRECONDSTAGES
#define __PRECONDSTAGES

#include "stagetables.hpp"

struct PrecondStageContext : ResidualStageContext {
    ExasimDriverABI& driver_abi;
};

inline void PrecondElemStage(solstruct &sol, resstruct &res, appstruct &app, ExasimDriverABI& driver_abi,
        masterstruct &master, meshstruct &mesh, tempstruct &tmp, commonstruct &common, cublasHandle_t handle,
        Int jth, Int backend)
{
    const PrecondStageTable* t = PrecondStagesOf(common);
    if (t != nullptr && t->elem != nullptr) {
        PrecondStageContext c{{sol, res, app, master, mesh, tmp, common, handle, backend}, driver_abi};
        t->elem(c, jth);
    }
    else uEquationElemBlock<exasim::detail::AbiAdapter>(sol, res, app, master, mesh, tmp, common, handle, jth, backend);
}

inline void PrecondElemFaceStage(solstruct &sol, resstruct &res, appstruct &app, ExasimDriverABI& driver_abi,
        masterstruct &master, meshstruct &mesh, tempstruct &tmp, commonstruct &common, cublasHandle_t handle,
        Int jth, Int backend)
{
    const PrecondStageTable* t = PrecondStagesOf(common);
    if (t != nullptr && t->elemface != nullptr) {
        PrecondStageContext c{{sol, res, app, master, mesh, tmp, common, handle, backend}, driver_abi};
        t->elemface(c, jth);
    }
    else uEquationElemFaceBlockLDG(sol, res, app, driver_abi, master, mesh, tmp, common, handle, jth, backend);
}

inline void PrecondSchurStage(solstruct &sol, resstruct &res, appstruct &app, ExasimDriverABI& driver_abi,
        masterstruct &master, meshstruct &mesh, tempstruct &tmp, commonstruct &common, cublasHandle_t handle,
        Int jth, Int backend, LDGSchurBenchmarkTimes* benchmark)
{
    const PrecondStageTable* t = PrecondStagesOf(common);
    if (t != nullptr && t->schur != nullptr) {
        ResidualStageContext c{sol, res, app, master, mesh, tmp, common, handle, backend};
        t->schur(c, jth);
    }
    else uEquationSchurBlockLDG(sol, res, app, driver_abi, master, mesh, tmp, common, handle, jth, backend, benchmark);
}

inline void PrecondCrossStage(dstype* K, solstruct &sol, resstruct &res, appstruct &app, ExasimDriverABI& driver_abi,
        masterstruct &master, meshstruct &mesh, tempstruct &tmp, commonstruct &common)
{
    const PrecondStageTable* t = PrecondStagesOf(common);
    if (t != nullptr && t->cross != nullptr) {
        PrecondStageContext c{{sol, res, app, master, mesh, tmp, common, common.cublasHandle, common.backend}, driver_abi};
        t->cross(c, K);
    }
    else RuFaceCrossDerivOptimized(K, sol, res, app, driver_abi, master, mesh, tmp, common);
}

inline void PrecondInverseStage(solstruct &sol, resstruct &res, appstruct &app, masterstruct &master, meshstruct &mesh,
        tempstruct &tmp, commonstruct &common, cublasHandle_t handle, dstype* A, Int n, Int batch, Int backend)
{
    const PrecondStageTable* t = PrecondStagesOf(common);
    if (t != nullptr && t->inverse != nullptr) {
        ResidualStageContext c{sol, res, app, master, mesh, tmp, common, handle, backend};
        t->inverse(c, A, n, batch);
    }
    else if(common.hdgldgBoundary>0) {
        // Only the coupled path needs the no-allocation workspace variant.
        // H is the native LDG inverse-output work, distinct from the K prefix.
        if(!res.fhAliasesK || !res.K || !res.H || res.H<res.K ||
           std::size_t(res.H-res.K)>res.kArenaCursor || res.kArenaCursor>res.szK)
            error("HDG/LDG inverse has invalid native assembly work");
        InverseWithWorkspace(handle,A,res.H,res.ipiv,n,batch,backend,
            tmp.tempg,std::size_t(tmp.sztempg)*sizeof(dstype),
            res.kArenaCursor-std::size_t(res.H-res.K),res.szipiv);
    }
    else Inverse(handle, A, res.H, res.ipiv, n, batch, backend);
}

#endif
