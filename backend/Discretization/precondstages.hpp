/*
    precondstages.hpp -- replaceable stages of the LDG block-Jacobian preconditioner build.

    BlockJacobianLDG (one rank) and mpiBlockJacobianLDG (several ranks) keep their orchestration; two of their
    per-block stages go through a registered PrecondStageTable when one is set:

        schur(ctx, jth)              uEquationSchurBlockLDG for element block jth: res.D <- the block's Schur complement
        inverse(ctx, A, n, batch)    Inverse: A (batch column-major n x n blocks) <- their inverses; res.H is scratch

    Every entry is optional (nullptr selects the production stage), and without a registered table the build is the
    unchanged production code. Stage tables are case-side code; they receive the discretization structs through the
    same ResidualStageContext as the residual stages (residualstages.hpp).
*/
#ifndef __PRECONDSTAGES
#define __PRECONDSTAGES

struct PrecondStageTable {
    int version = 1;
    const char* name = "";
    void (*schur)(ResidualStageContext&, Int jth) = nullptr;
    void (*inverse)(ResidualStageContext&, dstype* A, Int n, Int batch) = nullptr;
};

inline const PrecondStageTable*& PrecondStageRegistry() { static const PrecondStageTable* t = nullptr; return t; }
// Register (or, with nullptr, clear) the preconditioner stage table. The table must outlive its use.
inline void SetPrecondStages(const PrecondStageTable* t) { PrecondStageRegistry() = t; }

inline void PrecondSchurStage(solstruct &sol, resstruct &res, appstruct &app, ExasimDriverABI& driver_abi,
        masterstruct &master, meshstruct &mesh, tempstruct &tmp, commonstruct &common, cublasHandle_t handle,
        Int jth, Int backend, LDGSchurBenchmarkTimes* benchmark)
{
    const PrecondStageTable* t = PrecondStageRegistry();
    if (t != nullptr && t->schur != nullptr) {
        ResidualStageContext c{sol, res, app, master, mesh, tmp, common, handle, backend};
        t->schur(c, jth);
    }
    else uEquationSchurBlockLDG(sol, res, app, driver_abi, master, mesh, tmp, common, handle, jth, backend, benchmark);
}

inline void PrecondInverseStage(solstruct &sol, resstruct &res, appstruct &app, masterstruct &master, meshstruct &mesh,
        tempstruct &tmp, commonstruct &common, cublasHandle_t handle, dstype* A, Int n, Int batch, Int backend)
{
    const PrecondStageTable* t = PrecondStageRegistry();
    if (t != nullptr && t->inverse != nullptr) {
        ResidualStageContext c{sol, res, app, master, mesh, tmp, common, handle, backend};
        t->inverse(c, A, n, batch);
    }
    else Inverse(handle, A, res.H, res.ipiv, n, batch, backend);
}

#endif
