/*
    stagetables.hpp -- the replaceable-stage tables of the LDG residual (residualstages.hpp) and block-Jacobian
    preconditioner (precondstages.hpp), their stage contexts, and their per-discretization registration.

    Needs only common.h, so code that just consults a table (the preconditioner apply hook in preconditioner.cpp,
    the GMRES CGS hook in gmres.cpp) compiles without the discretization headers.
*/
#ifndef __STAGETABLES
#define __STAGETABLES

template <class T=dstype, class I=Int>
struct ResidualStageContextT {
    solstructT<T,I>& sol; resstructT<T,I>& res; appstructT<T,I>& app; masterstructT<T,I>& master;
    meshstructT<T,I>& mesh; tempstructT<T,I>& tmp; commonstructT<T,I>& common; cublasHandle_t handle; Int backend;
};
using ResidualStageContext = ResidualStageContextT<dstype, Int>;

// Runtime stage table (default precision only). Every entry is optional; nullptr selects the production stage.
struct ResidualStageTable {
    int version = 1;
    const char* name = "";
    void (*uhat)(ResidualStageContext&, int pass) = nullptr;
    void (*q)(ResidualStageContext&, Int nbe1, Int nbe2, int pass) = nullptr;
    void (*w)(ResidualStageContext&, Int nbe1, Int nbe2) = nullptr;
    void (*elem)(ResidualStageContext&, Int nbe1, Int nbe2) = nullptr;
    void (*face)(ResidualStageContext&) = nullptr;
    void (*tail)(ResidualStageContext&) = nullptr;
};

// Register (or, with nullptr, clear) the stage table this discretization's residual uses. The table must outlive its use.
template <class T, class I>
inline void SetResidualStages(commonstructT<T,I>& common, const ResidualStageTable* t) { common.residualStages = t; }
template <class T, class I>
inline const ResidualStageTable* ResidualStagesOf(const commonstructT<T,I>& common)
{
    if constexpr (std::is_same_v<T, dstype> && std::is_same_v<I, Int>) return static_cast<const ResidualStageTable*>(common.residualStages);
    else return nullptr;   // tables are default-precision only
}

struct PrecondStageContext;   // ResidualStageContext + the model driver ABI (precondstages.hpp)

struct PrecondStageTable {
    int version = 4;
    const char* name = "";
    void (*elem)(PrecondStageContext&, Int jth) = nullptr;
    void (*elemface)(PrecondStageContext&, Int jth) = nullptr;
    void (*schur)(ResidualStageContext&, Int jth) = nullptr;
    void (*inverse)(ResidualStageContext&, dstype* A, Int n, Int batch) = nullptr;
    void (*cross)(PrecondStageContext&, dstype* K) = nullptr;       // v3
    void (*apply)(ResidualStageContext&, dstype* x) = nullptr;      // v4
    void (*cgs)(cublasHandle_t handle, dstype* V, dstype* H, dstype* temp, Int N, Int m, Int backend) = nullptr;   // v4
};

// Register (or, with nullptr, clear) this discretization's preconditioner stage table. The table must outlive its use.
inline void SetPrecondStages(commonstruct& common, const PrecondStageTable* t) { common.precondStages = t; }
inline const PrecondStageTable* PrecondStagesOf(const commonstruct& common)
{ return static_cast<const PrecondStageTable*>(common.precondStages); }

#endif
