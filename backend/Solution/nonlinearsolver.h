/**
 * @class CNonlinearSolver
 * @brief Drives the nonlinear solve -- the pseudo-transient-continuation (PTCsolver) and Newton
 *        (NewtonSolver) iterations that sit above the linear solve.
 *
 * CSolution split (nonlinear half): these loops are the nonlinear solver, distinct from the time-
 * integration orchestration (SolveProblem / DIRK / SteadyProblem) that remains in CSolution and
 * from the output I/O that moved to CSolutionWriter. Extracting the writer first (Sep12) removed
 * the crash-dump's direct stream poking, which is what previously tied these loops to the streams.
 *
 * The solver holds references to the pieces it drives: the discretization, the residual R(u), the
 * HDG assembler, the preconditioner, the linear solver (CSolver), and the writer (for crash dumps).
 * Method bodies are unchanged from CSolution -- the referenced members keep the same names.
 */
#ifndef __NONLINEARSOLVER_H__
#define __NONLINEARSOLVER_H__

#include <exasim/detail/abi_adapter.hpp>

template <class, class> class CDiscretizationT; using CDiscretization = CDiscretizationT<::dstype, ::Int>;   // forward declarations (CNonlinearSolver only holds references)
template <class M> class CResidual;
template <class M> class CAssembler;
template <class M> class CPreconditioner;
template <class M> class CSolver;
template <class M> class CSolutionWriter;

// Templated on the user Model type M (default = AbiAdapter): holds the model-dependent pieces it
// drives by reference; M threads through to its one model call (GetW<M>) and to the typed members.
template <class M = exasim::detail::AbiAdapter>
class CNonlinearSolver {
public:
    CDiscretization& disc;
    CResidual<M>& residual;
    CAssembler<M>& assembler;
    CPreconditioner<M>& prec;
    CSolver<M>& solv;
    CSolutionWriter<M>& writer;

    CNonlinearSolver(CDiscretization& disc_, CResidual<M>& residual_, CAssembler<M>& assembler_,
                     CPreconditioner<M>& prec_, CSolver<M>& solv_, CSolutionWriter<M>& writer_)
        : disc(disc_), residual(residual_), assembler(assembler_),
          prec(prec_), solv(solv_), writer(writer_) {}

    // pseudo-transient continuation (LDG) and Newton (HDG) nonlinear iterations
    Int PTCsolver(ofstream &out, Int backend);
    Int NewtonSolver(ofstream &out, Int N, Int spatialScheme, Int backend);
};

#endif
