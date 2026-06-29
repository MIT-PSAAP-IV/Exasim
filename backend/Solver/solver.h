/**
 * @class CSolver
 * @brief Solver class for handling linear and nonlinear systems in the Exasim backend.
 *
 * This class encapsulates the solver logic, including pseudo-transient continuation (PTC)
 * and Newton-based methods for solving systems arising from discretization.
 *
 * Members:
 * - sys:   System structure containing problem-specific data (Krylov vectors, history).
 * - state: Mutable solver/reduced-basis runtime state (Stage 1; was in commonstruct).
 * - mpiRank: MPI rank for parallel computations.
 *
 * Constructors & Destructors:
 * - CSolver(CDiscretization& disc, Int backend): Constructs a solver with given discretization and backend.
 * - ~CSolver(): Destructor for cleanup.
 *
 * Methods (Stage 2: the linear-algebra solve surface is now owned by CSolver, operating on
 * its own sys/state; previously these were free functions taking sys/state by reference):
 * - linearSolve(...): one (LDG or HDG) linear solve = preconditioner setup + GMRES.
 * - gmres(...):       restarted GMRES with polynomial preconditioning (private machinery).
 * - updateRB(...):    update the reduced-basis space after a nonlinear step.
 * The nonlinear drivers (PTC / Newton loops) live in CSolution, which owns the residual/
 * output/visualization context they need; they call these CSolver methods.
 */
#ifndef __SOLVER_H__
#define __SOLVER_H__

#include "exasim/execution_mode.hpp"
#include <exasim/detail/abi_adapter.hpp>

template <class M> class CAssembler;       // HDG global-system assembler (Stage 3); HDG linearSolve assembles via it
template <class M> class CResidual;        // the discretized PDE residual R(u) (full-split); HDG/LDG linearSolve evaluates R via it
template <class M> class CPreconditioner;  // referenced by ref in linearSolve/gmres/updateRB

// Templated on the user Model type M (default = AbiAdapter): needs M only to name
// CResidual<M>/CAssembler<M>/CPreconditioner<M> in its solve-surface signatures.
template <class M = exasim::detail::AbiAdapter>
class CSolver {
private:
public:
    sysstruct sys; // system struct
    solverstatestruct state; // mutable solver/reduced-basis runtime state (lifted out of
                             // commonstruct in Stage 1; default-initialized on construction)

    int mpiRank;
    
    // constructor
    CSolver(CDiscretization& disc, Int backend,
            ExasimExecutionMode mode = ExasimExecutionMode::Solve);

    // destructor
    ~CSolver();

    // one linear solve (preconditioner setup + GMRES); LDG returns the GMRES iteration count.
    int  linearSolve(CResidual<M>& residual, CAssembler<M>& assembler, CDiscretization& disc, CPreconditioner<M>& prec, ofstream &out, Int it, Int backend);
    void linearSolve(CResidual<M>& residual, CAssembler<M>& assembler, CDiscretization& disc, CPreconditioner<M>& prec, ofstream &out, Int N, Int spatialScheme, Int it, Int backend);

    // restarted GMRES (operates on this->sys / this->state)
    Int  gmres(CAssembler<M>& assembler, CDiscretization &disc, CPreconditioner<M>& prec, Int backend);
    Int  gmres(CAssembler<M>& assembler, CDiscretization &disc, CPreconditioner<M>& prec, Int N, Int spatialScheme, Int backend);

    // update the reduced-basis space after a nonlinear step
    void updateRB(CDiscretization& disc, CPreconditioner<M>& prec, Int backend);
    void updateRB(CDiscretization& disc, CPreconditioner<M>& prec, Int N, Int backend);
};

#endif        
