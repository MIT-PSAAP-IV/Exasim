/**
 * @class CPreconditioner
 * @brief Manages the construction and application of preconditioners for linear systems.
 *
 * This class encapsulates the logic for creating and applying preconditioners, which are used to
 * improve the convergence of iterative solvers for linear systems. It interacts with discretization
 * and system structures, and supports different computational backends and spatial schemes.
 *
 * Members:
 * - precond: Stores the preconditioner structure.
 * - mpiRank: The MPI rank of the current process.
 *
 * Methods:
 * - CPreconditioner(CDiscretization& disc, Int backend): Constructor that initializes the preconditioner with the given discretization and backend.
 * - ~CPreconditioner(): Destructor.
 * - ComputeInitialGuessAndPreconditioner(sysstruct& sys, solverstatestruct& state, CAssembler& assembler, CDiscretization& disc, Int backend): Computes the initial guess and constructs the preconditioner.
 * - ComputeInitialGuessAndPreconditioner(sysstruct& sys, solverstatestruct& state, CAssembler& assembler, CDiscretization& disc, Int N, Int spatialScheme, Int backend): Overloaded method to compute initial guess and preconditioner with additional parameters.
 * - ApplyPreconditioner(dstype* v, sysstruct& sys, CDiscretization& disc, Int backend): Applies the preconditioner to a vector.
 * - ApplyPreconditioner(dstype* v, sysstruct& sys, CDiscretization& disc, Int spatialScheme, Int backend): Overloaded method to apply the preconditioner with a specific spatial scheme.
 */
#ifndef __PRECONDITIONER_H__
#define __PRECONDITIONER_H__

#include "exasim/execution_mode.hpp"
#include <exasim/detail/abi_adapter.hpp>

template <class M> class CAssembler;  // forward decl (preconditioner only references it by ref)

// Templated on the user Model type M (default = AbiAdapter): needs M only to name CAssembler<M>
// in its method signatures; its own driver calls (BlockJacobianLDG) stay direct for now.
template <class M = exasim::detail::AbiAdapter>
class CPreconditioner {
private:
public:
    precondstruct precond; // store precondioner struct
        
    int mpiRank;
    
    // constructor 
    CPreconditioner(CDiscretization& disc, Int backend,
                    ExasimExecutionMode mode = ExasimExecutionMode::Solve); 
    
    // destructor        
    ~CPreconditioner(); 
            
    void ComputeInitialGuessAndPreconditioner(sysstruct& sys, solverstatestruct& state, CAssembler<M>& assembler, CDiscretization& disc, Int backend);

    void ComputeInitialGuessAndPreconditioner(sysstruct& sys, solverstatestruct& state, CAssembler<M>& assembler, CDiscretization& disc, Int N, Int spatialScheme, Int backend);
    
    // apply the precontioner: Pv = P(u)*v
    void ApplyPreconditioner(dstype* v, sysstruct& sys, CDiscretization& disc, Int backend);
    void ApplyPreconditioner(dstype* v, sysstruct& sys, CDiscretization& disc, Int spatialScheme, Int backend);

    // compute the LDG block-Jacobi preconditioner matrix K from the discretization's state
    // (re-homed from CDiscretization in C4 -- computing the preconditioner is a preconditioner concern)
    void ComputeLDGPreconditioner(CDiscretization& disc, dstype* K, dstype* u, Int backend);

    // compute the HDG preconditioner matrix K (block-Jacobi / additive-Schwarz / block-ILU0)
    // from the assembled global system H. Re-homed from CAssembler::hdgAssembleLinearSystem so
    // assembly (building H/Rh) and preconditioning (building K from H) live in separate classes.
    void ComputeHDGPreconditioner(CDiscretization& disc, Int backend);
};

#endif        
