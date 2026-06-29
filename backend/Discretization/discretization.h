/**
 * @class CDiscretization
 * @brief Handles the discretization process for numerical simulations, supporting both CPU and GPU backends.
 *
 * This class encapsulates data structures and methods required for discretizing PDEs, assembling linear systems,
 * evaluating residuals, fluxes, artificial viscosity fields, and performing conversions between DG and CG representations.
 *
 * Members:
 * - sol: Solution structure containing state variables.
 * - res: Residual structure for storing residuals.
 * - app: Application-specific parameters and data.
 * - master: Master element data for discretization.
 * - mesh: Mesh structure containing grid information.
 * - tmp: Temporary storage for intermediate computations.
 * - common: Common parameters shared across computations.
 *
 * Methods:
 * - CDiscretization(...): Constructor initializing the discretization with input/output files and parallelization parameters.
 * - ~CDiscretization(): Destructor for cleanup.
 * - compGeometry(...): Computes geometry-related quantities.
 * - compMassInverse(...): Computes the inverse of the mass matrix.
 * - hdgAssembleLinearSystem(...): Assembles the linear system for HDG methods.
 * - hdgAssembleResidual(...): Assembles the residual vector for HDG methods.
 * - evalResidual(...): Evaluates the residual vector.
 * - evalResidual(...): Evaluates the residual vector at a given solution.
 * - evalQ(...): Evaluates the flux and stores it in the solution structure.
 * - evalQSer(...): Serial version of flux evaluation.
 * - evalQ(...): Evaluates the flux at a given solution.
 * - evalMatVec(...): Evaluates the matrix-vector product Jv = J(u)*v.
 * - evalMatVec(...): Overloaded version with spatial scheme selection.
 * - updateUDG(...): Updates the solution structure with new values and computes flux.
 * - updateU(...): Updates the solution structure with new values.
 * - evalAVfield(...): Evaluates the artificial viscosity field at a given solution.
 * - evalAVfield(...): Evaluates the artificial viscosity field at the current solution.
 * - evalOutput(...): Evaluates output quantities at the current solution.
 * - evalMonitor(...): Evaluates a monitor function for tracking solution changes during pseudotime stepping.
 * - DG2CG(...): Converts DG representation to CG.
 * - DG2CG2(...): Alternative DG to CG conversion.
 * - DG2CG3(...): Another variant of DG to CG conversion.
 */
#ifndef __DISCRETIZATION_H__
#define __DISCRETIZATION_H__

#include "exasim/execution_mode.hpp"

class CDiscretization {
private:
public:
    solstruct sol;
    resstruct res;
    appstruct app;
    wallmodelstruct wallmodel;
    masterstruct master;
    meshstruct mesh;
    tempstruct tmp;
    commonstruct common;
    ExasimDriverABI driver_abi;
    scratcharenastruct scratch;  // owns the K backing buffer; res.K/views + sys.v are non-owning reserves (S5 step 3)
    // solstruct hsol;

    // constructor for both CPU and GPU
    CDiscretization(std::string filein, std::string fileout, std::string exasimpath, Int mpiprocs, 
                    Int mpirank, Int ompthreads, Int omprank, Int backend,
                    Int builtinmodelID, const ExasimDriverABI& abi,
                    Int nsca = 0, Int nvec = 0, Int nten = 0,
                    Int nsurf = 0, Int nvqoi = 0,
                    ExasimExecutionMode mode = ExasimExecutionMode::Solve,
                    const std::vector<dstype>* physicsparamOverride = nullptr,
                    Int saveParaview = 0);

    // No-ABI constructor (C3): the concrete-model build (CSolution<M>, M != AbiAdapter) has no
    // runtime ExasimDriverABI -- every model call is inlined through the templated exasim::Name<M>
    // kernels. Delegate to the ABI constructor with a default (all-null) ABI; the fn-pointers are
    // only dereferenced by the discarded AbiAdapter branch of EXASIM_DRIVER_CALL, never for concrete M.
    // The temporary lives through the delegated constructor (which copies it into the driver_abi member).
    CDiscretization(std::string filein, std::string fileout, std::string exasimpath, Int mpiprocs,
                    Int mpirank, Int ompthreads, Int omprank, Int backend, Int builtinmodelID)
        : CDiscretization(filein, fileout, exasimpath, mpiprocs, mpirank, ompthreads, omprank,
                          backend, builtinmodelID, ExasimDriverABI{}) {}

    // destructor
    ~CDiscretization();
        
    // compute the geometry
    void compGeometry(Int backend);    
    
    // compute the mass inverse
    void compMassInverse(Int backend);    

    // (ComputeLDGPreconditioner moved to CPreconditioner in C4)
    // (hdgAssembleLinearSystem / hdgAssembleResidual / evalMatVec moved to CAssembler)
    // (evalResidual / evalQ / evalQSer / evalAVfield / updateUDG / updateU moved to CResidual)

    // (BuildWallModelData moved to CWallModel::build)
    
    // converge DG to CG (a basis transform on the function space; used by AV smoothing + output)
    void DG2CG(dstype* ucg, dstype* udg, dstype *utm, Int ncucg, Int ncudg, Int ncu, Int backend);
    void DG2CG2(dstype* ucg, dstype* udg, dstype *utm, Int ncucg, Int ncudg, Int ncu, Int backend);
    void DG2CG3(dstype* ucg, dstype* udg, dstype *utm, Int ncucg, Int ncudg, Int ncu, Int backend);

    // (interface/boundary sampling methods moved to CInterfaceSampler)
};

#endif        
