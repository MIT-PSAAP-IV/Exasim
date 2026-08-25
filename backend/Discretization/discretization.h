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
 * - CDiscretizationT(...): Constructor initializing the discretization with input/output files and parallelization parameters.
 * - ~CDiscretizationT(): Destructor for cleanup.
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

namespace exasim { template <class, class> struct PreprocessedT; }  // fwd decl: in-memory ctor input
                                           // (buildstructs.hpp); the ctor bodies live in discretization_inmemory.hpp (consumer-only)

// Templated on scalar precision T and index type I (Phase 2 of dstype->template threading, see
// docs/internals/precision-threading.md). Member `using` aliases shadow the global dstype/Int AND the
// Phase-1 struct aliases, so every member declaration and every out-of-line method body below is
// UNCHANGED yet now resolves to the <T,I> struct instantiations; with the default args the type is
// byte-identical to the pre-Phase-2 concrete class.
template <class T = ::dstype, class I = ::Int>
class CDiscretizationT {
private:
    using dstype             = T;
    using Int                = I;
    using solstruct          = solstructT<T, I>;
    using resstruct          = resstructT<T, I>;
    using appstruct          = appstructT<T, I>;
    using masterstruct       = masterstructT<T, I>;
    using meshstruct         = meshstructT<T, I>;
    using tempstruct         = tempstructT<T, I>;
    using commonstruct       = commonstructT<T, I>;
    using scratcharenastruct = scratcharenastructT<T, I>;
    // wallmodelstruct is not templated (Phase 1) -> stays the global type
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
    CDiscretizationT(std::string filein, std::string fileout, std::string exasimpath, Int mpiprocs, 
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
    CDiscretizationT(std::string filein, std::string fileout, std::string exasimpath, Int mpiprocs,
                    Int mpirank, Int ompthreads, Int omprank, Int backend, Int builtinmodelID)
        : CDiscretizationT(filein, fileout, exasimpath, mpiprocs, mpirank, ompthreads, omprank,
                          backend, builtinmodelID, ExasimDriverABI{}) {}

    // In-memory (no-ABI) constructor (P0): build the discretization from an already-preprocessed,
    // in-memory mesh (exasim::Preprocessed from ExasimSolver<M>::set_mesh) instead of reading datain
    // binaries -- no files, no driver_abi. DEFINED out-of-line and inline in
    // <backend/Discretization/discretization_inmemory.hpp> (a consumer-only header included after
    // buildstructs.hpp, where Preprocessed is complete); the backend unity build never instantiates it.
    CDiscretizationT(exasim::PreprocessedT<T, I>&& pre, std::string fileout, std::string exasimpath, Int mpiprocs,
                    Int mpirank, Int fileoffset, Int omprank, Int backend, Int builtinmodelID);

    // Convenience in-memory ctor for the common serial operator-export case: just the
    // preprocessed bundle + the backend. Defaults fileout="", serial (1 rank, no file
    // offset). A hand-written templated model dispatches on M (not on a builtinmodelID),
    // so the model id is not a parameter here. exasimpath="" relies on the baked data
    // dir / $EXASIM_DATA_DIR for the master/gauss node files. Defined inline in
    // discretization_inmemory.hpp (delegates to the full ctor).
    CDiscretizationT(exasim::PreprocessedT<T, I>&& pre, Int backend, std::string exasimpath = "");

    // MPI variant of the convenience ctor: adds ONLY the MPI rank/size (each rank passes its
    // own per-rank bundle, e.g. from CPreprocessing::takeParallel). Everything else defaults
    // as in the serial convenience ctor (no fileoffset/omprank/model-id in the call).
    CDiscretizationT(exasim::PreprocessedT<T, I>&& pre, Int backend, Int mpiprocs, Int mpirank,
                    std::string exasimpath = "");

    // destructor
    ~CDiscretizationT();

    // post-init construction tail (read_uh/vis/geometry/mass-inverse/HDG setup), shared by the
    // file and in-memory constructors. Defined in discretization.cpp (no Preprocessed dependency).
    void finalizeConstruction(Int backend, ExasimExecutionMode mode,
                              Int nsca, Int nvec, Int nten, Int nsurf, Int nvqoi, Int saveParaview);
        
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

    // Batched, backend-portable (CPU/CUDA/HIP), MPI-ready L2 projection of a DG
    // field from a source nodal basis onto the target (this pass's) basis, one
    // element at a time. Thin wrapper over DGProjection (dgprojection_backend.hpp).
    //   U1      [npe_target * nc * ne]  out
    //   U       [npe_s      * nc * ne]  in (source basis)
    //   shapegs [nge * npe_s]           source shape values at the target Gauss points
    void projectField(dstype* U1, dstype* U, dstype* shapegs, Int npe_s, Int nc, Int backend);
    // On-device validation of the batched projection: an identity projection
    // (source basis == target basis) must reproduce a real field to ~machine
    // precision on every MPI rank. Runs on the active backend; gated by the
    // EXASIM_TEST_PROJECTION env var during construction.
    void projectionSelfTest(Int backend);

    // (interface/boundary sampling methods moved to CInterfaceSampler)
};
using CDiscretization = CDiscretizationT<::dstype, ::Int>;

// Phase 2: the backend-path members below are defined in discretization.cpp (compiled once in the
// unity ExasimSolver.cpp build) and explicitly instantiated there for the default precision. Declare
// them extern here so other TUs -- e.g. main.cpp instantiating CSolution<M>, whose inline ctor/dtor
// construct/destroy a `CDiscretization disc` by value -- link to that single definition instead of
// failing to implicitly instantiate an out-of-line body they cannot see. The in-memory ctors
// (discretization_inmemory.hpp) are deliberately NOT listed: consumer TUs that include that header
// see their definitions and instantiate them locally. See docs/internals/precision-threading.md.
extern template CDiscretizationT<::dstype, ::Int>::CDiscretizationT(
    std::string, std::string, std::string, Int, Int, Int, Int, Int, Int, const ExasimDriverABI&,
    Int, Int, Int, Int, Int, ExasimExecutionMode, const std::vector<dstype>*, Int);
extern template CDiscretizationT<::dstype, ::Int>::~CDiscretizationT();
extern template void CDiscretizationT<::dstype, ::Int>::finalizeConstruction(
    Int, ExasimExecutionMode, Int, Int, Int, Int, Int, Int);
extern template void CDiscretizationT<::dstype, ::Int>::compGeometry(Int);
extern template void CDiscretizationT<::dstype, ::Int>::compMassInverse(Int);
extern template void CDiscretizationT<::dstype, ::Int>::projectField(dstype*, dstype*, dstype*, Int, Int, Int);
extern template void CDiscretizationT<::dstype, ::Int>::projectionSelfTest(Int);
extern template void CDiscretizationT<::dstype, ::Int>::DG2CG(dstype*, dstype*, dstype*, Int, Int, Int, Int);
extern template void CDiscretizationT<::dstype, ::Int>::DG2CG2(dstype*, dstype*, dstype*, Int, Int, Int, Int);
extern template void CDiscretizationT<::dstype, ::Int>::DG2CG3(dstype*, dstype*, dstype*, Int, Int, Int, Int);

#endif        
