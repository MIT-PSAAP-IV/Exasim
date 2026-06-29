/**
 * @file solution.h
 * @brief Defines the CSolution class for managing and solving PDE problems.
 *
 * The CSolution class encapsulates the main components and routines for
 * discretization, preconditioning, and solving linear/nonlinear systems
 * arising from PDEs. It provides interfaces for steady-state and time-dependent
 * problem solving, solution initialization, and input/output operations.
 */

 /**
  * @class CSolution
  * @brief Main class for handling PDE solution workflow.
  *
  * This class manages the discretization, preconditioning, and solver modules,
  * and provides methods for solving steady-state and time-dependent problems,
  * as well as saving and loading solutions and outputs.
  *
  * @section Members
  * - CDiscretization disc: Handles spatial discretization.
  * - CPreconditioner prec: Manages preconditioning for solvers.
  * - CSolver solv: Provides linear and nonlinear solver routines.
  *
  * @section Methods
  * - CSolution(...): Constructor initializing all components.
  * - ~CSolution(): Destructor.
  * - SteadyProblem(...): Solve steady-state problems.
  * - SteadyProblem_PTC(...): Solve steady-state problems using PTC.
  * - TimeStepping(...): Advance solution in time using DIRK/BDF schemes.
  * - UnsteadyProblem(...): Solve time-dependent problems.
  * - DIRK(...): Time integration using DIRK scheme.
  * - InitSolution(...): Precompute quantities for solution.
  * - SolveProblem(...): Solve both steady-state and time-dependent problems.
  * - SaveSolutions(...): Save solutions to binary files.
  * - SaveSolutionsOnBoundary(...): Save boundary solutions to binary files.
  * - SaveNodesOnBoundary(...): Save boundary nodes to binary files.
  * - ReadSolutions(...): Read solutions from binary files.
  * - SaveOutputDG(...): Save DG output to binary files.
  * - SaveOutputCG(...): Save CG output to binary files.
  */
#ifndef __SOLUTION_H__
#define __SOLUTION_H__

#include "exasim/execution_mode.hpp"
#include "../Discretization/assembler.h"
#include "../Discretization/residualeval.h"
#include "../Discretization/interfacesampler.h"
#include "solutionwriter.h"
#include "nonlinearsolver.h"

// Common helper: open file and write 3-element header [a0, a1, a2]
void open_and_write(std::ofstream& ofs,
                    const std::string& prefix,
                    int rank, int offset,
                    int a0, int a1, int a2,
                    const std::string& fileout)
{
    std::string filename = fileout + prefix +
                           NumberToString(rank - offset) + ".bin";
    ofs.open(filename.c_str(), std::ios::out | std::ios::binary);
    if (!ofs) error("Failed to open file: " + filename);

    dstype a[3] = { dstype(a0), dstype(a1), dstype(a2) };
    writearray(ofs, a, 3);
}

void printinterfaceinfo(CDiscretization &disc)
{
    disc.common.printinfo();
    
    int nnbintf = disc.common.nnbintf;
    int nfacesend = disc.common.nfacesend;
    int nfacerecv = disc.common.nfacerecv;
        
  //printf("%d %d %d %d %d %d %d %d\n", common.mpiRank, common.couplingparams.coupledboundarycondition, common.couplingparams.nintfaces, common.nfacerecv, common.nfacesend, common.couplingparams.ncie, common.meshsizes.ne, ncu12);
    printf("coupled boundary condition: %d\n", disc.common.couplingparams.coupledboundarycondition);      
    printf("coupled interface condition: %d\n", disc.common.couplingparams.coupledcondition);      
    printf("szinterfacefluxmap: %d\n", disc.common.szinterfacefluxmap);      
    printf("number of neighboring interface subdomains: %d\n", nnbintf);      
    printf("number of faces to send: %d\n", nfacesend);
    printf("number of faces to receive: %d\n", nfacerecv);
    printf("number of interior elements: %d\n", disc.common.meshsizes.ne0);
    printf("number of interior+interface elements: %d\n", disc.common.meshsizes.ne1);
    printf("number of interior+interface+exterior elements: %d\n", disc.common.meshsizes.ne);
  
    printf("nbintf array: %d by %d\n", 1, nnbintf);  
    print2iarray(disc.common.nbintf, 1, nnbintf);   
    printf("facesend array: %d by %d\n", 1, nfacesend);  
    print2iarray(disc.common.facesend, 1, nfacesend);   
    printf("facerecv array: %d by %d\n", 1, nfacerecv);  
    print2iarray(disc.common.facerecv, 1, nfacerecv);   
    printf("facesendpts array: %d by %d\n", 1, nnbintf);  
    print2iarray(disc.common.facesendpts, 1, nnbintf);   
    printf("facerecvpts array: %d by %d\n", 1, nnbintf);  
    print2iarray(disc.common.facerecvpts, 1, nnbintf);   

    printf("interfacefluxmap array: %d by %d\n", 1, disc.common.szinterfacefluxmap);  
    print2iarray(disc.app.interfacefluxmap, 1, disc.common.szinterfacefluxmap);     
    printf("faceperm array: %d by %d\n", 1, disc.mesh.szfaceperm);  
    print2iarray(disc.mesh.faceperm, 1, disc.mesh.szfaceperm);     
    printf("intfaces array: %d by %d\n", 1, disc.common.couplingparams.nintfaces);  
    print2iarray(disc.mesh.intfaces, 1, disc.common.couplingparams.nintfaces);     
    printf("bf array: %d by %d\n", disc.common.meshsizes.nfe, disc.common.meshsizes.ne);  
    print2iarray(disc.mesh.bf, disc.common.meshsizes.nfe, disc.common.meshsizes.ne);     
    printf("eblks array: %d by %d\n", 3, disc.common.meshsizes.nbe);  
    print2iarray(disc.common.eblks, 3, disc.common.meshsizes.nbe);    
    printf("xdgint array: %d by %d\n", disc.common.grid.npf, disc.common.couplingparams.nintfaces*disc.common.components.ncx);  
    print2darray(disc.sol.xdgint, disc.common.grid.npf, disc.common.couplingparams.nintfaces*disc.common.components.ncx);                      
    printf("xdg array: %d by %d\n", disc.common.grid.npe, disc.common.meshsizes.ne*disc.common.components.ncx);  
    print2darray(disc.sol.xdg, disc.common.grid.npe, disc.common.meshsizes.ne*disc.common.components.ncx);                      
    // printf("udg array: %d by %d\n", disc.common.grid.npe, disc.common.meshsizes.ne*disc.common.components.nc);  
    // print2darray(disc.sol.udg, disc.common.grid.npe, disc.common.meshsizes.ne*disc.common.components.nc);                      
}

// Templated on the user Model type M (default = AbiAdapter, the runtime-ABI build). This is the
// top of the model-dependent FEM stack: it owns the model-parameterized pieces by value, so M
// threads from here (or from the header facade's run<M>()) down to every model call. For
// M=AbiAdapter the build is byte-identical to the non-templated original; a concrete M makes the
// solve fully inlined with no driver_abi. disc/sampler/vis are model-free and stay non-templated.
template <class M>
class CSolution {
private:
    struct PDEStateSnapshot {
        bool initialized = false;
        dstype *udg = nullptr;
        dstype *uh = nullptr;
        dstype *wdg = nullptr;
        dstype *odg = nullptr;
    };

    PDEStateSnapshot snapshot;
public:
    CDiscretization disc;  // spatial discretization class (the function space)
    CResidual<M> residual;    // the discretized PDE residual R(u)/flux q (evaluates from disc)
    CAssembler<M> assembler;  // HDG global linear-system assembler + operator-apply (from disc)
    CInterfaceSampler sampler; // interface/boundary field sampling for coupling (from disc)
    CPreconditioner<M> prec;  // precondtioner class
    CSolver<M> solv;          // linear and nonlinear solvers
    CVisualization vis;    // visualization class
    CSolutionWriter<M> writer; // solution output (streams + Save*/Read*/Get*/evalOutput) -- the I/O half
    CNonlinearSolver<M> nonlinear; // PTC/Newton nonlinear iterations -- the nonlinear-solver half

    // constructor 
    CSolution(string filein, string fileout, string exasimpath, Int mpiprocs,
              Int mpirank, Int fileoffset, Int omprank, Int backend,
              Int builtinmodelID, const ExasimDriverABI& abi,
              Int nsca = 0, Int nvec = 0, Int nten = 0,
              Int nsurf = 0, Int nvqoi = 0,
              ExasimExecutionMode mode = ExasimExecutionMode::Solve,
              const std::vector<dstype>* physicsparamOverride = nullptr,
              Int saveParaview = 0)
       : disc(filein, fileout, exasimpath, mpiprocs, mpirank, fileoffset,
              omprank, backend, builtinmodelID, abi, nsca, nvec, nten, nsurf, nvqoi, mode,
              physicsparamOverride, saveParaview),
         residual(disc), assembler(disc), sampler(disc),
         prec(disc, backend, mode), solv(disc, backend, mode), vis(disc, backend),
         writer(disc, residual, vis, solv),
         nonlinear(disc, residual, assembler, prec, solv, writer)
    {
        if ((disc.common.couplingparams.nintfaces > 0) && (disc.common.couplingparams.coupledcondition>0)) disc.common.meshsizes.ne0 = disc.common.intepartpts[0];

        const bool postprocessOnly = (mode == ExasimExecutionMode::Postprocess);

        // The operator initializes its own solution: first the model initial conditions (layer A,
        // fields the reader could not supply), then recover the operator state (q / uh / q-matrices)
        // from that initialized u -- before the initial solution is written. Both were in the
        // discretization constructor; they belong to the operator (CResidual), not the function space.
        residual.initializeSolution();
        residual.recoverInitialState(backend, postprocessOnly);

        // Open the output streams and write the initial solution (the I/O half lives on the writer).
        writer.setup(postprocessOnly);
    };

    // No-ABI constructor (C3): the concrete-model build (M != AbiAdapter) has no runtime ABI -- the
    // solve is fully inlined through the templated exasim::Name<M> kernels. Matches run.hpp /
    // solver_facade.hpp's 9-arg call; delegates to the ABI constructor with a default (all-null) ABI
    // (its fn-pointers are only dereferenced by the discarded AbiAdapter branch, never for concrete M).
    CSolution(string filein, string fileout, string exasimpath, Int mpiprocs,
              Int mpirank, Int fileoffset, Int omprank, Int backend, Int builtinmodelID)
        : CSolution(filein, fileout, exasimpath, mpiprocs, mpirank, fileoffset, omprank,
                    backend, builtinmodelID, ExasimDriverABI{}) {}


    // destructor (output streams are owned by, and closed by, the writer)
    ~CSolution() {
        this->ClearSavedState();
    };

    void SteadyProblem(ofstream &out, Int backend);

    void SteadyProblem_PTC(ofstream &out, Int backend);

    // (evalMonitor / evalOutput / SaveSolutions / SaveQoI / SaveParaview / SaveSolutionsOnBoundary /
    //  SaveNodesOnBoundary / ResetOutputFiles / ReadSolutions / GetSolutions / SaveOutputCG moved to
    //  CSolutionWriter -- call them via the `writer` member)

    void DIRK(ofstream &out, Int backend);

    // precompute some quantities
    void InitSolution(Int backend);

    void SolveProblem(ofstream &out, Int backend);

    void SaveState();
    void RestoreState();
    void ClearSavedState();

    // (PTCsolver / NewtonSolver moved to CNonlinearSolver -- call via the `nonlinear` member)
};

#endif        
