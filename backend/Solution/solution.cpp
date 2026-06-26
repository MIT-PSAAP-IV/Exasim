/*
    solution.cpp

    This file implements the CSolution class methods for solving PDEs using various numerical schemes.
    The main functionalities include initialization, time-stepping, steady-state solving, saving and reading solutions,
    and handling output for both DG and CG methods. The code supports parallel execution (MPI), adaptive timestepping,
    and artificial viscosity (AV) field computation.

    Main Methods:
    -------------
    - SteadyProblem: Solves steady-state problems, computes AV fields, and prepares solution data for output.
    - InitSolution: Initializes solution variables, sets up geometry, mass matrix, and prepares for time-dependent or steady problems.
    - DIRK: Implements time-stepping using Diagonally Implicit Runge-Kutta (DIRK) schemes, including source term updates and solution saving.
    - SteadyProblem_PTC: Solves steady problems using Pseudo-Transient Continuation (PTC) with adaptive timestep control and convergence monitoring.
    - SolveProblem: Entry point for solving either time-dependent or steady-state problems, including solution initialization and output.
    - SaveSolutions / ReadSolutions: Save and load solution data to/from binary files, supporting both time-dependent and steady-state cases.
    - SaveOutputDG / SaveOutputCG: Save DG and CG output data, including post-processing and conversion between DG and CG representations.
    - SaveSolutionsOnBoundary / SaveNodesOnBoundary: Save solution and node data on domain boundaries for post-processing or coupling.

    Features:
    ---------
    - Supports multiple spatial discretization schemes (HDG, DG, etc.).
    - Handles artificial viscosity computation and smoothing, including parallel communication.
    - Adaptive timestep control for PTC and DIRK schemes.
    - MPI parallelization for distributed memory computation.
    - Flexible output and checkpointing for solutions and boundary data.
    - Modular design with external dependencies for geometry, mass matrix, and residual computation.

    Usage:
    ------
    - Instantiate CSolution and call SolveProblem() to solve a PDE problem.
    - Use SaveSolutions(), SaveOutputDG(), SaveOutputCG() for output and post-processing.
    - Configure via disc.common for problem-specific parameters (timestepping, output frequency, AV options, etc.).

    Note:
    -----
    - Requires external files: solution.h, previoussolutions.cpp, updatesolution.cpp, updatesource.cpp, timestepcoeff.cpp, avsolution.cpp.
    - Some methods rely on external functions for array manipulation, MPI communication, and numerical linear algebra.
    - Timing and debugging output are controlled via preprocessor macros (TIMING, TIMESTEP, HAVE_MPI, etc.).
*/
#ifndef __SOLUTION
#define __SOLUTION

#include "solution.h"
#include "previoussolutions.cpp"
#include "updatesolution.cpp"
#include "updatesource.cpp"
#include "timestepcoeff.cpp"
#include "avsolution.cpp"

#include <chrono>

#ifdef TIMESTEP  
#include <sys/time.h>
#endif

namespace {

void printFirstNonFiniteFlat(const char* label, const dstype* data, Int size, Int rank)
{
    dstype maxabs = 0.0;
    Int imax = -1;
    for (Int i = 0; i < size; ++i) {
        dstype absval = std::abs(data[i]);
        if (absval > maxabs) {
            maxabs = absval;
            imax = i;
        }
        if (!std::isfinite(data[i])) {
            std::cout << "First non-finite entry in " << label
                      << " on rank " << rank
                      << " at flat index " << i
                      << " with value " << data[i]
                      << "; max abs entry is at index " << imax
                      << " with value " << data[imax] << std::endl;
            return;
        }
    }

    std::cout << "No non-finite entry found in " << label
              << " on rank " << rank
              << "; max abs entry is at index " << imax
              << " with value " << data[imax] << std::endl;
}

double SolutionBenchmarkNowMs()
{
    return std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

void SolutionBenchmarkFence(const Int backend)
{
    Kokkos::fence();
#ifdef HAVE_CUDA
    if (backend == 2)
        CHECK(cudaDeviceSynchronize());
#endif
#ifdef HAVE_HIP
    if (backend == 3)
        CHECK(hipDeviceSynchronize());
#endif
}

double SolutionBenchmarkStart(const Int backend)
{
    SolutionBenchmarkFence(backend);
    return SolutionBenchmarkNowMs();
}

double SolutionBenchmarkStop(const double start, const Int backend)
{
    SolutionBenchmarkFence(backend);
    return SolutionBenchmarkNowMs() - start;
}

}

void CSolution::SaveState()
{
    Int backend = disc.common.backend;

    if (snapshot.udg == nullptr && disc.sol.szudg > 0)
        TemplateMalloc(&snapshot.udg, disc.sol.szudg, backend);

    if (disc.common.spatialScheme == 1 && snapshot.uh == nullptr && disc.sol.szuh > 0)
        TemplateMalloc(&snapshot.uh, disc.sol.szuh, backend);

    if (disc.common.components.ncw > 0 && snapshot.wdg == nullptr && disc.sol.szwdg > 0)
        TemplateMalloc(&snapshot.wdg, disc.sol.szwdg, backend);

    if (disc.common.components.nco > 0 && snapshot.odg == nullptr && disc.sol.szodg > 0)
        TemplateMalloc(&snapshot.odg, disc.sol.szodg, backend);

    if (disc.sol.szudg > 0)
        ArrayCopy(disc.common.cublasHandle, snapshot.udg, disc.sol.udg, disc.sol.szudg, backend);

    if (disc.common.spatialScheme == 1 && disc.sol.szuh > 0)
        ArrayCopy(disc.common.cublasHandle, snapshot.uh, disc.sol.uh, disc.sol.szuh, backend);

    if (disc.common.components.ncw > 0 && disc.sol.szwdg > 0)
        ArrayCopy(disc.common.cublasHandle, snapshot.wdg, disc.sol.wdg, disc.sol.szwdg, backend);

    if (disc.common.components.nco > 0 && disc.sol.szodg > 0)
        ArrayCopy(disc.common.cublasHandle, snapshot.odg, disc.sol.odg, disc.sol.szodg, backend);

    snapshot.initialized = true;
}

void CSolution::RestoreState()
{
    Int backend = disc.common.backend;

    if (snapshot.initialized == false)
        error("No saved PDE state is available to restore.");

    if (disc.sol.szudg > 0)
        ArrayCopy(disc.common.cublasHandle, disc.sol.udg, snapshot.udg, disc.sol.szudg, backend);

    if (disc.common.spatialScheme == 1 && disc.sol.szuh > 0)
        ArrayCopy(disc.common.cublasHandle, disc.sol.uh, snapshot.uh, disc.sol.szuh, backend);

    if (disc.common.components.ncw > 0 && disc.sol.szwdg > 0)
        ArrayCopy(disc.common.cublasHandle, disc.sol.wdg, snapshot.wdg, disc.sol.szwdg, backend);

    if (disc.common.components.nco > 0 && disc.sol.szodg > 0)
        ArrayCopy(disc.common.cublasHandle, disc.sol.odg, snapshot.odg, disc.sol.szodg, backend);

    if (disc.common.spatialScheme == 0) {
        ArrayExtract(solv.sys.u, disc.sol.udg, disc.common.grid.npe, disc.common.components.nc, disc.common.meshsizes.ne1,
              0, disc.common.grid.npe, 0, disc.common.components.ncu, 0, disc.common.meshsizes.ne1);
    }
    else if (disc.common.spatialScheme == 1) {
        ArrayCopy(disc.common.cublasHandle, solv.sys.u, disc.sol.uh, disc.common.sizes.ndofuhat, backend);
    }
    else {
        error("Spatial discretization scheme is not implemented");
    }
}

void CSolution::ClearSavedState()
{
    Int backend = disc.common.backend;

    if (snapshot.udg != nullptr) {
        TemplateFree(snapshot.udg, backend);
        snapshot.udg = nullptr;
    }

    if (snapshot.uh != nullptr) {
        TemplateFree(snapshot.uh, backend);
        snapshot.uh = nullptr;
    }

    if (snapshot.wdg != nullptr) {
        TemplateFree(snapshot.wdg, backend);
        snapshot.wdg = nullptr;
    }

    if (snapshot.odg != nullptr) {
        TemplateFree(snapshot.odg, backend);
        snapshot.odg = nullptr;
    }

    snapshot.initialized = false;
}

Int CSolution::PTCsolver(ofstream &out, Int backend)       
{
    Int N = disc.common.sizes.ndof1;     
    Int it = 0, maxit = disc.common.solverparams.nonlinearSolverMaxIter;  
    dstype nrmr, tol;
    tol = disc.common.solverparams.nonlinearSolverTol; // tolerance for the residual
    
    nrmr = PNORM(disc.common.cublasHandle, N, solv.sys.u, backend);
    if (disc.common.mpiRank==0)
        cout<<"Newton Iteration: "<<it<<",  Solution Norm: "<<nrmr<<endl;                                                    
    
    // compute both the residual vector and sol.udg  
    residual.evalResidual(solv.sys.r, solv.sys.u, backend);
    nrmr = PNORM(disc.common.cublasHandle, N, solv.sys.r, backend);
    if (disc.common.mpiRank==0)
        cout<<"Newton Iteration: "<<it<<",  Residual Norm: "<<nrmr<<endl;                           
    
    // use PTC to solve the system: R(u) = 0
    for (it=0; it<maxit; it++) {                        
        double ldgIterationStart = SolutionBenchmarkStart(backend);
        double ldgPreconditionerTime = 0.0;
        double linearSolverTime = 0.0;
        double residualEvalTime = 0.0;
        double t0;

        // Build the LDG block-Jacobi preconditioner for the current state.
        if ((disc.common.spatialScheme == 0) && (disc.common.solverparams.preconditioner == 1)) {
            t0 = SolutionBenchmarkStart(backend);
            if (disc.common.timeparams.tdep==1) {
                if (it==0 && disc.common.timestate.currentstage==0) prec.ComputeLDGPreconditioner(disc, disc.res.K, solv.sys.u, backend);
            } else
                prec.ComputeLDGPreconditioner(disc, disc.res.K, solv.sys.u, backend);
            ldgPreconditionerTime += SolutionBenchmarkStop(t0, backend);
        }

        dstype nrm0 = nrmr;

        int status = 0;
        const dstype minAlpha = 1.0e-12;
        const Int maxLinearAttempts = 4;
        dstype alpha = one;
        bool acceptedStep = false;

        for (Int attempt = 0; attempt < maxLinearAttempts; attempt++) {
            // solve the linear system: (lambda*B + J(u))x = -R(u)
            t0 = SolutionBenchmarkStart(backend);
            status = solv.linearSolve(residual, assembler, disc, prec, out, it, backend);
            linearSolverTime += SolutionBenchmarkStop(t0, backend);

            ArrayCopy(disc.common.cublasHandle, solv.sys.v, solv.sys.u, N, backend);
            alpha = one;

            // update the solution: u = u + alpha*x
            ArrayAXPY(disc.common.cublasHandle, solv.sys.u, solv.sys.x, alpha, N, backend);

            // compute both the residual vector and sol.udg
            residual.evalResidual(solv.sys.r, solv.sys.u, backend);
            nrmr = PNORM(disc.common.cublasHandle, N, solv.sys.r, backend);

            while ((IS_NAN(nrmr) || nrmr > nrm0) && alpha > minAlpha) {
                if (disc.common.mpiRank==0)
                    cout<<"Newton Iteration: "<<it<<", Alpha: "<<alpha
                        <<", Original Norm: "<<nrm0
                        <<", Updated Norm: "<<nrmr<<endl;

                dstype newAlpha = 0.5*alpha;
                ArrayAXPY(disc.common.cublasHandle, solv.sys.u, solv.sys.x,
                        newAlpha - alpha, N, backend);
                alpha = newAlpha;

                t0 = SolutionBenchmarkStart(backend);
                residual.evalResidual(solv.sys.r, solv.sys.u, backend);
                nrmr = PNORM(disc.common.cublasHandle, N, solv.sys.r, backend);
                residualEvalTime += SolutionBenchmarkStop(t0, backend);
            }

            acceptedStep = (!IS_NAN(nrmr) && nrmr <= nrm0 && nrmr <= 1.0e6);
            if (acceptedStep)
                break;

            // Reject this direction and restore the base state before retrying.
            ArrayCopy(disc.common.cublasHandle, solv.sys.u, solv.sys.v, N, backend);
            t0 = SolutionBenchmarkStart(backend);
            residual.evalResidual(solv.sys.r, solv.sys.u, backend);
            nrmr = PNORM(disc.common.cublasHandle, N, solv.sys.r, backend);
            residualEvalTime += SolutionBenchmarkStop(t0, backend);

            if (attempt + 1 < maxLinearAttempts) {
                disc.common.solverparams.matvecTol *= 0.1;
                if (disc.common.mpiRank==0)
                    cout<<"Newton Iteration: "<<it
                        <<", rejected linear direction; retrying with matvecTol = "
                        <<disc.common.solverparams.matvecTol<<endl;
            }
        }

        if (acceptedStep && alpha != one)
            ArrayMultiplyScalar(disc.common.cublasHandle, solv.sys.x, alpha, N, backend);

        if (!acceptedStep) {
            string filename = disc.common.fileout + "_np" + NumberToString(disc.common.mpiRank) + ".bin";                    
            writearray2file(filename, disc.sol.udg, disc.common.sizes.ndofudg1, backend);       
            if (vis.savemode > 0) this->SaveParaview(backend, "_CRASH", true);     
            if (outsol.is_open()) { outsol.close(); }
            if (outwdg.is_open()) { outwdg.close(); }
            if (outuhat.is_open()) { outuhat.close(); }
            if (outbouxdg.is_open()) { outbouxdg.close(); }
            if (outboundg.is_open()) { outboundg.close(); }
            if (outbouudg.is_open()) { outbouudg.close(); }
            if (outbouwdg.is_open()) { outbouwdg.close(); }
            if (outbouuhat.is_open()) { outbouuhat.close(); }
            if (outqoi.is_open()) { outqoi.close(); }              
            error("Newton line search failed or residual norm is non-finite. Save and exit.");
        }
        
        if (disc.common.mpiRank==0 && disc.common.outputparams.saveResNorm==1) {
            disc.common.timing[122] = it + 0.0; 
            disc.common.timing[123] = nrmr;        
            writearray(out, &disc.common.timing[120], 4);    
        }
        
        if (disc.common.mpiRank==0)
            cout<<"Newton Iteration: "<<it<<",  Residual Norm: "<<nrmr<<endl;                           

        if ((disc.common.mpiRank==0) && (disc.common.spatialScheme == 0) && (disc.common.solverparams.preconditioner == 1)) {
            double ldgIterationTime = SolutionBenchmarkStop(ldgIterationStart, backend);
            cout << "==> LDG Newton Solver benchmark, iteration " << it << " (milliseconds)" << endl;
            cout << "    total time       : " << ldgIterationTime << endl;
            cout << "    ComputeLDGPreconditioner: " << ldgPreconditionerTime << endl;
            cout << "    LinearSolver/GMRES    : " << linearSolverTime << endl;
        }
                        
        // update the reduced basis
        if ((status==0) && (disc.common.solverparams.RBdim > 0)) // fix bug here 
            solv.updateRB(disc, prec, backend);      
        
        // check convergence
        if (nrmr < tol) {            
            return it;   
        }
    }
        
    return it;
}

Int CSolution::NewtonSolver(ofstream &out, Int N, Int spatialScheme, Int backend)       
{
    Int it = 0, maxit = disc.common.solverparams.nonlinearSolverMaxIter;  
    dstype nrmr, nrm0, tol;
    tol = disc.common.solverparams.nonlinearSolverTol; // tolerance for the residual
                
    nrmr = PNORM(disc.common.cublasHandle, N, disc.common.couplingparams.ndofuhatinterface, solv.sys.u, backend);
//     if (disc.common.mpiProcs>1 && disc.common.spatialScheme==1) {
//       dstype nrm = PNORM(disc.common.cublasHandle, disc.common.components.ncu*disc.common.grid.npf*disc.common.couplingparams.ninterfacefaces, sys.u, backend);
//       nrmr = sqrt(nrmr*nrmr - 0.5*nrm*nrm);
//     }                
    
    if (disc.common.mpiRank==0)
      cout<<"Newton Iteration: "<<it<<",  Solution Norm: "<<nrmr<<endl;                                                        

    if (disc.common.outputparams.debugMode==1) {
      writearray2file(disc.common.fileout + NumberToString(it) + "newton_uh.bin", disc.sol.uh, N, backend);
      writearray2file(disc.common.fileout + NumberToString(it) + "newton_udg.bin", disc.sol.udg, disc.common.grid.npe*disc.common.components.nc*disc.common.meshsizes.ne1, backend);
    }

    if (spatialScheme == 1) { 

      if (disc.common.components.ncq > 0) hdgGetQ<exasim::detail::AbiAdapter>(disc.sol.udg, disc.sol.uh, disc.sol, disc.res, disc.mesh, disc.tmp, disc.common, backend);                
      if (disc.common.components.ncw > 0) GetW<exasim::detail::AbiAdapter>(disc.sol.wdg, disc.sol, disc.tmp, disc.app, disc.common, backend);
      
      // compute the residual vector R = [Ru; Rh]
      assembler.hdgAssembleResidual(solv.sys.b, backend);
            
      nrmr = PNORM(disc.common.cublasHandle, N, disc.common.couplingparams.ndofuhatinterface, solv.sys.b, backend);       
      // cout<<"Rank = "<<disc.common.mpiRank<<", norm Rh = "<<NORM(disc.common.cublasHandle, N, solv.sys.b, backend)<<endl;          
      // cout<<"Rank = "<<disc.common.mpiRank<<", norm Ru = "<<NORM(disc.common.cublasHandle, disc.common.grid.npe*disc.common.components.ncu*disc.common.meshsizes.ne1, disc.res.Ru, backend)<<endl;          
      // if (IS_NAN(nrmr)) {
      //   for (int m=0; m<N; m++) { 
      //     nrm0 = solv.sys.b[m];
      //     if (IS_NAN(nrm0)) 
      //       cout<<"Rank = "<<disc.common.mpiRank<<", m = "<<m<<", Rh[m] = "<<solv.sys.b[m]<<endl;       
      //   }
      //   if (disc.common.mpiRank==0) cout<<"Rhat is nan"<<endl;
      //   printFirstNonFiniteFlat("sys.b", solv.sys.b, N, disc.common.mpiRank);
      // }
      nrmr += PNORM(disc.common.cublasHandle, disc.common.grid.npe*disc.common.components.ncu*disc.common.meshsizes.ne1, disc.res.Ru, backend);                 
      // nrmr += nrmru;
      // if (IS_NAN(nrmru)) {        
      //   for (int m=0; m<disc.common.grid.npe*disc.common.components.ncu*disc.common.meshsizes.ne1; m++) {
      //     nrm0 = disc.res.Ru[m];
      //     if (IS_NAN(nrm0)) 
      //       cout<<"Rank = "<<disc.common.mpiRank<<", m = "<<m<<", Ru[m] = "<<disc.res.Ru[m]<<endl;       
      //   }   
      //   if (disc.common.mpiRank==0) cout<<"Ru is nan"<<endl;
      //   printFirstNonFiniteFlat("res.Ru", disc.res.Ru, disc.common.grid.npe*disc.common.components.ncu*disc.common.meshsizes.ne1, disc.common.mpiRank);
      // }
      if (disc.common.mpiRank==0)
        cout<<"Newton Iteration: "<<0<<",  Residual Norm: "<<nrmr<<endl;      

      if (IS_NAN(nrmr)) {                        
        string filename = disc.common.fileout + "_np" + NumberToString(disc.common.mpiRank) + ".bin";                    
        writearray2file(filename, disc.sol.udg, disc.common.sizes.ndofudg1, backend);   
        if (disc.common.components.ncw > 0) {
          string filename1 = disc.common.fileout + "_wdg_np" + NumberToString(disc.common.mpiRank) + ".bin";                    
          writearray2file(filename1, disc.sol.wdg, disc.common.grid.npe*disc.common.components.ncw*disc.common.meshsizes.ne1, backend);   
        }
        if (vis.savemode > 0) this->SaveParaview(backend, "_CRASH", true);     
        if (outsol.is_open()) { outsol.close(); }
        if (outwdg.is_open()) { outwdg.close(); }
        if (outuhat.is_open()) { outuhat.close(); }
        if (outbouxdg.is_open()) { outbouxdg.close(); }
        if (outboundg.is_open()) { outboundg.close(); }
        if (outbouudg.is_open()) { outbouudg.close(); }
        if (outbouwdg.is_open()) { outbouwdg.close(); }
        if (outbouuhat.is_open()) { outbouuhat.close(); }
        if (outqoi.is_open()) { outqoi.close(); }              
        error("Residual norm is nan. Save and exit.");                                    
      }
    }                
    
    // use PTC to solve the system: R(u) = 0
    for (it=0; it<maxit; it++) {              
                      
        // solve the linear system:  J(u) x = -R(u)        
        solv.linearSolve(residual, assembler, disc, prec, out, N, spatialScheme, it, backend);
              
        // int npf = disc.common.grid.npf;
        // int nfe = disc.common.meshsizes.nfe;        
        // print3darray(disc.res.Hi, npf, npf*nfe, disc.common.nfacerecv, "Hi", MPI_COMM_WORLD);
        // print2darray(solv.sys.x, npf, disc.common.meshsizes.nf, "uhat face", MPI_COMM_WORLD);
        // GetElementFaceNodes(disc.res.Rq, solv.sys.x, disc.mesh.elemcon, npf*nfe, disc.common.components.ncu, 0, disc.common.meshsizes.ne1, 2);
        // print3darray(&disc.res.Rq[npf*nfe*disc.common.meshsizes.ne0], npf, nfe, 4, "uhat element", MPI_COMM_WORLD);
        // printf("%d %d\n", disc.common.mpiRank, disc.common.meshsizes.ne0);

        solv.sys.alpha = 1.0;        
        // update the solution: u = u + alpha*x
        ArrayAXPY(disc.common.cublasHandle, solv.sys.u, solv.sys.x, solv.sys.alpha, N, backend); 
                
        if (spatialScheme == 0) {          
          // compute both the residual vector and sol.udg  
          residual.evalResidual(solv.sys.r, solv.sys.u, backend);          
          nrmr = PNORM(disc.common.cublasHandle, N, solv.sys.r, backend);          
        } 
        else if (spatialScheme == 1) {      
          ArrayCopy(disc.sol.uh, solv.sys.u, N);
          hdgGetDUDG<exasim::detail::AbiAdapter>(disc.res.Ru, disc.res.F, solv.sys.x, disc.res.Rq, disc.mesh, disc.common, backend);          
          ArrayCopy(solv.sys.v, disc.res.Ru, disc.common.grid.npe*disc.common.components.ncu*disc.common.meshsizes.ne1);
          UpdateUDG(disc.sol.udg, disc.res.Ru, solv.sys.alpha, disc.common.grid.npe, disc.common.components.nc, disc.common.meshsizes.ne1, 0, disc.common.grid.npe, 0, disc.common.components.ncu, 0, disc.common.meshsizes.ne1);                    
                    
          if (disc.common.outputparams.debugMode==1) {
            writearray2file(disc.common.fileout + NumberToString(it+1) + "newton_x.bin", solv.sys.x, N, backend);
            writearray2file(disc.common.fileout + NumberToString(it+1) + "newton_u.bin", solv.sys.u, N, backend);
            writearray2file(disc.common.fileout + NumberToString(it+1) + "newton_uh.bin", disc.sol.uh, N, backend);
            writearray2file(disc.common.fileout + NumberToString(it+1) + "newton_udg.bin", disc.sol.udg, disc.common.grid.npe*disc.common.components.nc*disc.common.meshsizes.ne1, backend);
            error("stop for debugging...");
          }          
                    
          if (disc.common.components.ncq > 0) hdgGetQ<exasim::detail::AbiAdapter>(disc.sol.udg, disc.sol.uh, disc.sol, disc.res, disc.mesh, disc.tmp, disc.common, backend);          
          if (disc.common.components.ncw > 0) GetW<exasim::detail::AbiAdapter>(disc.sol.wdg, disc.sol, disc.tmp, disc.app, disc.common, backend);
                              
          nrm0 = nrmr; // original norm          
          // compute the updated residual norm |[Ru; Rh]|
          assembler.hdgAssembleResidual(solv.sys.b, backend);          
          nrmr = PNORM(disc.common.cublasHandle, N, disc.common.couplingparams.ndofuhatinterface, solv.sys.b, backend);           
          nrmr += PNORM(disc.common.cublasHandle, disc.common.grid.npe*disc.common.components.ncu*disc.common.meshsizes.ne1, disc.res.Ru, backend);   
                    
          if ((nrmr > nrm0 && nrmr > 1.0e6) || IS_NAN(nrmr)) {                        
            string filename = disc.common.fileout + "_np" + NumberToString(disc.common.mpiRank) + ".bin";                    
            writearray2file(filename, disc.sol.udg, disc.common.sizes.ndofudg1, backend);   
            if (disc.common.components.ncw > 0) {
              string filename1 = disc.common.fileout + "_wdg_np" + NumberToString(disc.common.mpiRank) + ".bin";                    
              writearray2file(filename1, disc.sol.wdg, disc.common.grid.npe*disc.common.components.ncw*disc.common.meshsizes.ne1, backend);   
            }
            if (vis.savemode > 0) this->SaveParaview(backend, "_CRASH", true);     
            if (outsol.is_open()) { outsol.close(); }
            if (outwdg.is_open()) { outwdg.close(); }
            if (outuhat.is_open()) { outuhat.close(); }
            if (outbouxdg.is_open()) { outbouxdg.close(); }
            if (outboundg.is_open()) { outboundg.close(); }
            if (outbouudg.is_open()) { outbouudg.close(); }
            if (outbouwdg.is_open()) { outbouwdg.close(); }
            if (outbouuhat.is_open()) { outbouuhat.close(); }
            if (outqoi.is_open()) { outqoi.close(); }              
            error("Residual norm increases more than 1e6 or nan. Save and exit.");                                    
          }
            
          // damped Newton loop to determine alpha
          while (nrmr>nrm0 && solv.sys.alpha > 0.1) 
          {
            if (disc.common.mpiRank==0)
              printf("Newton Iteration: %d, Alpha: %g, Original Norm: %g,  Updated Norm: %g\n", it+1, solv.sys.alpha, nrm0, nrmr);
            solv.sys.alpha = solv.sys.alpha/2.0;             
            ArrayAXPY(disc.common.cublasHandle, solv.sys.u, solv.sys.x, -solv.sys.alpha, N, backend); 
            ArrayCopy(disc.sol.uh, solv.sys.u, N);
            UpdateUDG(disc.sol.udg, solv.sys.v, -solv.sys.alpha, disc.common.grid.npe, disc.common.components.nc, disc.common.meshsizes.ne1, 0, disc.common.grid.npe, 0, disc.common.components.ncu, 0, disc.common.meshsizes.ne1);                    
            if (disc.common.components.ncq > 0) hdgGetQ<exasim::detail::AbiAdapter>(disc.sol.udg, disc.sol.uh, disc.sol, disc.res, disc.mesh, disc.tmp, disc.common, backend);          
            if (disc.common.components.ncw > 0) GetW<exasim::detail::AbiAdapter>(disc.sol.wdg, disc.sol, disc.tmp, disc.app, disc.common, backend);
            assembler.hdgAssembleResidual(solv.sys.b, backend);
            nrmr = PNORM(disc.common.cublasHandle, N, disc.common.couplingparams.ndofuhatinterface, solv.sys.b, backend); 
            nrmr += PNORM(disc.common.cublasHandle, disc.common.grid.npe*disc.common.components.ncu*disc.common.meshsizes.ne1, disc.res.Ru, backend);                       
          }          
        }

        // update the reduced basis space
        ArrayMultiplyScalar(disc.common.cublasHandle, solv.sys.x, solv.sys.alpha, N, backend);   
                        
        if (disc.common.solverparams.RBdim > 0) solv.updateRB(disc, prec, N, backend);         
                
        if (disc.common.mpiRank==0)
          printf("Newton Iteration: %d, Alpha: %g, Original Norm: %g,  Updated Norm: %g\n", it+1, solv.sys.alpha, nrm0, nrmr);
        
        // check convergence
        if (nrmr < tol) return (it+1);           
    }
    
    return it;
}

void CSolution::SteadyProblem(ofstream &out, Int backend) 
{   
    INIT_TIMING;        
#ifdef TIMING    
    for (int i=0; i<100; i++)
        disc.common.timing[i] = 0.0; 
#endif
    // obtain odg from the solutions of the other PDE models
    if (disc.common.nomodels>1) {
        Int nco = disc.common.components.nco;
        Int npe = disc.common.grid.npe;
        Int ne = disc.common.meshsizes.ne;            
        for (Int n=0; n<nco; n++) {            
            Int m = disc.common.vindx[n];     // model index
            Int k = disc.common.vindx[nco+n]; // solution index
            // extract the kth component of udg from PDE model m and store it in Ru
            ArrayExtract(disc.res.Ru, disc.sol.udgarray[m], npe, disc.common.ncarray[m], ne, 0, npe, k, k+1, 0, ne);         
            // insert Ru into odg
            ArrayInsert(disc.sol.odg, disc.res.Ru, npe, nco, ne, 0, npe, n, n+1, 0, ne);          
        }
    }
    
    // calculate AV field
    if (disc.common.physicsparams.ncAV>0 && disc.common.physicsparams.frozenAVflag > 0) {
        // START_TIMING;

        Int nco = disc.common.components.nco;
        Int ncAV = disc.common.physicsparams.ncAV;
        Int npe = disc.common.grid.npe;
        Int ne = disc.common.meshsizes.ne;            
        
        // store AV field 
        dstype *avField = &disc.res.Rq[0];
        dstype *utm = &disc.res.Rq[npe*ncAV*ne];

        // evaluate AV field
        residual.evalAVfield(avField, backend);

        for (Int iav = 0; iav<disc.common.physicsparams.AVsmoothingIter; iav++){
            // printf("Solution AV smoothing iter: %i\n", iav);
            disc.DG2CG2(avField, avField, utm, disc.common.physicsparams.ncAV, disc.common.physicsparams.ncAV, disc.common.physicsparams.ncAV, backend);

#ifdef  HAVE_MPI    
            Int bsz = disc.common.grid.npe*disc.common.physicsparams.ncAV;
            Int nudg = disc.common.grid.npe*disc.common.components.nco;
            Int n;

            //for (n=0; n<disc.common.nelemsend; n++)
            //    ArrayCopy(&disc.tmp.buffsend[bsz*n], &disc.sol.odg[nudg*disc.common.elemsend[n]], bsz, backend);
            GetArrayAtIndex(disc.tmp.buffsend, avField, disc.mesh.elemsendodg, bsz*disc.common.nelemsend);

#ifdef HAVE_CUDA
            cudaDeviceSynchronize();
#endif

#ifdef HAVE_HIP
            hipDeviceSynchronize();
#endif
            
            Int neighbor, nsend, psend = 0, request_counter = 0;
            for (n=0; n<disc.common.nnbsd; n++) {
                neighbor = disc.common.nbsd[n];
                nsend = disc.common.elemsendpts[n]*bsz;
                if (nsend>0) {
                    MPI_Isend(&disc.tmp.buffsend[psend], nsend, MPI_DOUBLE, neighbor, 0,
                        EXASIM_COMM_LOCAL, &disc.common.requests[request_counter]);
                    psend += nsend;
                    request_counter += 1;
                }
            }

            Int nrecv, precv = 0;
            for (n=0; n<disc.common.nnbsd; n++) {
                neighbor = disc.common.nbsd[n];
                nrecv = disc.common.elemrecvpts[n]*bsz;
                if (nrecv>0) {
                    MPI_Irecv(&disc.tmp.buffrecv[precv], nrecv, MPI_DOUBLE, neighbor, 0,
                        EXASIM_COMM_LOCAL, &disc.common.requests[request_counter]);
                    precv += nrecv;
                    request_counter += 1;
                }
            }

            MPI_Waitall(request_counter, disc.common.requests, disc.common.statuses);
            //for (n=0; n<disc.common.nelemrecv; n++)
            //   ArrayCopy(&disc.sol.odg[nudg*disc.common.elemrecv[n]], &disc.tmp.buffrecv[bsz*n], bsz, backend);
            PutArrayAtIndex(avField, disc.tmp.buffrecv, disc.mesh.elemrecvodg, bsz*disc.common.nelemrecv);
#endif
    //    END_TIMING_DISC(98);    
        }

        // insert avField into odg
        ArrayInsert(disc.sol.odg, avField, npe, nco, ne, 0, npe, nco-ncAV, nco, 0, ne);          
    }

    if (disc.common.components.nco>0) {
        for (Int j=0; j<disc.common.meshsizes.nbe; j++) {
            Int e1 = disc.common.eblks[3*j]-1;
            Int e2 = disc.common.eblks[3*j+1];                
            GetElemNodes(disc.tmp.tempn, disc.sol.odg, disc.common.grid.npe, disc.common.components.nco, 
                    0, disc.common.components.nco, e1, e2);        
            Node2Gauss(disc.common.cublasHandle, &disc.sol.odgg[disc.common.grid.nge*disc.common.components.nco*e1], 
              disc.tmp.tempn, disc.master.shapegt, disc.common.grid.nge, disc.common.grid.npe, (e2-e1)*disc.common.components.nco, backend);        
        }         
        for (Int j=0; j<disc.common.meshsizes.nbf; j++) {
            Int f1 = disc.common.fblks[3*j]-1;
            Int f2 = disc.common.fblks[3*j+1];            
            
            GetFaceNodes(disc.tmp.tempn, disc.sol.odg, disc.mesh.facecon, disc.common.grid.npf, disc.common.components.nco, 
                    disc.common.grid.npe, disc.common.components.nco, f1, f2, 1);          
            Node2Gauss(disc.common.cublasHandle, &disc.sol.og1[disc.common.grid.ngf*disc.common.components.nco*f1], 
              disc.tmp.tempn, disc.master.shapfgt, disc.common.grid.ngf, disc.common.grid.npf, (f2-f1)*disc.common.components.nco, backend);               
            
            GetFaceNodes(disc.tmp.tempn, disc.sol.odg, disc.mesh.facecon, disc.common.grid.npf, disc.common.components.nco, 
                    disc.common.grid.npe, disc.common.components.nco, f1, f2, 2);          
            Node2Gauss(disc.common.cublasHandle, &disc.sol.og2[disc.common.grid.ngf*disc.common.components.nco*f1], 
              disc.tmp.tempn, disc.master.shapfgt, disc.common.grid.ngf, disc.common.grid.npf, (f2-f1)*disc.common.components.nco, backend);               
        }        
    }
    
    if (disc.common.components.ncs>0) {
        for (Int j=0; j<disc.common.meshsizes.nbe; j++) {
            Int e1 = disc.common.eblks[3*j]-1;
            Int e2 = disc.common.eblks[3*j+1];                
            GetElemNodes(disc.tmp.tempn, disc.sol.sdg, disc.common.grid.npe, disc.common.components.ncs, 0, disc.common.components.ncs, e1, e2);        
            Node2Gauss(disc.common.cublasHandle, &disc.sol.sdgg[disc.common.grid.nge*disc.common.components.ncs*e1], 
              disc.tmp.tempn, disc.master.shapegt, disc.common.grid.nge, disc.common.grid.npe, (e2-e1)*disc.common.components.ncs, backend);        
        } 
    }
    
    // use PTC to solve steady problem
    if (disc.common.spatialScheme==0) {
      this->PTCsolver(out, backend);           
    }
    else if (disc.common.spatialScheme==1) {      
      this->NewtonSolver(out, disc.common.sizes.ndofuhat, disc.common.spatialScheme, backend);           
    }
    else
      error("Spatial discretization scheme is not implemented");
        
#ifdef TIMING         
    if (disc.common.mpiRank==0) {
        printf("\nComputing initial guess time: %g miliseconds\n", disc.common.timing[99]);   
        printf("Computing AV fields time: %g miliseconds\n", disc.common.timing[98]);   
        printf("Nonlinear solver time: %g miliseconds\n", disc.common.timing[97]);                
    }
#endif    
}

void CSolution::InitSolution(Int backend) 
{    
//     // compute the geometry quantities
//     disc.compGeometry(backend);
//     
//     //printArray2D(&disc.sol.elemg[0],disc.common.grid.nge,10,backend);
//     
//     // compute the inverse of the mass matrix
//     disc.compMassInverse(backend);
            
    // compute q
    // if ((disc.common.components.ncq>0) & (disc.common.timeparams.wave==0))
    //     disc.evalQSer(backend);
            
    // // set pointer depending on the matrix type
    // if (disc.common.solverparams.precMatrixType==0)
    //     prec.precond.Cmat = &prec.precond.C[0];
    // else //if (disc.common.solverparams.precMatrixType==2)
    //     prec.precond.Cmat = &disc.res.Minv[0];    
        
    if (disc.common.spatialScheme==0) {
        ArrayExtract(solv.sys.u, disc.sol.udg, disc.common.grid.npe, disc.common.components.nc, disc.common.meshsizes.ne1, 
              0, disc.common.grid.npe, 0, disc.common.components.ncu, 0, disc.common.meshsizes.ne1);                                                  
    }
    else if (disc.common.spatialScheme==1) {      
        ArrayCopy(solv.sys.u, disc.sol.uh, disc.common.sizes.ndofuhat);
    }
    else
        error("Spatial discretization scheme is not implemented");
            
    // save solutions into binary files
    this->SaveNodesOnBoundary(backend);     
    
    if (disc.common.timeparams.tdep==1) { // DIRK schemes
        //DIRK coefficients 
        disc.common.timeparams.temporalScheme = 0; 
        TimestepCoefficents(disc.common); 
                
        if (disc.common.mpiRank==0)
            cout<<"Compute solution average = "<<disc.common.outputparams.compudgavg<<endl;
        
        if (disc.common.outputparams.compudgavg == 1) {
            string filename = disc.common.fileout + "avg_np" + NumberToString(disc.common.mpiRank) + ".bin";
            disc.common.outputparams.readudgavg = fileexists(filename);
            if (disc.common.mpiRank==0)
                cout<<"File exist = "<<disc.common.outputparams.readudgavg<<endl;
            if (disc.common.outputparams.readudgavg == 0)
                ArraySetValue(disc.sol.udgavg, zero, disc.common.sizes.ndofudg1+1);
            else 
                readarrayfromfile(filename, &disc.sol.udgavg, disc.common.sizes.ndofudg1+1, backend);   
        }        
    }    

    if (disc.common.sizes.ndofbou>0) {
        ArraySetValue(disc.sol.bouudgavg, zero, disc.common.sizes.ndofbou*disc.common.components.nc+1);
        ArraySetValue(disc.sol.bouuhavg, zero, disc.common.sizes.ndofbou*disc.common.components.ncu+1);
        if (disc.common.components.ncw > 0) ArraySetValue(disc.sol.bouwdgavg, zero, disc.common.sizes.ndofbou*disc.common.components.ncw+1); 
    }  
}

void CSolution::DIRK(ofstream &out, Int backend)
{    
    INIT_TIMING;        
    
    // initial time
    dstype time = disc.common.timestate.time;           
    
    //DIRK coefficients 
    disc.common.timeparams.temporalScheme = 0; 
    TimestepCoefficents(disc.common); 
                
#ifdef TIMESTEP                  
    struct timeval tv1, tv2;
#endif                
    
    // time stepping with DIRK schemes
    for (Int istep=0; istep<disc.common.timeparams.tsteps; istep++)            
    {            
        // current timestep        
        disc.common.timestate.currentstep = istep;
        
        // store previous solutions to calculate the source term        
        PreviousSolutions(disc.sol, solv.sys, disc.common, backend);

#ifdef TIMESTEP              
        gettimeofday(&tv1, NULL); 
#endif
                    
        // compute the solution at the next step
        for (Int j=0; j<disc.common.timeparams.tstages; j++) {            
            // current timestage
            disc.common.timestate.currentstage = j;
        
            // current time
            disc.common.timestate.time = time + disc.common.dt[istep]*disc.common.DIRKcoeff_t[j];

            if (disc.common.mpiRank==0)
                printf("\nTimestep :  %d,  Timestage :  %d,   Time : %g\n",istep+1,j+1,disc.common.timestate.time);            
        
#ifdef TIMING    
            disc.common.timing[100] = 0.0; 
            disc.common.timing[101] = 0.0; 
#endif
        
            START_TIMING;

            // update source term             
            UpdateSource(disc.sol, solv.sys, disc.app, disc.driver_abi, disc.res, disc.common, backend);
            END_TIMING_DISC(100);    

            // solve the problem 
            this->SteadyProblem(out, backend);                             

            START_TIMING;
            // update solution 
            UpdateSolution(disc.sol, solv.sys, disc.common, backend);                     
            END_TIMING_DISC(101);

#ifdef TIMING         
            if (disc.common.mpiRank==0) {
                printf("Updating source term time: %g miliseconds\n", disc.common.timing[100]);   
                printf("Updating solution time: %g miliseconds\n\n", disc.common.timing[101]);                           
            }
#endif            
        }
        
        //compute time-average solution
        if (disc.common.outputparams.compudgavg == 1) {
            ArrayAXPBY(disc.sol.udgavg, disc.sol.udgavg, disc.sol.udg, one, one, disc.common.sizes.ndofudg1);            
            ArrayAddScalar(&disc.sol.udgavg[disc.common.sizes.ndofudg1], one, 1);
        }

        // save solutions into binary files
        //SaveSolutions(disc.sol, solv.sys, disc.common, backend);            
        this->SaveSolutions(backend);
        this->SaveQoI(backend);
        if (vis.savemode > 0) this->SaveParaview(backend); 
        this->SaveSolutionsOnBoundary(backend); 
        if (disc.common.components.nce>0)
            this->SaveOutputCG(backend);    
        
#ifdef TIMESTEP                          
        gettimeofday(&tv2, NULL);            
        if (disc.common.mpiRank==0)
            printf("\nExecution time (in millisec) for timestep %d:  %g\n", istep+1,
                (double)(tv2.tv_usec-tv1.tv_usec)/1000 + 
                (double)(tv2.tv_sec -tv1.tv_sec )*1000);
#endif                    
        // update time
        time = time + disc.common.dt[istep];                    
    }           
}

// Re-homed from CDiscretization (S4): the PTC monitor field is a solver-convergence artifact,
// not a discretization quantity. Uses the owned disc's structs to call the model MonitorDriver.
void CSolution::evalMonitor(dstype* output, dstype* udg, dstype* wdg, Int nc, Int backend)
{
    MonitorDriver(output, nc, disc.sol.xdg, udg, disc.sol.odg, wdg, disc.driver_abi,
                  disc.mesh, disc.master, disc.app, disc.sol, disc.tmp, disc.common, backend);
}

// Re-homed from CDiscretization (S4): computing the output field for I/O is an output concern.
// MPI-halo-exchanges the owned disc's udg across neighbors, then calls the model OutputDriver.
void CSolution::evalOutput(dstype* output, Int backend)
{
#ifdef  HAVE_MPI
    Int bsz = disc.common.grid.npe*disc.common.components.nc;
    Int n;

    /* copy some portion of u to buffsend */
    GetArrayAtIndex(disc.tmp.buffsend, disc.sol.udg, disc.mesh.elemsendudg, bsz*disc.common.nelemsend);

#ifdef HAVE_CUDA
    cudaDeviceSynchronize();
#endif
#ifdef HAVE_HIP
    hipDeviceSynchronize();
#endif

    /* non-blocking send */
    Int neighbor, nsend, psend = 0, request_counter = 0;
    for (n=0; n<disc.common.nnbsd; n++) {
        neighbor = disc.common.nbsd[n];
        nsend = disc.common.elemsendpts[n]*bsz;
        if (nsend>0) {
            MPI_Isend(&disc.tmp.buffsend[psend], nsend, MPI_DOUBLE, neighbor, 0,
                   EXASIM_COMM_LOCAL, &disc.common.requests[request_counter]);
            psend += nsend;
            request_counter += 1;
        }
    }

    /* non-blocking receive */
    Int nrecv, precv = 0;
    for (n=0; n<disc.common.nnbsd; n++) {
        neighbor = disc.common.nbsd[n];
        nrecv = disc.common.elemrecvpts[n]*bsz;
        if (nrecv>0) {
            MPI_Irecv(&disc.tmp.buffrecv[precv], nrecv, MPI_DOUBLE, neighbor, 0,
                   EXASIM_COMM_LOCAL, &disc.common.requests[request_counter]);
            precv += nrecv;
            request_counter += 1;
        }
    }

    /* wait until all send and receive operations are completely done */
    MPI_Waitall(request_counter, disc.common.requests, disc.common.statuses);

    /* copy buffrecv to udg */
    PutArrayAtIndex(disc.sol.udg, disc.tmp.buffrecv, disc.mesh.elemrecvudg, bsz*disc.common.nelemrecv);
#endif

    // compute the output field
    OutputDriver(output, disc.sol.xdg, disc.sol.udg, disc.sol.odg, disc.sol.wdg, disc.driver_abi,
                 disc.mesh, disc.master, disc.app, disc.sol, disc.tmp, disc.common, backend);
}

void CSolution::SteadyProblem_PTC(ofstream &out, Int backend) {

    // initial time
    double time = disc.common.timestate.time;           
    double monitor_diff, monitor_scale, delta_monitor;
    int N = disc.common.sizes.ndofuhat;
    int NLiters = disc.common.solverparams.nonlinearSolverMaxIter;
    double nrmr = 0;
    int conv_flag = 0;

    Int nc = disc.common.components.nc; // number of compoments of (u, q, p)
    Int ncu = disc.common.components.ncu;// number of compoments of (u)    
    Int ncs = disc.common.components.ncs;// number of compoments of (s)        
    Int npe = disc.common.grid.npe; // number of nodes on master element    
    //Int ne = common.meshsizes.ne1; // number of elements in this subdomain         
    Int ne2 = disc.common.meshsizes.ne2; // number of elements in this subdomain       
    //Int N = common.sizes.ndof1;
    Int N2 = npe*disc.common.components.ncw*ne2;  

    // time stepping with DIRK schemes
    for (Int istep=0; istep<disc.common.timeparams.tsteps-1; istep++)            
    {            
        disc.common.solverparams.nonlinearSolverMaxIter = 1;

        // current timestep        
        disc.common.timestate.currentstep = istep;

        // store previous solutions to calculate the source term        
        PreviousSolutions(disc.sol, solv.sys, disc.common, backend);
                            
        // compute the solution at the next step
        for (int j=0; j<disc.common.timeparams.tstages; j++) {     
            
            if (disc.common.mpiRank==0)
                printf("\nTimestep :  %d,  Timestage :  %d,   Time : %g\n",istep+1,j+1,time + disc.common.dt[istep]*disc.common.DIRKcoeff_t[j]);                                
                            
            // current timestage
            disc.common.timestate.currentstage = j;

            // current time
            disc.common.timestate.time = time + disc.common.dt[istep]*disc.common.DIRKcoeff_t[j];

            // update source term             
            UpdateSource(disc.sol, solv.sys, disc.app, disc.driver_abi, disc.res, disc.common, backend);
            
            // solve the problem 
            this->SteadyProblem(out, backend);                             

            // update solution 
            UpdateSolution(disc.sol, solv.sys, disc.app, disc.driver_abi, disc.res, disc.tmp, disc.common, backend);
            
            // TODO: input wprev
            evalMonitor(disc.tmp.tempn,  disc.sol.udg, disc.sol.wdg, disc.common.components.nc, backend);
            evalMonitor(disc.tmp.tempg,  solv.sys.udgprev, solv.sys.wprev, disc.common.components.ncu, backend);
            
            ArrayAXPBY(disc.tmp.tempn, disc.tmp.tempn, disc.tmp.tempg, 1.0, -1.0, disc.common.grid.npe*disc.common.components.ncm*disc.common.meshsizes.ne);            
            
            monitor_diff  = PNORM(disc.common.cublasHandle,  disc.common.grid.npe*disc.common.components.ncm*disc.common.meshsizes.ne,disc.tmp.tempn, backend);
            monitor_scale = PNORM(disc.common.cublasHandle,  disc.common.grid.npe*disc.common.components.ncm*disc.common.meshsizes.ne, disc.tmp.tempg, backend);

            delta_monitor = monitor_diff / monitor_scale;
            std::cout << "delta_monitor: " << delta_monitor << std::endl;

            if ((delta_monitor > 1.0 || solv.sys.alpha < 0.1))
            {
                std::cout << "Linesearch failed or large change in solution: reducing timestep" << std::endl;
                // Revert time step
                disc.common.timestate.time = disc.common.timestate.time - disc.common.dt[istep]*disc.common.DIRKcoeff_t[j];

                // Copy udg old to udg
                ArrayExtract(disc.sol.udg, solv.sys.udgprev, npe, ncu, ne2, 0, npe, 0, nc, 0, ne2);     

                // Compute a new UH
                GetFaceNodes(disc.sol.uh, disc.sol.udg, disc.mesh.f2e, disc.mesh.perm, disc.common.grid.npf, disc.common.components.ncu, disc.common.grid.npe, disc.common.components.nc, disc.common.meshsizes.nf);

                // Recompute gradient from udg old
                hdgGetQ<exasim::detail::AbiAdapter>(disc.sol.udg, disc.sol.uh, disc.sol, disc.res, disc.mesh, disc.tmp, disc.common, backend);

                // decrease timestep by 10
                std::cout << "Current time step: " << disc.common.dt[istep] << std::endl;
                disc.common.dt[istep+1] = disc.common.dt[istep] / 10;
                std::cout << "next time step: " << disc.common.dt[istep+1] << std::endl;
                if (disc.common.dt[istep+1] < 1e-8){
                    std::cout << "WARNING: PTC stalled" << std::endl;
                    istep = disc.common.timeparams.tsteps+1;
                }
            }
            else if (delta_monitor < 0.1 && solv.sys.alpha == 1.0)
            {
                if (solv.state.linearSolverIter < disc.common.solverparams.linearSolverMaxIter){
                    // increase timestep by 2
                    disc.common.dt[istep+1] = disc.common.dt[istep]*2;
                    // std::cout << "Doubling timestep: " << disc.common.dt[istep+1] << std::endl;
                }
                else{ //TODO: Probably overly conservative, consider turning off 
                    disc.common.dt[istep+1] = disc.common.dt[istep]*1;
                    std::cout << "Too many GMRES iterations, not increasing timestep: " << disc.common.dt[istep+1] << std::endl;
                }
                

                if (delta_monitor < 1e-3 && disc.common.dt[istep] > 1e-4) {
                    if (disc.common.runmode == 10) {
                        std::cout << "Evaluate steady residual..." << std::endl;
                        disc.common.timeparams.tdep=0;
    
                        if (disc.common.components.ncq > 0) hdgGetQ<exasim::detail::AbiAdapter>(disc.sol.udg, disc.sol.uh, disc.sol, disc.res, disc.mesh, disc.tmp, disc.common, backend);          
            
                        // compute the residual vector R = [Ru; Rh]
                        assembler.hdgAssembleResidual(solv.sys.b, backend);
                                
                        nrmr = PNORM(disc.common.cublasHandle, N, solv.sys.b, backend);       
                        nrmr += PNORM(disc.common.cublasHandle, disc.common.grid.npe*disc.common.components.ncu*disc.common.meshsizes.ne1, disc.res.Ru, backend); 
                        std::cout << " Steady residual = " << nrmr << std::endl;
    
                        // SaveSolutions(backend); 
                        if (nrmr < disc.common.solverparams.nonlinearSolverTol) {
                            conv_flag = 1;
                            istep = disc.common.timeparams.tsteps+10;
                            this->SaveSolutions(backend); 
                            this->SaveQoI(backend);
                            if (vis.savemode > 0) this->SaveParaview(backend); 
                            this->SaveSolutionsOnBoundary(backend); 
                        }
                        // istep = disc.common.timeparams.tsteps+1;
                        disc.common.timeparams.tdep=1;
                    }
                    if (disc.common.runmode == 11) { // Compute steady solve
                        std::cout << "Steady solve..." << std::endl;
                        disc.common.timeparams.tdep=0;
                        disc.common.solverparams.nonlinearSolverMaxIter = NLiters;
                        this->SolveProblem(out, backend);
                        istep = disc.common.timeparams.tsteps+10;
                    }
                    
                }    
            }
            else
            {
                // Do not change  timestep
                disc.common.dt[istep+1] = disc.common.dt[istep];
            }
        }
        // update time
        time = time + disc.common.dt[istep];                    
    }   
    if (conv_flag == 0) {                
        std::cout << "Warning: PTC reached max iterations without converging." << std::endl;
        // Save steady solution anyways
        disc.common.timeparams.tdep=0;
        this->SaveSolutions(backend); 
        this->SaveQoI(backend);
        if (vis.savemode > 0) this->SaveParaview(backend); 
        this->SaveSolutionsOnBoundary(backend); 
    }
}

void CSolution::SolveProblem(ofstream &out, Int backend) 
{          
    this->InitSolution(backend); 
        
    if (disc.common.timeparams.tdep==1) {        
        // solve time-dependent problems using DIRK
        this->DIRK(out, backend);            
    }
    else {
        // solve steady-state problems
        this->SteadyProblem(out, backend);        
                
        // save solutions into binary files            
        this->SaveSolutions(backend);    
        this->SaveQoI(backend);
        if (vis.savemode > 0) this->SaveParaview(backend); 
                
        this->SaveSolutionsOnBoundary(backend);         
        if (disc.common.components.nce>0)
            this->SaveOutputCG(backend);            
    }        
}

void CSolution::SaveSolutions(Int backend) 
{
    bool save = false;
    if (disc.common.timeparams.tdep==0) save = true;
    else 
        if (((disc.common.timestate.currentstep+1) % disc.common.outputparams.saveSolFreq) == 0) save = true;             

    if (save == true) {        
        if (disc.common.outputparams.saveSolOpt==0) {
            if (disc.common.spatialScheme > 0) {
                ArrayExtract(disc.res.Rq, disc.sol.udg, disc.common.grid.npe, disc.common.components.nc, disc.common.meshsizes.ne1, 0, disc.common.grid.npe, 0, disc.common.components.ncu, 0, disc.common.meshsizes.ne1);                                                  
                writearray(outsol, disc.res.Rq, disc.common.sizes.ndof1, backend);    
            }
            else
                writearray(outsol, solv.sys.u, disc.common.sizes.ndof1, backend);    
        }
        else
            writearray(outsol, disc.sol.udg, disc.common.sizes.ndofudg1, backend);    
        
        if (disc.common.components.ncw>0)
            writearray(outwdg, disc.sol.wdg, disc.common.sizes.ndofw1, backend);

        if (disc.common.spatialScheme==1)
            writearray(outuhat, disc.sol.uh, disc.common.sizes.ndofuhat, backend);
    }
    
    if (disc.common.timeparams.tdep==1) { 
        if (((disc.common.timestate.currentstep+1) % disc.common.outputparams.saveRestart) == 0)             
        {        
            string filename = disc.common.fileout + "udg_t" + NumberToString(disc.common.timestate.currentstep+disc.common.outputparams.timestepOffset+1) + "_np" + NumberToString(disc.common.mpiRank-disc.common.outputparams.fileoffset) + ".bin";     
            writearray2file(filename, disc.sol.udg, disc.common.sizes.ndofudg1, backend);

            if (disc.common.outputparams.compudgavg == 1) {
                string fn1 = disc.common.fileout + "solavg_np" + NumberToString(disc.common.mpiRank-disc.common.outputparams.fileoffset) + ".bin"; 
                writearray2file(fn1, disc.sol.udgavg, disc.common.sizes.ndofudg1+1, backend);
            }        
          
            if (disc.common.sizes.ndofbou > 0) {
                string fn0 = disc.common.fileout + "bouudgavg_np" + NumberToString(disc.common.mpiRank-disc.common.outputparams.fileoffset) + ".bin"; 
                writearray2file(fn0, disc.sol.bouudgavg, disc.common.sizes.ndofbou*disc.common.components.nc+1, backend);
                fn0 = disc.common.fileout + "bouuhavg_np" + NumberToString(disc.common.mpiRank-disc.common.outputparams.fileoffset) + ".bin"; 
                writearray2file(fn0, disc.sol.bouuhavg, disc.common.sizes.ndofbou*disc.common.components.ncu+1, backend);
                if (disc.common.components.ncw > 0) {
                    fn0 = disc.common.fileout + "bouwdgavg_np" + NumberToString(disc.common.mpiRank-disc.common.outputparams.fileoffset) + ".bin"; 
                    writearray2file(fn0, disc.sol.bouwdgavg, disc.common.sizes.ndofbou*disc.common.components.ncw+1, backend);
                }
            }        
          
            if (disc.common.components.ncw>0) {
                string fn = disc.common.fileout + "wdg_t" + NumberToString(disc.common.timestate.currentstep+disc.common.outputparams.timestepOffset+1) + "_np" + NumberToString(disc.common.mpiRank-disc.common.outputparams.fileoffset) + ".bin";                    
                writearray2file(fn, solv.sys.wtmp, disc.common.sizes.ndofw1, backend);
            }                        

            if (disc.common.spatialScheme==1) {
                string fn2 = disc.common.fileout + "_uhat_t" + NumberToString(disc.common.timestate.currentstep+disc.common.outputparams.timestepOffset+1) + "_np" + NumberToString(disc.common.mpiRank-disc.common.outputparams.fileoffset) + ".bin";                    
                writearray2file(fn2, disc.sol.uh, disc.common.sizes.ndofuhat, backend);        
            }
        }    
    }
    
   // if (disc.common.timeparams.tdep==1) { 
   //      if (((disc.common.timestate.currentstep+1) % disc.common.outputparams.saveSolFreq) == 0)             
   //      {        
   //          string filename = disc.common.fileout + "udg_t" + NumberToString(disc.common.timestate.currentstep+disc.common.outputparams.timestepOffset+1) + "_np" + NumberToString(disc.common.mpiRank-disc.common.outputparams.fileoffset) + ".bin";     
   //          if (disc.common.outputparams.saveSolOpt==0)
   //              writearray2file(filename, solv.sys.u, disc.common.sizes.ndof1, backend);
   //          else
   //              writearray2file(filename, disc.sol.udg, disc.common.sizes.ndofudg1, backend);
   // 
   //          if (disc.common.components.ncw>0) {
   //              string fn = disc.common.fileout + "_wdg_t" + NumberToString(disc.common.timestate.currentstep+disc.common.outputparams.timestepOffset+1) + "_np" + NumberToString(disc.common.mpiRank-disc.common.outputparams.fileoffset) + ".bin";                    
   //              writearray2file(fn, solv.sys.wtmp, disc.common.sizes.ndofw1, backend);
   //          }                        
   // 
   //          if (disc.common.outputparams.compudgavg == 1) {
   //              string fn1 = disc.common.fileout + "avg_np" + NumberToString(disc.common.mpiRank-disc.common.outputparams.fileoffset) + ".bin"; 
   //              writearray2file(fn1, disc.sol.udgavg, disc.common.sizes.ndofudg1+1, backend);
   //          }
   // 
   //          if (disc.common.spatialScheme==1) {
   //              string fn2 = disc.common.fileout + "_uhat_t" + NumberToString(disc.common.timestate.currentstep+disc.common.outputparams.timestepOffset+1) + "_np" + NumberToString(disc.common.mpiRank-disc.common.outputparams.fileoffset) + ".bin";                    
   //              writearray2file(fn2, disc.sol.uh, disc.common.sizes.ndofuhat, backend);        
   //          }
   //      }    
   // }
   // else {
   //      string filename = disc.common.fileout + "udg_np" + NumberToString(disc.common.mpiRank-disc.common.outputparams.fileoffset) + ".bin";                    
   //      if (disc.common.outputparams.saveSolOpt==0)
   //          writearray2file(filename, solv.sys.u, disc.common.sizes.ndof1, backend);
   //      else
   //          writearray2file(filename, disc.sol.udg, disc.common.sizes.ndofudg1, backend);       
   // 
   //      if (disc.common.components.ncw>0) {
   //          string fn = disc.common.fileout + "_wdg_np" + NumberToString(disc.common.mpiRank-disc.common.outputparams.fileoffset) + ".bin";                    
   //          writearray2file(fn, disc.sol.wdg, disc.common.sizes.ndofw1, backend);     
   //      }                
   // 
   //      if (disc.common.spatialScheme==1) {
   //          string filename1 = disc.common.fileout + "_uhat_np" + NumberToString(disc.common.mpiRank-disc.common.outputparams.fileoffset) + ".bin";                    
   //          writearray2file(filename1, disc.sol.uh, disc.common.sizes.ndofuhat, backend);        
   //      }
   // }    
}

void CSolution::ReadSolutions(Int backend) 
{
   if (disc.common.timeparams.tdep==1) { 
        if (((disc.common.timestate.currentstep+1) % disc.common.outputparams.saveRestart) == 0)             
        {        
            string filename = disc.common.fileout + "udg_t" + NumberToString(disc.common.timestate.currentstep+disc.common.outputparams.timestepOffset+1) + "_np" + NumberToString(disc.common.mpiRank-disc.common.outputparams.fileoffset) + ".bin";     
            // if (disc.common.outputparams.saveSolOpt==0) {
            //     readarrayfromfile(filename, &disc.res.Rq, disc.common.sizes.ndof1, backend);
            //     // insert u into udg
            //     ArrayInsert(disc.sol.udg, disc.res.Rq, disc.common.grid.npe, disc.common.components.nc, 
            //      disc.common.meshsizes.ne, 0, disc.common.grid.npe, 0, disc.common.components.ncu, 0, disc.common.meshsizes.ne1);  
            // }
            // else
                readarrayfromfile(filename, &disc.sol.udg, disc.common.sizes.ndofudg1, backend);        
            
            if (disc.common.components.ncw>0) {
                string fn = disc.common.fileout+"wdg_t" + NumberToString(disc.common.timestate.currentstep+disc.common.outputparams.timestepOffset+1) + "_np" + NumberToString(disc.common.mpiRank-disc.common.outputparams.fileoffset) + ".bin";                    
                readarrayfromfile(fn, &disc.sol.wdg, disc.common.sizes.ndofw1, backend);     
            }                      

            if (disc.common.spatialScheme==1) {
                string fn2 = disc.common.fileout + "_uhat_t" + NumberToString(disc.common.timestate.currentstep+disc.common.outputparams.timestepOffset+1) + "_np" + NumberToString(disc.common.mpiRank-disc.common.outputparams.fileoffset) + ".bin";                    
                readarrayfromfile(fn2, &disc.sol.uh, disc.common.sizes.ndofuhat, backend);        
            }              
        }                                
   }
   else {
        string filename = disc.common.fileout + "udg_np" + NumberToString(disc.common.mpiRank-disc.common.outputparams.fileoffset) + ".bin";                    
        if (disc.common.outputparams.saveSolOpt==0) {
            readarrayfromfile(filename, &solv.sys.u, disc.common.sizes.ndof1, backend);
            // insert u into udg
            ArrayInsert(disc.sol.udg, solv.sys.u, disc.common.grid.npe, disc.common.components.nc, 
             disc.common.meshsizes.ne, 0, disc.common.grid.npe, 0, disc.common.components.ncu, 0, disc.common.meshsizes.ne1);              
        }
        else
            readarrayfromfile(filename, &disc.sol.udg, disc.common.sizes.ndofudg1, backend, 3);      
             
        if (disc.common.components.ncw>0) {
            string fn = disc.common.fileout + "wdg_np" + NumberToString(disc.common.mpiRank-disc.common.outputparams.fileoffset) + ".bin";                    
            readarrayfromfile(fn, &disc.sol.wdg, disc.common.sizes.ndofw1, backend, 3);     
        }                

        if (disc.common.spatialScheme==1) {
            string fn = disc.common.fileout + "uhat_np" + NumberToString(disc.common.mpiRank-disc.common.outputparams.fileoffset) + ".bin";                    
            readarrayfromfile(fn, &disc.sol.uh, disc.common.sizes.ndofuhat, backend, 3);        
        }                                    
   }    
}

void CSolution::GetSolutions(Int step, Int backend)
{
    if (step < 0)
        error("GetSolutions: step must be nonnegative");

    const Int rank = disc.common.mpiRank - disc.common.outputparams.fileoffset;
    const Int headerSize = 3;
    string filename = disc.common.fileout + "udg_np" + NumberToString(rank) + ".bin";

    if (disc.common.outputparams.saveSolOpt == 0) {
        const Int skip = headerSize + step * disc.common.sizes.ndof1;
        readarrayfromfile(filename, &disc.res.Rq, disc.common.sizes.ndof1, backend, skip);
        ArrayInsert(disc.sol.udg, disc.res.Rq, disc.common.grid.npe, disc.common.components.nc,
                    disc.common.meshsizes.ne, 0, disc.common.grid.npe, 0, disc.common.components.ncu,
                    0, disc.common.meshsizes.ne1);
    }
    else {
        const Int skip = headerSize + step * disc.common.sizes.ndofudg1;
        readarrayfromfile(filename, &disc.sol.udg, disc.common.sizes.ndofudg1, backend, skip);
    }

    if (disc.common.components.ncw > 0) {
        string fn = disc.common.fileout + "wdg_np" + NumberToString(rank) + ".bin";
        const Int skip = headerSize + step * disc.common.sizes.ndofw1;
        readarrayfromfile(fn, &disc.sol.wdg, disc.common.sizes.ndofw1, backend, skip);
    }

    if (disc.common.spatialScheme == 1) {
        string fn = disc.common.fileout + "uhat_np" + NumberToString(rank) + ".bin";
        const Int skip = headerSize + step * disc.common.sizes.ndofuhat;
        readarrayfromfile(fn, &disc.sol.uh, disc.common.sizes.ndofuhat, backend, skip);
    }

    if ((disc.common.outputparams.saveSolOpt == 0) && (disc.common.components.ncq > 0))
        residual.evalQ(backend);
}
 
void CSolution::SaveParaview(Int backend, std::string fname_modifier, bool force_tdep_write) 
{
    // Decide whether we should write a file on this step
    bool writeSolution = false;
    
    if (disc.common.timeparams.tdep == 1) {
       if (disc.common.timestate.currentstep==0 && disc.common.mpiRank==0) {
          string ext = (disc.common.mpiProcs==1) ? "vtu" : "pvtu";                                  
          vis.pvdwrite_series(disc.common.fileout + "vis", disc.common.dt, disc.common.timeparams.tsteps, disc.common.outputparams.saveSolFreq, ext);                          
       }
        
        // Time-dependent: only write every 'saveSolFreq' steps
        writeSolution = ((disc.common.timestate.currentstep + 1) % disc.common.outputparams.saveSolFreq) == 0;
        writeSolution = writeSolution || force_tdep_write;
    } else {
        // Steady / not time-dependent: always write
        writeSolution = true;
    }

   if (writeSolution) { 
       int nc = disc.common.components.nc;  
       int ncx = disc.common.components.ncx;   
       int nco = disc.common.components.nco;  
       int ncw = disc.common.components.ncw;  
       int nsca = disc.common.qoiparams.nsca; 
       int nvec = disc.common.qoiparams.nvec;  
       int nten = disc.common.qoiparams.nten;     
       int npe  = disc.common.grid.npe;     
       int ne   = disc.common.meshsizes.ne1;      
       int ndg  = npe * ne;
       int ncg  = vis.npoints;
    
       dstype* udg = disc.res.Rq;  
       dstype* wdg = disc.res.Ru;
       int nvis = max(max(nsca, 3*nvec), vis.ntc*nten);
       int szvis = npe*(ncx+nco+nvis)*ne;
       bool ownsTempn = false;
       dstype* tempn = disc.tmp.tempn;
       if (disc.tmp.sztempn + disc.tmp.sztempg < szvis) {
         TemplateMalloc(&tempn, szvis, backend);
         ownsTempn = true;
       }
       dstype* xdg = &tempn[0];         
       dstype* vdg = &tempn[npe*ncx*ne];           
       dstype* f = &tempn[npe*(ncx+nco)*ne];
    
       GetElemNodes(xdg, disc.sol.xdg, npe, ncx, 0, ncx, 0, ne);
       GetElemNodes(udg, disc.sol.udg, npe, nc, 0, nc, 0, ne);
       if (nco > 0) GetElemNodes(vdg, disc.sol.odg, npe, nco, 0, nco, 0, ne);
       if (ncw > 0) GetElemNodes(wdg, disc.sol.wdg, npe, ncw, 0, ncw, 0, ne);
    
       if (nsca > 0) {        
            VisScalarsDriver(f, xdg, udg, vdg, wdg, disc.driver_abi, disc.mesh, disc.master, disc.app, disc.sol, disc.tmp, disc.common, npe, 0, ne, backend);                                 
            VisDG2CG(vis.scafields, f, disc.mesh.cgent2dgent, disc.mesh.colent2elem, disc.mesh.rowent2elem, ne, ncg, ndg, 1, 1, nsca);
       }    
       if (nvec > 0) {        
            VisVectorsDriver(f, xdg, udg, vdg, wdg, disc.driver_abi, disc.mesh, disc.master, disc.app, disc.sol, disc.tmp, disc.common, npe, 0, ne, backend);                                 
            VisDG2CG(vis.vecfields, f, disc.mesh.cgent2dgent, disc.mesh.colent2elem, disc.mesh.rowent2elem, ne, ncg, ndg, 3, ncx, nvec);
       }
       if (nten > 0) {        
            VisTensorsDriver(f, xdg, udg, vdg, wdg, disc.driver_abi, disc.mesh, disc.master, disc.app, disc.sol, disc.tmp, disc.common, npe, 0, ne, backend);                                 
            VisDG2CG(vis.tenfields, f, disc.mesh.cgent2dgent, disc.mesh.colent2elem, disc.mesh.rowent2elem, ne, ncg, ndg, vis.ntc, vis.ntc, nten);
       }

       string baseName = disc.common.fileout + "vis" + fname_modifier;
       // A forced write (SaveParaviewStep / crash dump) is an explicit time-series
       // frame, so include the step index even when the run is not marked tdep
       // (e.g. a steady fluid re-solved each outer coupling step). Without this the
       // parallel pvtu/vtu names omit the step and every frame overwrites the last.
       if (disc.common.timeparams.tdep == 1 || force_tdep_write) {
           std::ostringstream ss;
           ss << std::setw(6) << std::setfill('0') << disc.common.timestate.currentstep+disc.common.outputparams.timestepOffset+1;
           baseName = baseName + "_" + ss.str();
       }

       if (disc.common.mpiProcs==1)
            vis.vtuwrite(baseName, vis.scafields, vis.vecfields, vis.tenfields);
       else
            vis.vtuwrite_parallel(baseName, disc.common.mpiRank, disc.common.mpiProcs, vis.scafields, vis.vecfields, vis.tenfields);

       if (ownsTempn)
         TemplateFree(tempn, backend);
   }
}

void CSolution::SaveQoI(Int backend) 
{
    if (disc.common.qoiparams.nvqoi > 0) qoiElement<exasim::detail::AbiAdapter>(disc.sol, disc.res, disc.app, disc.master, disc.mesh, disc.tmp, disc.common);
    if (disc.common.qoiparams.nsurf > 0) qoiFace<exasim::detail::AbiAdapter>(disc.sol, disc.res, disc.app, disc.master, disc.mesh, disc.tmp, disc.common);

    if (disc.common.mpiRank==0 && (disc.common.qoiparams.nvqoi > 0 || disc.common.qoiparams.nsurf > 0)) {
        writeQoIHeaderOnce(outqoi, disc.common.qoiparams);
        if (disc.common.timeparams.tdep==1)
            outqoi << std::setw(16) << std::scientific << std::setprecision(6) << disc.common.timestate.time;
        else outqoi << std::setw(16) << std::scientific << std::setprecision(6) << 0.0;
        writeQoIRow(outqoi, disc.common.qoiparams);
        outqoi << "\n";
    }
}

void CSolution::SaveOutputCG(Int backend) 
{
   if (disc.common.timeparams.tdep==1) { 
        if (((disc.common.timestate.currentstep+1) % disc.common.outputparams.saveSolFreq) == 0)             
        {                    
            string filename1 = disc.common.fileout + "_outputCG_t" + NumberToString(disc.common.timestate.currentstep+disc.common.outputparams.timestepOffset+1) + "_np" + NumberToString(disc.common.mpiRank-disc.common.outputparams.fileoffset) + ".bin";     
            evalOutput(disc.res.Rq, backend);
            disc.DG2CG(disc.res.Rq, disc.res.Rq, disc.tmp.tempn, disc.common.components.nce, 
                     disc.common.components.nce, disc.common.components.nce, backend);
            writearray2file(filename1, disc.res.Rq, disc.common.sizes.ndofedg1, backend);                   
//             disc.DG2CG3(solv.sys.v, solv.sys.v, solv.sys.x, disc.common.components.nce, 
//                  disc.common.components.nce, disc.common.components.nce, backend);
//             writearray2file(filename1, solv.sys.v, disc.common.sizes.ndofucg, backend);               
        }                                
   }
   else {
        string filename1 = disc.common.fileout + "_outputCG_np" + NumberToString(disc.common.mpiRank-disc.common.outputparams.fileoffset) + ".bin";                            
        evalOutput(disc.res.Rq, backend);
        disc.DG2CG(disc.res.Rq, disc.res.Rq, disc.tmp.tempn, disc.common.components.nce, 
                 disc.common.components.nce, disc.common.components.nce, backend);
        writearray2file(filename1, disc.res.Rq, disc.common.sizes.ndofedg1, backend);               
//         disc.DG2CG3(solv.sys.v, solv.sys.v, solv.sys.x, disc.common.components.nce, 
//                  disc.common.components.nce, disc.common.components.nce, backend);        
//         writearray2file(filename1, solv.sys.v, disc.common.sizes.ndofucg, backend);               
   }    
}        

void CSolution::SaveSolutionsOnBoundary(Int backend) 
{   
    if ( disc.common.outputparams.saveSolBouFreq>0 ) {
        if (((disc.common.timestate.currentstep+1) % disc.common.outputparams.saveSolBouFreq) == 0)             
        {        
            for (Int j=0; j<disc.common.meshsizes.nbf; j++) {
                Int f1 = disc.common.fblks[3*j]-1;
                Int f2 = disc.common.fblks[3*j+1];    
                Int ib = disc.common.fblks[3*j+2];            
                if (ib == disc.common.qoiparams.ibs) {     
                    Int npf = disc.common.grid.npf; // number of nodes on master face      
                    Int npe = disc.common.grid.npe; // number of nodes on master face      
                    Int nf = f2-f1;
                    Int nn = npf*nf; 
                    Int nc = disc.common.components.nc; // number of compoments of (u, q, p)            
                    Int ncu = disc.common.components.ncu;
                    Int ncw = disc.common.components.ncw;
                    GetArrayAtIndex(disc.tmp.tempn, disc.sol.udg, &disc.mesh.findudg1[npf*nc*f1], nn*nc);
                    writearray(outbouudg, disc.tmp.tempn, nn*nc, backend);                                        
                    if (disc.common.spatialScheme==1)
                      GetFaceNodesHDG(disc.tmp.tempn, disc.sol.uh, npf, ncu, 0, ncu, f1, f2);
                    else
                      GetElemNodes(disc.tmp.tempn, disc.sol.uh, npf, ncu, 0, ncu, f1, f2);
                    writearray(outbouuhat, disc.tmp.tempn, nn*ncu, backend);
                    if (ncw>0) {
                        GetFaceNodes(disc.tmp.tempn, disc.sol.wdg, disc.mesh.facecon, npf, ncw, npe, ncw, f1, f2, 1);      
                        writearray(outbouwdg, disc.tmp.tempn, nn*ncw, backend);
                    }
                }
            }          
        }                                
    }
}

void CSolution::SaveNodesOnBoundary(Int backend) 
{   
    if ( disc.common.outputparams.saveSolBouFreq>0 ) {
        for (Int j=0; j<disc.common.meshsizes.nbf; j++) {
            Int f1 = disc.common.fblks[3*j]-1;
            Int f2 = disc.common.fblks[3*j+1];    
            Int ib = disc.common.fblks[3*j+2];            
            if (ib == disc.common.qoiparams.ibs) {     
                Int nd = disc.common.grid.nd; 
                Int npf = disc.common.grid.npf; // number of nodes on master face      
                Int nf = f2-f1;
                Int nn = npf*nf; 
                Int ncx = disc.common.components.ncx; // number of compoments of (u, q, p)                            
                GetArrayAtIndex(disc.tmp.tempn, disc.sol.xdg, &disc.mesh.findxdg1[npf*ncx*f1], nn*ncx);                
                writearray(outbouxdg, disc.tmp.tempn, nn*ncx, backend);

                Int n1 = nn*ncx;                           // nlg
                Int n2 = nn*(ncx+nd);                      // jac
                Int n3 = nn*(ncx+nd+1);                    // Jg
                if (nd==1) {
                    FaceGeom1D(&disc.tmp.tempn[n2], &disc.tmp.tempn[n1], &disc.tmp.tempn[n3], nn);    
                    FixNormal1D(&disc.tmp.tempn[n1], &disc.mesh.facecon[2*f1], nn);    
                }
                else if (nd==2){
                    Node2Gauss(disc.common.cublasHandle, &disc.tmp.tempn[n3], disc.tmp.tempn, &disc.master.shapfnt[npf*npf], npf, npf, nf*nd, backend);                
                    FaceGeom2D(&disc.tmp.tempn[n2], &disc.tmp.tempn[n1], &disc.tmp.tempn[n3], nn);
                }
                else if (nd==3) {
                    Node2Gauss(disc.common.cublasHandle, &disc.tmp.tempn[n3], disc.tmp.tempn, &disc.master.shapfnt[npf*npf], npf, npf, nf*nd, backend);                     
                    Node2Gauss(disc.common.cublasHandle, &disc.tmp.tempn[n3+nn*nd], disc.tmp.tempn, &disc.master.shapfnt[2*npf*npf], npf, npf, nf*nd, backend);                
                    FaceGeom3D(&disc.tmp.tempn[n2], &disc.tmp.tempn[n1], &disc.tmp.tempn[n3], nn);
                }
                writearray(outboundg, &disc.tmp.tempn[n1], nn*nd, backend);
            }
        }
        if (outbouxdg.is_open()) { outbouxdg.close(); }
        if (outboundg.is_open()) { outboundg.close(); }
    }
}

// void CSolution::SaveSolutionsOnBoundary(Int backend) 
// {   
//     if ( disc.common.outputparams.saveSolBouFreq>0 ) {
//         if (((disc.common.timestate.currentstep+1) % disc.common.outputparams.saveSolBouFreq) == 0)             
//         {        
//             for (Int j=0; j<disc.common.meshsizes.nbf; j++) {
//                 Int f1 = disc.common.fblks[3*j]-1;
//                 Int f2 = disc.common.fblks[3*j+1];    
//                 Int ib = disc.common.fblks[3*j+2];            
//                 if (ib == disc.common.qoiparams.ibs) {     
//                     Int npf = disc.common.grid.npf; // number of nodes on master face      
//                     Int nf = f2-f1;
//                     Int nn = npf*nf; 
//                     Int nc = disc.common.components.nc; // number of compoments of (u, q, p)            
//                     GetArrayAtIndex(disc.tmp.tempn, disc.sol.udg, &disc.mesh.findudg1[npf*nc*f1], nn*nc);
//                     string filename = disc.common.fileout + "bou_t" + NumberToString(disc.common.timestate.currentstep+disc.common.outputparams.timestepOffset+1) + "_np" + NumberToString(disc.common.mpiRank-disc.common.outputparams.fileoffset) + ".bin";     
//                     writearray2file(filename, disc.tmp.tempn, nn*nc, backend);            
//                 }
//             }                                               
//         }                                
//     }
// }
// 
// void CSolution::SaveNodesOnBoundary(Int backend) 
// {   
//     if ( disc.common.outputparams.saveSolBouFreq>0 ) {
//         for (Int j=0; j<disc.common.meshsizes.nbf; j++) {
//             Int f1 = disc.common.fblks[3*j]-1;
//             Int f2 = disc.common.fblks[3*j+1];    
//             Int ib = disc.common.fblks[3*j+2];            
//             if (ib == disc.common.qoiparams.ibs) {     
//                 Int npf = disc.common.grid.npf; // number of nodes on master face      
//                 Int nf = f2-f1;
//                 Int nn = npf*nf; 
//                 Int ncx = disc.common.components.ncx; // number of compoments of (u, q, p)                            
//                 GetArrayAtIndex(disc.tmp.tempn, disc.sol.xdg, &disc.mesh.findxdg1[npf*ncx*f1], nn*ncx);
//                 string filename = disc.common.fileout + "node_np" + NumberToString(disc.common.mpiRank-disc.common.outputparams.fileoffset) + ".bin";     
//                 writearray2file(filename, disc.tmp.tempn, nn*ncx, backend);            
//             }
//         }                                                                   
//     }
// }

#endif        
