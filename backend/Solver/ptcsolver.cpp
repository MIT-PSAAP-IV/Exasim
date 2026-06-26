/*
    ptcsolver.cpp

    This file contains implementations for nonlinear and linear solvers using Pseudo-Time Continuation (PTC) and Newton's method, 
    with support for reduced basis (RB) preconditioning and timing analysis. The solvers are designed for high-performance 
    scientific computing, potentially leveraging CUDA and MPI for parallelism.

    Functions:

    - int CSolver::linearSolve(CAssembler& assembler, CDiscretization& disc, CPreconditioner& prec, ofstream &out, Int it, Int backend)
        Solves the linear system arising in each nonlinear iteration. Evaluates the residual, constructs the preconditioner, 
        applies GMRES, and manages timing and logging. Handles reduced basis updates and checks for convergence.

    - void CSolver::updateRB(CDiscretization& disc, CPreconditioner& prec, Int backend)
        Updates the reduced basis vectors used for preconditioning if the current solution increment is significant.

    - Int PTCsolver(sysstruct &sys,  CDiscretization& disc, CPreconditioner& prec, ofstream &out, Int backend)
        Implements the Pseudo-Time Continuation (PTC) nonlinear solver. Iteratively solves the nonlinear system R(u) = 0 
        using the linear solver, updates the solution, checks for divergence, logs residual norms, and manages reduced basis updates.

    - void CSolver::updateRB(CDiscretization& disc, CPreconditioner& prec, Int N, Int backend)
        Alternate version of UpdateRB with explicit dimension argument. Updates the reduced basis vectors for preconditioning.

    - void CSolver::linearSolve(CDiscretization& disc, CPreconditioner& prec, ofstream &out, Int N, Int spatialScheme, Int it, Int backend)
        Overloaded LinearSolver supporting different spatial discretization schemes. Assembles the linear system, constructs 
        the preconditioner, applies GMRES, and logs timing information.

    - Int NonlinearSolver(sysstruct &sys,  CDiscretization& disc, CPreconditioner& prec, ofstream &out, Int N, Int spatialScheme, Int backend)
        Implements Newton's method for nonlinear systems. Iteratively solves the linearized system, applies solution updates 
        with damping, computes residuals, manages reduced basis updates, and checks for convergence.

    Key Concepts:
    - Residual evaluation and norm computation for convergence checks.
    - Reduced basis (RB) preconditioning for accelerating linear solves.
    - GMRES iterative solver for linear systems.
    - Timing and performance analysis for solver components.
    - Support for different spatial discretization schemes (e.g., HDG).
    - Logging and debugging support for solution and residual vectors.
    - Handling of divergence and NaN residuals with output and termination.

    Dependencies:
    - CUDA and MPI support (conditional compilation).
    - External array and linear algebra utilities (e.g., ArrayAXPY, ArrayCopy, PNORM).
    - Data structures: sysstruct, CDiscretization, CPreconditioner.

*/
#ifndef __PTCSOLVER
#define __PTCSOLVER

int CSolver::linearSolve(CAssembler& assembler, CDiscretization& disc, CPreconditioner& prec, ofstream &out, Int it, Int backend)
{    
        
#ifdef TIMING    
    auto begin = chrono::high_resolution_clock::now();      
#endif
   
//#ifdef HAVE_CUDA
//    cudaDeviceSynchronize();
//#endif
     
    // evaluate the residual R(u) and set it to sys.b
    disc.evalResidual(sys.b, sys.u, backend);

    int N = disc.common.sizes.ndof1;

    // residual norm
    dstype oldnrm = PNORM(disc.common.cublasHandle, N, sys.b, backend); 
                
    if (disc.common.mpiRank==0 && disc.common.outputparams.saveResNorm==1 && it==1) {
        disc.common.timing[120] = disc.common.timestate.currentstep + 1.0;
        disc.common.timing[121] = disc.common.timestate.currentstage + 1.0;
        disc.common.timing[122] = 0.0; 
        disc.common.timing[123] = oldnrm;        
        writearray(out, &disc.common.timing[120], 4);    
    }
    
    // construct the preconditioner
    if (state.RBcurrentdim>0) {
        //prec.ConstructPreconditioner(sys, disc, backend);                  
        prec.ComputeInitialGuessAndPreconditioner(sys, state, assembler, disc, backend); 
        
        // v = u + x 
        //int N = disc.common.sizes.ndof1;
        ArrayAXPBY(disc.common.cublasHandle, sys.v, sys.u, sys.x, one, one, N, backend);  
        disc.evalResidual(sys.r, sys.v, backend);  
        dstype nrmr = PNORM(disc.common.cublasHandle, N, sys.r, backend);
                
        if (nrmr>1.05*oldnrm) {
            //ArraySetValue(sys.x, zero, N, backend);
            ArrayMultiplyScalar(disc.common.cublasHandle, sys.x, zero, N, backend);                       
            // reset the reduced basis
            state.RBremovedind = 0;
            state.RBcurrentdim = 0;
            //ArrayCopy(&prec.precond.W[state.RBremovedind*N], sys.x, N, backend);         
            //state.RBcurrentdim = 1;
            //state.RBremovedind = 1;            
        }

        if (nrmr < disc.common.solverparams.nonlinearSolverTol) 
            return 1;               
    }    
    else {
        //ArraySetValue(sys.x, zero, disc.common.sizes.ndof1, backend);
        ArrayMultiplyScalar(disc.common.cublasHandle, sys.x, zero, N, backend);   
    }
    
//   ArraySetValue(sys.x, zero, disc.common.sizes.ndof1, backend);
    
#ifdef TIMING        
    auto end = chrono::high_resolution_clock::now();
    disc.common.timing[4] = chrono::duration_cast<chrono::nanoseconds>(end-begin).count()/1e6;        
#endif
    
    // set Wcurrentdim
    //state.Wcurrentdim = state.RBcurrentdim;
    
#ifdef TIMING    
    begin = chrono::high_resolution_clock::now();      
#endif    
    state.linearSolverIter = gmres(assembler, disc, prec, backend);  
    if (disc.common.mpiRank==0)             
        printf("GMRES converges to the tolerance %g within % d iterations and %d RB dimensions\n",disc.common.solverparams.linearSolverTol,state.linearSolverIter,state.RBcurrentdim);        
    
#ifdef TIMING        
    end = chrono::high_resolution_clock::now();
    disc.common.timing[5] = chrono::duration_cast<chrono::nanoseconds>(end-begin).count()/1e6;        
        
    if (disc.common.mpiRank==0) {
    printf("--------- Linear Solver Analysis -------\n");    
    printf("Constructing preconditioner time: %g miliseconds\n", disc.common.timing[4]);
    printf("Matrix-vector product time: %g miliseconds\n", disc.common.timing[0]);
    printf("Orthgonalization time: %g miliseconds\n", disc.common.timing[1]);
    printf("Applying preconditioner time: %g miliseconds\n", disc.common.timing[2]);
    printf("Solution update time: %g miliseconds\n", disc.common.timing[3]);
    printf("Linear solver time: %g miliseconds\n\n", disc.common.timing[5]);
        
    printf("--------- Residual Calculation Analysis -------\n");
    printf("Copy to buffsend time: %g miliseconds\n", disc.common.timing[13]);
    printf("Non-blocking send/receive time: %g miliseconds\n", disc.common.timing[6]);
    printf("GetUhat time: %g miliseconds\n", disc.common.timing[7]);
    printf("GetQ<exasim::detail::AbiAdapter>(interior elements) time: %g miliseconds\n", disc.common.timing[8]);
    printf("RuElem<exasim::detail::AbiAdapter>(interior elements) time: %g miliseconds\n", disc.common.timing[9]);
    printf("MPI_WAITALL time: %g miliseconds\n", disc.common.timing[10]);
    printf("RuElem and GetQ<exasim::detail::AbiAdapter>(exterior elements) time: %g miliseconds\n", disc.common.timing[11]);
    printf("RuFace time: %g miliseconds\n", disc.common.timing[12]);    
    printf("Copy from buffrecv time: %g miliseconds\n\n", disc.common.timing[14]);        
    }
#endif
    
    return 0;    
}

void CSolver::updateRB(CDiscretization& disc, CPreconditioner& prec, Int backend)
{
    Int N = disc.common.sizes.ndof1;
                    
    dstype nrmr = PNORM(disc.common.cublasHandle, N, sys.x, backend);
    if (nrmr>zero) {
      // update the reduced basis        
      //ArrayCopy(&prec.precond.W[state.RBremovedind*N], sys.x, N, backend);  
      ArrayCopy(disc.common.cublasHandle, &prec.precond.W[state.RBremovedind*N], sys.x, N, backend);  

      // update the current dimension of the RB dimension
      if (state.RBcurrentdim<disc.common.solverparams.RBdim) 
          state.RBcurrentdim += 1;                    

      // update the position of the RB vector to be replaced  
      state.RBremovedind += 1;
      if (state.RBremovedind==disc.common.solverparams.RBdim) 
          state.RBremovedind = 0;                
    }
}

void CSolver::updateRB(CDiscretization& disc, CPreconditioner& prec, Int N, Int backend)
{                       
    dstype nrmr = PNORM(disc.common.cublasHandle, N, sys.x, backend);
    if (nrmr>zero) {
      // update the reduced basis        
      ArrayCopy(&prec.precond.W[state.RBremovedind*N], sys.x, N);  
      //ArrayCopy(disc.common.cublasHandle, &prec.precond.W[state.RBremovedind*N], sys.x, N, backend);  

      // update the current dimension of the RB dimension
      if (state.RBcurrentdim<disc.common.solverparams.RBdim) 
          state.RBcurrentdim += 1;                    

      // update the position of the RB vector to be replaced  
      state.RBremovedind += 1;
      if (state.RBremovedind==disc.common.solverparams.RBdim) 
          state.RBremovedind = 0;                
    }
}

void CSolver::linearSolve(CAssembler& assembler, CDiscretization& disc, CPreconditioner& prec, ofstream &out, Int N, Int spatialScheme, Int it, Int backend)
{
    // evaluate the residual R(u) and set it to sys.b
    if (spatialScheme==0) {
      disc.evalResidual(sys.b, sys.u, backend);
    }
    else if (spatialScheme==1) {
      auto begin = chrono::high_resolution_clock::now();

      assembler.hdgAssembleLinearSystem(sys.b, backend);
              
      auto end = chrono::high_resolution_clock::now();
      double t1 = chrono::duration_cast<chrono::nanoseconds>(end-begin).count()/1e6;      

      if (disc.common.mpiRank==0) printf("hdgAssembleLinearSystem time: %g miliseconds\n", t1);
      
      if (disc.common.outputparams.debugMode==1) {
        int n = disc.common.grid.npe*disc.common.components.ncu;
        int m = disc.common.grid.npf*disc.common.meshsizes.nfe*disc.common.components.ncu;
        int ne = disc.common.meshsizes.ne1;
        writearray2file(disc.common.fileout + NumberToString(it+1) + "newton_AE.bin", disc.res.H, m*m*ne, backend);
        writearray2file(disc.common.fileout + NumberToString(it+1) + "newton_FE.bin", disc.res.Rh, m*ne, backend);
        writearray2file(disc.common.fileout + NumberToString(it+1) + "newton_DUDG.bin", disc.res.Ru, n*ne, backend);
        writearray2file(disc.common.fileout + NumberToString(it+1) + "newton_DUDG_DUH.bin", disc.res.F, n*m*ne, backend);
      }
    }
    
    // construct the preconditioner
    if (state.RBcurrentdim>0) {
        prec.ComputeInitialGuessAndPreconditioner(sys, state, assembler, disc, N, spatialScheme, backend);         
    }    
    else {
        ArraySetValue(sys.x, zero, N);     
    }
            
    auto begin = chrono::high_resolution_clock::now();   
        
    state.linearSolverIter = gmres(assembler, disc, prec, N, spatialScheme, backend);  

    auto end = chrono::high_resolution_clock::now();
    double t1 = chrono::duration_cast<chrono::nanoseconds>(end-begin).count()/1e6;        
    
    if (disc.common.mpiRank==0)  {
        printf("GMRES time: %g miliseconds\n", t1);
        printf("GMRES(%d) converges to the tolerance %g within % d iterations and %d RB dimensions\n",disc.common.solverparams.gmresRestart,disc.common.solverparams.linearSolverTol,state.linearSolverIter,state.RBcurrentdim);                    
    }
}

#endif


