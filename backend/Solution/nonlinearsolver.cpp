/*
    CNonlinearSolver -- the PTC (LDG) and Newton (HDG) nonlinear iterations, extracted from
    CSolution (the "nonlinear solver" half of the CSolution split). Method bodies are unchanged;
    the referenced members keep the same names (disc/residual/assembler/prec/solv/writer), so only
    the class qualifier changed.
*/
#ifndef __NONLINEARSOLVER
#define __NONLINEARSOLVER

#include "nonlinearsolver.h"

template <class M>
Int CNonlinearSolver<M>::PTCsolver(ofstream &out, Int backend)       
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
            writer.crashDump(backend);
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

template <class M>
Int CNonlinearSolver<M>::NewtonSolver(ofstream &out, Int N, Int spatialScheme, Int backend)       
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

      if (disc.common.components.ncq > 0) hdgGetQ(disc.sol.udg, disc.sol.uh, disc.sol, disc.res, disc.mesh, disc.tmp, disc.common, backend);                
      if (disc.common.components.ncw > 0) GetW<M>(disc.sol.wdg, disc.sol, disc.tmp, disc.app, disc.common, backend);
      
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
                        writer.crashDump(backend);
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
          hdgGetDUDG(disc.res.Ru, disc.res.F, solv.sys.x, disc.res.Rq, disc.mesh, disc.common, backend);          
          ArrayCopy(solv.sys.v, disc.res.Ru, disc.common.grid.npe*disc.common.components.ncu*disc.common.meshsizes.ne1);
          UpdateUDG(disc.sol.udg, disc.res.Ru, solv.sys.alpha, disc.common.grid.npe, disc.common.components.nc, disc.common.meshsizes.ne1, 0, disc.common.grid.npe, 0, disc.common.components.ncu, 0, disc.common.meshsizes.ne1);                    
                    
          if (disc.common.outputparams.debugMode==1) {
            writearray2file(disc.common.fileout + NumberToString(it+1) + "newton_x.bin", solv.sys.x, N, backend);
            writearray2file(disc.common.fileout + NumberToString(it+1) + "newton_u.bin", solv.sys.u, N, backend);
            writearray2file(disc.common.fileout + NumberToString(it+1) + "newton_uh.bin", disc.sol.uh, N, backend);
            writearray2file(disc.common.fileout + NumberToString(it+1) + "newton_udg.bin", disc.sol.udg, disc.common.grid.npe*disc.common.components.nc*disc.common.meshsizes.ne1, backend);
            error("stop for debugging...");
          }          
                    
          if (disc.common.components.ncq > 0) hdgGetQ(disc.sol.udg, disc.sol.uh, disc.sol, disc.res, disc.mesh, disc.tmp, disc.common, backend);          
          if (disc.common.components.ncw > 0) GetW<M>(disc.sol.wdg, disc.sol, disc.tmp, disc.app, disc.common, backend);
                              
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
                            writer.crashDump(backend);
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
            if (disc.common.components.ncq > 0) hdgGetQ(disc.sol.udg, disc.sol.uh, disc.sol, disc.res, disc.mesh, disc.tmp, disc.common, backend);          
            if (disc.common.components.ncw > 0) GetW<M>(disc.sol.wdg, disc.sol, disc.tmp, disc.app, disc.common, backend);
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


#endif
