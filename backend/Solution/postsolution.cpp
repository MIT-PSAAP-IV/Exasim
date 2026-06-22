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
#ifndef __POSTSOLUTION
#define __POSTSOLUTION

#include "postsolution.h"
#include "previoussolutions.cpp"
#include "updatesolution.cpp"
#include "updatesource.cpp"
#include "timestepcoeff.cpp"
//#include "avsolution.cpp"

#ifdef TIMESTEP  
#include <sys/time.h>
#endif

void CSolution::InitSolution(Int backend) 
{            
    if (disc.common.spatialScheme==0) {
        ArrayExtract(solv.sys.u, disc.sol.udg, disc.common.npe, disc.common.nc, disc.common.ne1, 
              0, disc.common.npe, 0, disc.common.ncu, 0, disc.common.ne1);                                                  
    }
    else if (disc.common.spatialScheme==1) {      
        ArrayCopy(solv.sys.u, disc.sol.uh, disc.common.ndofuhat);
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
            cout<<"Compute solution average = "<<disc.common.compudgavg<<endl;
        
        if (disc.common.compudgavg == 1) {
            string filename = disc.common.fileout + "avg_np" + NumberToString(disc.common.mpiRank) + ".bin";
            disc.common.readudgavg = fileexists(filename);
            if (disc.common.mpiRank==0)
                cout<<"File exist = "<<disc.common.readudgavg<<endl;
            if (disc.common.readudgavg == 0)
                ArraySetValue(disc.sol.udgavg, zero, disc.common.ndofudg1+1);
            else 
                readarrayfromfile(filename, &disc.sol.udgavg, disc.common.ndofudg1+1, backend);   
        }        
    }    

    if (disc.common.ndofbou>0) {
        ArraySetValue(disc.sol.bouudgavg, zero, disc.common.ndofbou*disc.common.nc+1);
        ArraySetValue(disc.sol.bouuhavg, zero, disc.common.ndofbou*disc.common.ncu+1);
        if (disc.common.ncw > 0) ArraySetValue(disc.sol.bouwdgavg, zero, disc.common.ndofbou*disc.common.ncw+1); 
    }  
}

void CSolution::SaveSolutions(Int backend) 
{
    bool save = false;
    if (disc.common.timeparams.tdep==0) save = true;
    else 
        if (((disc.common.timestate.currentstep+1) % disc.common.saveSolFreq) == 0) save = true;             

    if (save == true) {        
        if (disc.common.saveSolOpt==0) {
            if (disc.common.spatialScheme > 0) {
                ArrayExtract(disc.res.Rq, disc.sol.udg, disc.common.npe, disc.common.nc, disc.common.ne1, 0, disc.common.npe, 0, disc.common.ncu, 0, disc.common.ne1);                                                  
                writearray(outsol, disc.res.Rq, disc.common.ndof1, backend);    
            }
            else
                writearray(outsol, solv.sys.u, disc.common.ndof1, backend);    
        }
        else
            writearray(outsol, disc.sol.udg, disc.common.ndofudg1, backend);    
        
        if (disc.common.ncw>0)
            writearray(outwdg, disc.sol.wdg, disc.common.ndofw1, backend);

        if (disc.common.spatialScheme==1)
            writearray(outuhat, disc.sol.uh, disc.common.ndofuhat, backend);
    }
   
   if (disc.common.timeparams.tdep==1) { 
        if (((disc.common.timestate.currentstep+1) % disc.common.saveRestart) == 0)             
        {        
            string filename = disc.common.fileout + "udg_t" + NumberToString(disc.common.timestate.currentstep+disc.common.timestepOffset+1) + "_np" + NumberToString(disc.common.mpiRank-disc.common.fileoffset) + ".bin";     
            writearray2file(filename, disc.sol.udg, disc.common.ndofudg1, backend);

            if (disc.common.compudgavg == 1) {
                string fn1 = disc.common.fileout + "solavg_np" + NumberToString(disc.common.mpiRank-disc.common.fileoffset) + ".bin"; 
                writearray2file(fn1, disc.sol.udgavg, disc.common.ndofudg1+1, backend);
            }        
          
            if (disc.common.ndofbou > 0) {
                string fn0 = disc.common.fileout + "bouudgavg_np" + NumberToString(disc.common.mpiRank-disc.common.fileoffset) + ".bin"; 
                writearray2file(fn0, disc.sol.bouudgavg, disc.common.ndofbou*disc.common.nc+1, backend);
                fn0 = disc.common.fileout + "bouuhavg_np" + NumberToString(disc.common.mpiRank-disc.common.fileoffset) + ".bin"; 
                writearray2file(fn0, disc.sol.bouuhavg, disc.common.ndofbou*disc.common.ncu+1, backend);
                if (disc.common.ncw > 0) {
                    fn0 = disc.common.fileout + "bouwdgavg_np" + NumberToString(disc.common.mpiRank-disc.common.fileoffset) + ".bin"; 
                    writearray2file(fn0, disc.sol.bouwdgavg, disc.common.ndofbou*disc.common.ncw+1, backend);
                }
            }        
          
            if (disc.common.ncw>0) {
                string fn = disc.common.fileout + "wdg_t" + NumberToString(disc.common.timestate.currentstep+disc.common.timestepOffset+1) + "_np" + NumberToString(disc.common.mpiRank-disc.common.fileoffset) + ".bin";                    
                writearray2file(fn, solv.sys.wtmp, disc.common.ndofw1, backend);
            }                        

            if (disc.common.spatialScheme==1) {
                string fn2 = disc.common.fileout + "_uhat_t" + NumberToString(disc.common.timestate.currentstep+disc.common.timestepOffset+1) + "_np" + NumberToString(disc.common.mpiRank-disc.common.fileoffset) + ".bin";                    
                writearray2file(fn2, disc.sol.uh, disc.common.ndofuhat, backend);        
            }
        }    
   }
    
   // if (disc.common.timeparams.tdep==1) { 
   //      if (((disc.common.timestate.currentstep+1) % disc.common.saveSolFreq) == 0)             
   //      {        
   //          string filename = disc.common.fileout + "udg_t" + NumberToString(disc.common.timestate.currentstep+disc.common.timestepOffset+1) + "_np" + NumberToString(disc.common.mpiRank-disc.common.fileoffset) + ".bin";     
   //          if (disc.common.saveSolOpt==0)
   //              writearray2file(filename, solv.sys.u, disc.common.ndof1, backend);
   //          else
   //              writearray2file(filename, disc.sol.udg, disc.common.ndofudg1, backend);
   // 
   //          if (disc.common.ncw>0) {
   //              string fn = disc.common.fileout + "_wdg_t" + NumberToString(disc.common.timestate.currentstep+disc.common.timestepOffset+1) + "_np" + NumberToString(disc.common.mpiRank-disc.common.fileoffset) + ".bin";                    
   //              writearray2file(fn, solv.sys.wtmp, disc.common.ndofw1, backend);
   //          }                        
   // 
   //          if (disc.common.compudgavg == 1) {
   //              string fn1 = disc.common.fileout + "avg_np" + NumberToString(disc.common.mpiRank-disc.common.fileoffset) + ".bin"; 
   //              writearray2file(fn1, disc.sol.udgavg, disc.common.ndofudg1+1, backend);
   //          }
   // 
   //          if (disc.common.spatialScheme==1) {
   //              string fn2 = disc.common.fileout + "_uhat_t" + NumberToString(disc.common.timestate.currentstep+disc.common.timestepOffset+1) + "_np" + NumberToString(disc.common.mpiRank-disc.common.fileoffset) + ".bin";                    
   //              writearray2file(fn2, disc.sol.uh, disc.common.ndofuhat, backend);        
   //          }
   //      }    
   // }
   // else {
   //      string filename = disc.common.fileout + "udg_np" + NumberToString(disc.common.mpiRank-disc.common.fileoffset) + ".bin";                    
   //      if (disc.common.saveSolOpt==0)
   //          writearray2file(filename, solv.sys.u, disc.common.ndof1, backend);
   //      else
   //          writearray2file(filename, disc.sol.udg, disc.common.ndofudg1, backend);       
   // 
   //      if (disc.common.ncw>0) {
   //          string fn = disc.common.fileout + "_wdg_np" + NumberToString(disc.common.mpiRank-disc.common.fileoffset) + ".bin";                    
   //          writearray2file(fn, disc.sol.wdg, disc.common.ndofw1, backend);     
   //      }                
   // 
   //      if (disc.common.spatialScheme==1) {
   //          string filename1 = disc.common.fileout + "_uhat_np" + NumberToString(disc.common.mpiRank-disc.common.fileoffset) + ".bin";                    
   //          writearray2file(filename1, disc.sol.uh, disc.common.ndofuhat, backend);        
   //      }
   // }    
}

void CSolution::ReadSolutions(Int backend) 
{
   if (disc.common.timeparams.tdep==1) { 
        if (((disc.common.timestate.currentstep+1) % disc.common.saveRestart) == 0)             
        {        
            string filename = disc.common.fileout + "udg_t" + NumberToString(disc.common.timestate.currentstep+disc.common.timestepOffset+1) + "_np" + NumberToString(disc.common.mpiRank-disc.common.fileoffset) + ".bin";     
            // if (disc.common.saveSolOpt==0) {
            //     readarrayfromfile(filename, &disc.res.Rq, disc.common.ndof1, backend);
            //     // insert u into udg
            //     ArrayInsert(disc.sol.udg, disc.res.Rq, disc.common.npe, disc.common.nc, 
            //      disc.common.ne, 0, disc.common.npe, 0, disc.common.ncu, 0, disc.common.ne1);  
            // }
            // else
                readarrayfromfile(filename, &disc.sol.udg, disc.common.ndofudg1, backend);        
            
            if (disc.common.mpiRank==0) cout<<"filename = "<<filename<<endl;
          
            if (disc.common.ncw>0) {
                string fn = disc.common.fileout+"wdg_t" + NumberToString(disc.common.timestate.currentstep+disc.common.timestepOffset+1) + "_np" + NumberToString(disc.common.mpiRank-disc.common.fileoffset) + ".bin";                    
                readarrayfromfile(fn, &disc.sol.wdg, disc.common.ndofw1, backend);     
            }                      

            if (disc.common.spatialScheme==1) {
                string fn2 = disc.common.fileout + "_uhat_t" + NumberToString(disc.common.timestate.currentstep+disc.common.timestepOffset+1) + "_np" + NumberToString(disc.common.mpiRank-disc.common.fileoffset) + ".bin";                    
                readarrayfromfile(fn2, &disc.sol.uh, disc.common.ndofuhat, backend);        
            }              
        }                                
   }
   else {
        string filename = disc.common.fileout + "udg_np" + NumberToString(disc.common.mpiRank-disc.common.fileoffset) + ".bin";                    
        if (disc.common.saveSolOpt==0) {
            readarrayfromfile(filename, &solv.sys.u, disc.common.ndof1, backend);
            // insert u into udg
            ArrayInsert(disc.sol.udg, solv.sys.u, disc.common.npe, disc.common.nc, 
             disc.common.ne, 0, disc.common.npe, 0, disc.common.ncu, 0, disc.common.ne1);              
        }
        else
            readarrayfromfile(filename, &disc.sol.udg, disc.common.ndofudg1, backend, 3);      
             
        if (disc.common.ncw>0) {
            string fn = disc.common.fileout + "wdg_np" + NumberToString(disc.common.mpiRank-disc.common.fileoffset) + ".bin";                    
            readarrayfromfile(fn, &disc.sol.wdg, disc.common.ndofw1, backend, 3);     
        }                

        if (disc.common.spatialScheme==1) {
            string fn = disc.common.fileout + "uhat_np" + NumberToString(disc.common.mpiRank-disc.common.fileoffset) + ".bin";                    
            readarrayfromfile(fn, &disc.sol.uh, disc.common.ndofuhat, backend, 3);        
        }                                    
   }    
}

void CSolution::GetSolutions(Int step, Int backend)
{
    if (step < 0)
        error("GetSolutions: step must be nonnegative");

    const Int rank = disc.common.mpiRank - disc.common.fileoffset;
    const Int headerSize = 3;
    string filename = disc.common.fileout + "udg_np" + NumberToString(rank) + ".bin";

    if (disc.common.saveSolOpt == 0) {
        const Int skip = headerSize + step * disc.common.ndof1;
        if (disc.common.spatialScheme > 0) {
            readarrayfromfile(filename, &disc.res.Rq, disc.common.ndof1, backend, skip);
            ArrayInsert(disc.sol.udg, disc.res.Rq, disc.common.npe, disc.common.nc,
                        disc.common.ne, 0, disc.common.npe, 0, disc.common.ncu,
                        0, disc.common.ne1);
        }
        else {
            readarrayfromfile(filename, &solv.sys.u, disc.common.ndof1, backend, skip);
            ArrayInsert(disc.sol.udg, solv.sys.u, disc.common.npe, disc.common.nc,
                        disc.common.ne, 0, disc.common.npe, 0, disc.common.ncu,
                        0, disc.common.ne1);
        }
    }
    else {
        const Int skip = headerSize + step * disc.common.ndofudg1;
        readarrayfromfile(filename, &disc.sol.udg, disc.common.ndofudg1, backend, skip);
    }

    if (disc.common.ncw > 0) {
        string fn = disc.common.fileout + "wdg_np" + NumberToString(rank) + ".bin";
        const Int skip = headerSize + step * disc.common.ndofw1;
        readarrayfromfile(fn, &disc.sol.wdg, disc.common.ndofw1, backend, skip);
    }

    if (disc.common.spatialScheme == 1) {
        string fn = disc.common.fileout + "uhat_np" + NumberToString(rank) + ".bin";
        const Int skip = headerSize + step * disc.common.ndofuhat;
        readarrayfromfile(fn, &disc.sol.uh, disc.common.ndofuhat, backend, skip);
    }

    if ((disc.common.saveSolOpt == 0) && (disc.common.ncq > 0))
        disc.evalQ(backend);
}
 
void CSolution::SaveParaview(Int backend, std::string fname_modifier, bool force_tdep_write) 
{  
    // Decide whether we should write a file on this step
    bool writeSolution = false;
    
    if (disc.common.timeparams.tdep == 1) {
       if (disc.common.timestate.currentstep==0 && disc.common.mpiRank==0) {
          string ext = (disc.common.mpiProcs==1) ? "vtu" : "pvtu";                                  
          vis.pvdwrite_series(disc.common.fileout + "vis", disc.common.dt, disc.common.timeparams.tsteps, disc.common.saveSolFreq, ext);                          
       }
        
        // Time-dependent: only write every 'saveSolFreq' steps
        writeSolution = ((disc.common.timestate.currentstep + 1) % disc.common.saveSolFreq) == 0;
        writeSolution = writeSolution || force_tdep_write;
    } else {
        // Steady / not time-dependent: always write
        writeSolution = true;
    }

   if (writeSolution) {      
       int nc = disc.common.nc;  
       int ncx = disc.common.ncx;   
       int nco = disc.common.nco;  
       int ncw = disc.common.ncw;  
       int nsca = disc.common.qoiparams.nsca; 
       int nvec = disc.common.qoiparams.nvec;  
       int nten = disc.common.qoiparams.nten;     
       int npe  = disc.common.npe;     
       int ne   = disc.common.ne1;      
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
            VisScalarsDriver(f, xdg, udg, vdg, wdg, disc.mesh, disc.master, disc.app, disc.sol, disc.tmp, disc.common, npe, 0, ne, backend);                                 
            VisDG2CG(vis.scafields, f, disc.mesh.cgent2dgent, disc.mesh.colent2elem, disc.mesh.rowent2elem, ne, ncg, ndg, 1, 1, nsca);
       }    
       if (nvec > 0) {        
            VisVectorsDriver(f, xdg, udg, vdg, wdg, disc.mesh, disc.master, disc.app, disc.sol, disc.tmp, disc.common, npe, 0, ne, backend);                                 
            VisDG2CG(vis.vecfields, f, disc.mesh.cgent2dgent, disc.mesh.colent2elem, disc.mesh.rowent2elem, ne, ncg, ndg, 3, ncx, nvec);
       }
       if (nten > 0) {        
            VisTensorsDriver(f, xdg, udg, vdg, wdg, disc.mesh, disc.master, disc.app, disc.sol, disc.tmp, disc.common, npe, 0, ne, backend);                                 
            VisDG2CG(vis.tenfields, f, disc.mesh.cgent2dgent, disc.mesh.colent2elem, disc.mesh.rowent2elem, ne, ncg, ndg, vis.ntc, vis.ntc, nten);
       }

       string baseName = disc.common.fileout + "vis" + fname_modifier;
       if (disc.common.timeparams.tdep == 1) {
           std::ostringstream ss; 
           ss << std::setw(6) << std::setfill('0') << disc.common.timestate.currentstep+disc.common.timestepOffset+1; 
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
        if (disc.common.timeparams.tdep==1) 
            outqoi << std::setw(16) << std::scientific << std::setprecision(6) << disc.common.timestate.time;
        else outqoi << std::setw(16) << std::scientific << std::setprecision(6) << 0.0;
        writeQoIRow(outqoi, disc.common);
        outqoi << "\n";
    }
}

void CSolution::SaveOutputDG(Int backend) 
{
   if (disc.common.timeparams.tdep==1) { 
        if (((disc.common.timestate.currentstep+1) % disc.common.saveSolFreq) == 0)             
        {                    
            string filename1 = disc.common.fileout + "_outputDG_t" + NumberToString(disc.common.timestate.currentstep+disc.common.timestepOffset+1) + "_np" + NumberToString(disc.common.mpiRank-disc.common.fileoffset) + ".bin";     
            disc.evalOutput(solv.sys.v, backend);                        
            writearray2file(filename1, solv.sys.v, disc.common.ndofedg1, backend);       
        }                                
   }
   else {
        string filename1 = disc.common.fileout + "_outputDG_np" + NumberToString(disc.common.mpiRank-disc.common.fileoffset) + ".bin";                           
        disc.evalOutput(solv.sys.v, backend);
        writearray2file(filename1, solv.sys.v, disc.common.ndofedg1, backend);       
   }    
}

void CSolution::SaveOutputCG(Int backend) 
{
   if (disc.common.timeparams.tdep==1) { 
        if (((disc.common.timestate.currentstep+1) % disc.common.saveSolFreq) == 0)             
        {                    
            string filename1 = disc.common.fileout + "_outputCG_t" + NumberToString(disc.common.timestate.currentstep+disc.common.timestepOffset+1) + "_np" + NumberToString(disc.common.mpiRank-disc.common.fileoffset) + ".bin";     
            disc.evalOutput(solv.sys.v, backend);
            disc.DG2CG(solv.sys.v, solv.sys.v, solv.sys.x, disc.common.nce, 
                     disc.common.nce, disc.common.nce, backend);
            writearray2file(filename1, solv.sys.v, disc.common.ndofedg1, backend);                   
//             disc.DG2CG3(solv.sys.v, solv.sys.v, solv.sys.x, disc.common.nce, 
//                  disc.common.nce, disc.common.nce, backend);
//             writearray2file(filename1, solv.sys.v, disc.common.ndofucg, backend);               
        }                                
   }
   else {
        string filename1 = disc.common.fileout + "_outputCG_np" + NumberToString(disc.common.mpiRank-disc.common.fileoffset) + ".bin";                            
        disc.evalOutput(solv.sys.v, backend);
        disc.DG2CG(solv.sys.v, solv.sys.v, solv.sys.x, disc.common.nce, 
                 disc.common.nce, disc.common.nce, backend);
        writearray2file(filename1, solv.sys.v, disc.common.ndofedg1, backend);               
//         disc.DG2CG3(solv.sys.v, solv.sys.v, solv.sys.x, disc.common.nce, 
//                  disc.common.nce, disc.common.nce, backend);        
//         writearray2file(filename1, solv.sys.v, disc.common.ndofucg, backend);               
   }    
}        

void CSolution::SaveSolutionsOnBoundary(Int backend) 
{   
    if ( disc.common.saveSolBouFreq>0 ) {
        if (((disc.common.timestate.currentstep+1) % disc.common.saveSolBouFreq) == 0)             
        {        
            for (Int j=0; j<disc.common.nbf; j++) {
                Int f1 = disc.common.fblks[3*j]-1;
                Int f2 = disc.common.fblks[3*j+1];    
                Int ib = disc.common.fblks[3*j+2];            
                if (ib == disc.common.qoiparams.ibs) {     
                    Int npf = disc.common.npf; // number of nodes on master face      
                    Int npe = disc.common.npe; // number of nodes on master face      
                    Int nf = f2-f1;
                    Int nn = npf*nf; 
                    Int nc = disc.common.nc; // number of compoments of (u, q, p)            
                    Int ncu = disc.common.ncu;
                    Int ncw = disc.common.ncw;
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
    if ( disc.common.saveSolBouFreq>0 ) {
        for (Int j=0; j<disc.common.nbf; j++) {
            Int f1 = disc.common.fblks[3*j]-1;
            Int f2 = disc.common.fblks[3*j+1];    
            Int ib = disc.common.fblks[3*j+2];            
            if (ib == disc.common.qoiparams.ibs) {     
                Int nd = disc.common.nd; 
                Int npf = disc.common.npf; // number of nodes on master face      
                Int nf = f2-f1;
                Int nn = npf*nf; 
                Int ncx = disc.common.ncx; // number of compoments of (u, q, p)                            
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
//     if ( disc.common.saveSolBouFreq>0 ) {
//         if (((disc.common.timestate.currentstep+1) % disc.common.saveSolBouFreq) == 0)             
//         {        
//             for (Int j=0; j<disc.common.nbf; j++) {
//                 Int f1 = disc.common.fblks[3*j]-1;
//                 Int f2 = disc.common.fblks[3*j+1];    
//                 Int ib = disc.common.fblks[3*j+2];            
//                 if (ib == disc.common.qoiparams.ibs) {     
//                     Int npf = disc.common.npf; // number of nodes on master face      
//                     Int nf = f2-f1;
//                     Int nn = npf*nf; 
//                     Int nc = disc.common.nc; // number of compoments of (u, q, p)            
//                     GetArrayAtIndex(disc.tmp.tempn, disc.sol.udg, &disc.mesh.findudg1[npf*nc*f1], nn*nc);
//                     string filename = disc.common.fileout + "bou_t" + NumberToString(disc.common.timestate.currentstep+disc.common.timestepOffset+1) + "_np" + NumberToString(disc.common.mpiRank-disc.common.fileoffset) + ".bin";     
//                     writearray2file(filename, disc.tmp.tempn, nn*nc, backend);            
//                 }
//             }                                               
//         }                                
//     }
// }
// 
// void CSolution::SaveNodesOnBoundary(Int backend) 
// {   
//     if ( disc.common.saveSolBouFreq>0 ) {
//         for (Int j=0; j<disc.common.nbf; j++) {
//             Int f1 = disc.common.fblks[3*j]-1;
//             Int f2 = disc.common.fblks[3*j+1];    
//             Int ib = disc.common.fblks[3*j+2];            
//             if (ib == disc.common.qoiparams.ibs) {     
//                 Int npf = disc.common.npf; // number of nodes on master face      
//                 Int nf = f2-f1;
//                 Int nn = npf*nf; 
//                 Int ncx = disc.common.ncx; // number of compoments of (u, q, p)                            
//                 GetArrayAtIndex(disc.tmp.tempn, disc.sol.xdg, &disc.mesh.findxdg1[npf*ncx*f1], nn*ncx);
//                 string filename = disc.common.fileout + "node_np" + NumberToString(disc.common.mpiRank-disc.common.fileoffset) + ".bin";     
//                 writearray2file(filename, disc.tmp.tempn, nn*ncx, backend);            
//             }
//         }                                                                   
//     }
// }

#endif        
