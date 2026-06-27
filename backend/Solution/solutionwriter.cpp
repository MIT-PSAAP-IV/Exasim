/*
    CSolutionWriter -- solution output (binary streams + QoI text) and read/write logic,
    extracted from CSolution (the "writer" half of the CSolution split). Method bodies are
    unchanged from CSolution; the referenced members keep the same names (disc/residual/vis/solv
    + the out* streams), so only the class qualifier changed.
*/
#ifndef __SOLUTIONWRITER
#define __SOLUTIONWRITER

#include "solutionwriter.h"

// --- open the output streams and write the initial solution (was the CSolution constructor body) ---
void CSolutionWriter::setup(bool postprocessOnly)
{
    int ncx = disc.common.components.ncx;
    int nd  = disc.common.grid.nd;
    int ncu = disc.common.components.ncu;
    int nc  = (disc.common.outputparams.saveSolOpt==0) ? disc.common.components.ncu : disc.common.components.nc;
    int ncw = disc.common.components.ncw;
    int npe = disc.common.grid.npe;
    int npf = disc.common.grid.npf;
    int ne  = disc.common.meshsizes.ne1;
    int nf  = disc.common.meshsizes.nf;
    int rank = disc.common.mpiRank;
    int offset = disc.common.outputparams.fileoffset;
    std::string base = disc.common.fileout;

    if (rank==0 && (disc.common.qoiparams.nvqoi > 0 || disc.common.qoiparams.nsurf > 0)) {
        // header written lazily on the first SaveQoI row (writeQoIHeaderOnce)
        outqoi.open(base + "qoi.txt", std::ios::out);
    }

    if (!postprocessOnly) {
        open_and_write(outsol, "udg_np", rank, offset, npe, nc, ne, base);

        if (ncw > 0)
            open_and_write(outwdg, "wdg_np", rank, offset, npe, ncw, ne, base);

        if (disc.common.spatialScheme==1)
            open_and_write(outuhat, "uhat_np", rank, offset, ncu, npf, nf, base);

        if ( disc.common.outputparams.saveSolBouFreq>0 ) {
            Int nfbou = 0;
            for (Int j=0; j<disc.common.meshsizes.nbf; j++) {
                Int f1 = disc.common.fblks[3*j]-1;
                Int f2 = disc.common.fblks[3*j+1];
                Int ib = disc.common.fblks[3*j+2];
                if (ib == disc.common.qoiparams.ibs) {
                    Int nfb = f2-f1;
                    nfbou += nfb;
                }
            }
            open_and_write(outbouxdg, "bouxdg_np", rank, offset, npf, nfbou, ncx, base);
            open_and_write(outboundg, "boundg_np", rank, offset, npf, nfbou, nd, base);
            open_and_write(outbouudg, "bouudg_np", rank, offset, npf, nfbou, disc.common.components.nc, base);
            open_and_write(outbouuhat, "bouuhat_np", rank, offset, npf, nfbou, ncu, base);
            if (ncw > 0) open_and_write(outbouwdg, "bouwdg_np", rank, offset, npf, nfbou, ncw, base);
        }
    }
}

// --- crash teardown: optional "_CRASH" Paraview, close all streams (caller dumps the .bin) ---
void CSolutionWriter::crashDump(Int backend)
{
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
}

// --- close/reopen output streams under a new fileout prefix (parameter sweeps) ---
void CSolutionWriter::ResetOutputFiles(const std::string& fileout)
    {
        if (outsol.is_open()) { outsol.close(); }
        if (outwdg.is_open()) { outwdg.close(); }
        if (outuhat.is_open()) { outuhat.close(); }
        if (outbouxdg.is_open()) { outbouxdg.close(); }
        if (outboundg.is_open()) { outboundg.close(); }
        if (outbouudg.is_open()) { outbouudg.close(); }
        if (outbouwdg.is_open()) { outbouwdg.close(); }
        if (outbouuhat.is_open()) { outbouuhat.close(); }
        if (outqoi.is_open()) { outqoi.close(); }

        disc.common.fileout = fileout;

        int ncx = disc.common.components.ncx;
        int nd = disc.common.grid.nd;
        int ncu = disc.common.components.ncu;
        int nc = (disc.common.outputparams.saveSolOpt==0) ? disc.common.components.ncu : disc.common.components.nc;
        int ncw = disc.common.components.ncw;
        int npe = disc.common.grid.npe;
        int npf = disc.common.grid.npf;
        int ne = disc.common.meshsizes.ne1;
        int nf = disc.common.meshsizes.nf;
        int rank = disc.common.mpiRank;
        int offset = disc.common.outputparams.fileoffset;

        if (rank==0 && (disc.common.qoiparams.nvqoi > 0 || disc.common.qoiparams.nsurf > 0)) {
            outqoi.open(fileout + "qoi.txt", std::ios::out);
            outqoi << std::setw(16) << std::left << "Time";
            for (size_t i = 0; i < disc.common.qoiparams.nvqoi; ++i)
                outqoi << std::setw(16) << std::left << "Domain_QoI" + std::to_string(i + 1);
            for (size_t i = 0; i < disc.common.qoiparams.nsurf; ++i)
                outqoi << std::setw(16) << std::left << "Boundary_QoI" + std::to_string(i + 1);
            outqoi << "\n";
        }

        open_and_write(outsol, "udg_np", rank, offset, npe, nc, ne, fileout);

        if (ncw > 0)
            open_and_write(outwdg, "wdg_np", rank, offset, npe, ncw, ne, fileout);

        if (disc.common.spatialScheme==1)
            open_and_write(outuhat, "uhat_np", rank, offset, ncu, npf, nf, fileout);

        if (disc.common.outputparams.saveSolBouFreq>0) {
            Int nfbou = 0;
            for (Int j=0; j<disc.common.meshsizes.nbf; j++) {
                Int f1 = disc.common.fblks[3*j]-1;
                Int f2 = disc.common.fblks[3*j+1];
                Int ib = disc.common.fblks[3*j+2];
                if (ib == disc.common.qoiparams.ibs)
                    nfbou += f2-f1;
            }

            open_and_write(outbouxdg, "bouxdg_np", rank, offset, npf, nfbou, ncx, fileout);
            open_and_write(outboundg, "boundg_np", rank, offset, npf, nfbou, nd, fileout);
            open_and_write(outbouudg, "bouudg_np", rank, offset, npf, nfbou, disc.common.components.nc, fileout);
            open_and_write(outbouuhat, "bouuhat_np", rank, offset, npf, nfbou, ncu, fileout);
            if (ncw > 0)
                open_and_write(outbouwdg, "bouwdg_np", rank, offset, npf, nfbou, ncw, fileout);
        }
    }

// --- methods moved verbatim from CSolution ---
void CSolutionWriter::evalMonitor(dstype* output, dstype* udg, dstype* wdg, Int nc, Int backend)
{
    MonitorDriver(output, nc, disc.sol.xdg, udg, disc.sol.odg, wdg, disc.driver_abi,
                  disc.mesh, disc.master, disc.app, disc.sol, disc.tmp, disc.common, backend);
}

// Re-homed from CDiscretization (S4): computing the output field for I/O is an output concern.
// MPI-halo-exchanges the owned disc's udg across neighbors, then calls the model OutputDriver.
void CSolutionWriter::evalOutput(dstype* output, Int backend)
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


void CSolutionWriter::SaveSolutions(Int backend) 
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

void CSolutionWriter::ReadSolutions(Int backend) 
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

void CSolutionWriter::GetSolutions(Int step, Int backend)
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
 
void CSolutionWriter::SaveParaview(Int backend, std::string fname_modifier, bool force_tdep_write) 
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

void CSolutionWriter::SaveQoI(Int backend) 
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

void CSolutionWriter::SaveOutputCG(Int backend) 
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

void CSolutionWriter::SaveSolutionsOnBoundary(Int backend) 
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

void CSolutionWriter::SaveNodesOnBoundary(Int backend) 
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

#endif
