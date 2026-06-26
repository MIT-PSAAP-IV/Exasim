/*
    CResidual -- evaluation of the discretized PDE residual R(u) and the auxiliary flux q,
    extracted from CDiscretization (full-split step 1). The residual is the equation
    discretized onto the function space (CDiscretization); it is a distinct object from the
    space it lives on. Behaviour re-home for now: CResidual holds a CDiscretization& and the
    method bodies are unchanged (the structs they used are bound as disc.* references). The
    data (res/sol/driver_abi) it operates on still lives on CDiscretization; ownership moves
    in a later step.
*/
#ifndef __RESIDUALEVAL
#define __RESIDUALEVAL

#include "residualeval.h"

void CResidual::evalResidual(Int backend)
{
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common; [[maybe_unused]] auto& driver_abi = disc.driver_abi;
    // compute the residual vector
    Residual<exasim::detail::AbiAdapter>(sol, res, app, master, mesh, tmp, common, common.cublasHandle, backend);
}

// residual evaluation
void CResidual::evalResidual(dstype* Ru, dstype* u, Int backend)
{ 
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common; [[maybe_unused]] auto& driver_abi = disc.driver_abi;
    // insert u into udg
    ArrayInsert(sol.udg, u, common.grid.npe, common.components.nc, common.meshsizes.ne, 0, common.grid.npe, 
            0, common.components.ncu, 0, common.meshsizes.ne1);  

    // compute the residual vector R(u)
    Residual<exasim::detail::AbiAdapter>(sol, res, app, master, mesh, tmp, common, common.cublasHandle, backend);

    // copy the residual vector to Ru
    ArrayCopy(Ru, res.Ru, common.sizes.ndof1);
}

// q evaluation
void CResidual::evalQ(Int backend)
{
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common; [[maybe_unused]] auto& driver_abi = disc.driver_abi;
    if (common.spatialScheme == 0) {
        // LDG computes q through the model flux kernels.
        ComputeQ<exasim::detail::AbiAdapter>(sol, res, app, master, mesh, tmp, common, common.cublasHandle, backend);
    }
    else if (common.spatialScheme == 1) {
        // HDG recovers q from the element state and trace unknowns.
        hdgGetQ<exasim::detail::AbiAdapter>(sol.udg, sol.uh, sol, res, mesh, tmp, common, backend);
    }
    else {
        error("Spatial discretization scheme is not implemented");
    }
}

void CResidual::evalQSer(Int backend)
{
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common; [[maybe_unused]] auto& driver_abi = disc.driver_abi;
    // compute the flux q    
    GetUhat<exasim::detail::AbiAdapter>(sol, res, app, master, mesh, tmp, common, common.cublasHandle, 0, common.meshsizes.nbf, backend);        
    GetQ<exasim::detail::AbiAdapter>(sol, res, app, master, mesh, tmp, common, common.cublasHandle, 0, common.meshsizes.nbe, 0, common.meshsizes.nbf, backend);        
}

void CResidual::evalQ(dstype* q, dstype* u, Int backend)
{
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common; [[maybe_unused]] auto& driver_abi = disc.driver_abi;
    // insert u into udg
    ArrayInsert(sol.udg, u, common.grid.npe, common.components.nc, common.meshsizes.ne, 0, common.grid.npe, 
            0, common.components.ncu, 0, common.meshsizes.ne1);

    if (common.spatialScheme == 0) {
        // LDG computes q through the model flux kernels.
        ComputeQ<exasim::detail::AbiAdapter>(sol, res, app, master, mesh, tmp, common, common.cublasHandle, backend);
    }
    else if (common.spatialScheme == 1) {
        // HDG recovers q from the element state and trace unknowns.
        hdgGetQ<exasim::detail::AbiAdapter>(sol.udg, sol.uh, sol, res, mesh, tmp, common, backend);
    }
    else {
        error("Spatial discretization scheme is not implemented");
    }

    // get q from udg
    ArrayExtract(q, sol.udg, common.grid.npe, common.components.nc, common.meshsizes.ne, 0, common.grid.npe, 
            common.components.ncu, common.components.ncu+common.components.ncq, 0, common.meshsizes.ne1);
}


void CResidual::updateUDG(dstype* u, Int backend)
{
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common; [[maybe_unused]] auto& driver_abi = disc.driver_abi;
    // insert u into udg
    ArrayInsert(sol.udg, u, common.grid.npe, common.components.nc, common.meshsizes.ne, 0, common.grid.npe, 
            0, common.components.ncu, 0, common.meshsizes.ne1);

    if (common.components.ncq>0)
        // compute the flux q
        ComputeQ<exasim::detail::AbiAdapter>(sol, res, app, master, mesh, tmp, common, common.cublasHandle, backend);
}

void CResidual::updateU(dstype* u, Int backend)
{
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common; [[maybe_unused]] auto& driver_abi = disc.driver_abi;
    // insert u into udg
    ArrayInsert(sol.udg, u, common.grid.npe, common.components.nc, common.meshsizes.ne, 0, common.grid.npe, 
            0, common.components.ncu, 0, common.meshsizes.ne1);
}

void CResidual::evalAVfield(dstype* avField, dstype* u, Int backend)
{
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common; [[maybe_unused]] auto& driver_abi = disc.driver_abi;
    // insert u into udg
    ArrayInsert(sol.udg, u, common.grid.npe, common.components.nc, common.meshsizes.ne, 0, common.grid.npe, 
            0, common.components.ncu, 0, common.meshsizes.ne);
    
    // compute the flux q
    if (common.components.ncq>0)        
        ComputeQ<exasim::detail::AbiAdapter>(sol, res, app, master, mesh, tmp, common, common.cublasHandle, backend);

    // compute the av field
    AvfieldDriver(avField, sol.xdg, sol.udg, sol.odg, sol.wdg, driver_abi, mesh, master, app, sol, tmp, common, backend);    
}

void CResidual::evalAVfield(dstype* avField, Int backend)
{    
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common; [[maybe_unused]] auto& driver_abi = disc.driver_abi;
    
#ifdef  HAVE_MPI    
    Int bsz = common.grid.npe*common.components.nc;
    Int nudg = common.grid.npe*common.components.nc;
    Int n;
    
    /* copy some portion of u to buffsend */
    //for (n=0; n<common.nelemsend; n++)         
    //    ArrayCopy(&tmp.buffsend[bsz*n], &sol.udg[nudg*common.elemsend[n]], bsz, backend);           
    GetArrayAtIndex(tmp.buffsend, sol.udg, mesh.elemsendudg, bsz*common.nelemsend);

#ifdef HAVE_CUDA
    cudaDeviceSynchronize();
#endif

#ifdef HAVE_HIP
    hipDeviceSynchronize();
#endif
    
    /* non-blocking send */
    Int neighbor, nsend, psend = 0, request_counter = 0;
    for (n=0; n<common.nnbsd; n++) {
        neighbor = common.nbsd[n];
        nsend = common.elemsendpts[n]*bsz;
        if (nsend>0) {
            MPI_Isend(&tmp.buffsend[psend], nsend, MPI_DOUBLE, neighbor, 0,
                   EXASIM_COMM_LOCAL, &common.requests[request_counter]);
            psend += nsend;
            request_counter += 1;
        }
    }

    /* non-blocking receive */
    Int nrecv, precv = 0;
    for (n=0; n<common.nnbsd; n++) {
        neighbor = common.nbsd[n];
        nrecv = common.elemrecvpts[n]*bsz;
        if (nrecv>0) {
            MPI_Irecv(&tmp.buffrecv[precv], nrecv, MPI_DOUBLE, neighbor, 0,
                   EXASIM_COMM_LOCAL, &common.requests[request_counter]);
            precv += nrecv;
            request_counter += 1;
        }
    }

    // non-blocking receive solutions on exterior and outer elements from neighbors
    /* wait until all send and receive operations are completely done */
    MPI_Waitall(request_counter, common.requests, common.statuses);

    /* copy buffrecv to udg */
    //for (n=0; n<common.nelemrecv; n++) 
    //    ArrayCopy(&sol.udg[nudg*common.elemrecv[n]], &tmp.buffrecv[bsz*n], bsz, backend);        
    PutArrayAtIndex(sol.udg, tmp.buffrecv, mesh.elemrecvudg, bsz*common.nelemrecv);
#endif
  
    // compute the av field
    AvfieldDriver(avField, sol.xdg, sol.udg, sol.odg, sol.wdg, driver_abi, mesh, master, app, sol, tmp, common, backend);           
}

#endif
