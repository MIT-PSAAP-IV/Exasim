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
        hdgGetQ(sol.udg, sol.uh, sol, res, mesh, tmp, common, backend);
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
        hdgGetQ(sol.udg, sol.uh, sol, res, mesh, tmp, common, backend);
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

// Compute the model initial conditions (layer A) for the owned discretization's solution.
// Thin entry over the free initializeSolution(); relocated out of the CDiscretization ctor so
// the operator drives its own initialization. The free function reads disc.common's sizes and
// dispatches the model init drivers in the solution's execution space (host on CPU, device on GPU).
void CResidual::initializeSolution()
{
    ::initializeSolution(disc.sol, disc.app, disc.driver_abi, disc.common);
}

// Recover the initial operator state (q, uh, q-matrices) from the initialized u.
// Extracted verbatim from the CDiscretization constructor (option 2): the q/uh recovery is
// the operator applying itself, so it belongs to the operator (CResidual), not the function
// space. Members are bound as disc.* references; the kernel calls are unchanged.
void CResidual::recoverInitialState(Int backend, bool postprocessOnly)
{
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common;

    if (common.spatialScheme == 0) {  // LDG: recover the auxiliary flux q from the initial u
        if ((common.components.ncq>0) && (common.timeparams.wave==0)) {
            GetUhat<exasim::detail::AbiAdapter>(sol, res, app, master, mesh, tmp, common, common.cublasHandle, 0, common.meshsizes.nbf, backend);
            GetQ<exasim::detail::AbiAdapter>(sol, res, app, master, mesh, tmp, common, common.cublasHandle, 0, common.meshsizes.nbe, 0, common.meshsizes.nbf, backend);
        }
    }
    else if (common.spatialScheme > 0) {  // HDG: recover the trace uh, q-matrices, and q
        Int npe = common.grid.npe;
        Int npf = common.grid.npf;
        Int ncu = common.components.ncu;
        Int nc  = common.components.nc;
        Int nf  = common.meshsizes.nf;
        Int ncq = common.components.ncq;
        Int ne  = common.meshsizes.ne;
        int ncu12 = common.szinterfacefluxmap;

        // compute uhat by getting u on faces (unless it was read from a restart file)
        if (!common.read_uh)
            GetFaceNodes(sol.uh, sol.udg, mesh.f2e, mesh.perm, npf, ncu, npe, nc, nf);

        if (ncq > 0) {
            if (common.couplingparams.coupledinterface>0 && !postprocessOnly) {
                res.szGi = npf*ncu12*npe*ncq*common.couplingparams.ncie;
                TemplateMalloc(&res.Gi, res.szGi, backend);
            }
            // compute M^{-1} * C and M^{-1} * E (q-matrices) and store in res.C / res.E
            qEquation(sol, res, app, master, mesh, tmp, common, backend);
            // compute the flux q = -nabla u and store it in sol.udg
            if (common.timeparams.wave == 0 && sol.szudg != npe*nc*ne)
                hdgGetQ(sol.udg, sol.uh, sol, res, mesh, tmp, common, backend);
        }
    }
}

#endif
