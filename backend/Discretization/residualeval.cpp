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

template <class M>
void CResidual<M>::evalResidual(Int backend)
{
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common; [[maybe_unused]] auto& driver_abi = disc.driver_abi;
    // compute the residual vector
    Residual<M>(sol, res, app, master, mesh, tmp, common, common.cublasHandle, backend);
}

// residual evaluation
template <class M>
void CResidual<M>::evalResidual(dstype* Ru, dstype* u, Int backend)
{ 
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common; [[maybe_unused]] auto& driver_abi = disc.driver_abi;
    // insert u into udg
    ArrayInsert(sol.udg, u, common.grid.npe, common.components.nc, common.meshsizes.ne, 0, common.grid.npe, 
            0, common.components.ncu, 0, common.meshsizes.ne1);  

    // compute the residual vector R(u)
    Residual<M>(sol, res, app, master, mesh, tmp, common, common.cublasHandle, backend);

    // copy the residual vector to Ru
    ArrayCopy(Ru, res.Ru, common.sizes.ndof1);
}

// q evaluation
template <class M>
void CResidual<M>::evalQ(Int backend)
{
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common; [[maybe_unused]] auto& driver_abi = disc.driver_abi;
    if (common.spatialScheme == 0) {
        // LDG computes q through the model flux kernels.
        ComputeQ<M>(sol, res, app, master, mesh, tmp, common, common.cublasHandle, backend);
    }
    else if (common.spatialScheme == 1) {
        // HDG recovers q from the element state and trace unknowns.
        hdgGetQ(sol.udg, sol.uh, sol, res, mesh, tmp, common, backend);
    }
    else {
        error("Spatial discretization scheme is not implemented");
    }
}

template <class M>
void CResidual<M>::evalQSer(Int backend)
{
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common; [[maybe_unused]] auto& driver_abi = disc.driver_abi;
    // compute the flux q    
    GetUhat<M>(sol, res, app, master, mesh, tmp, common, common.cublasHandle, 0, common.meshsizes.nbf, backend);        
    GetQ(sol, res, app, master, mesh, tmp, common, common.cublasHandle, 0, common.meshsizes.nbe, 0, common.meshsizes.nbf, backend);        
}

template <class M>
void CResidual<M>::evalQ(dstype* q, dstype* u, Int backend)
{
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common; [[maybe_unused]] auto& driver_abi = disc.driver_abi;
    // insert u into udg
    ArrayInsert(sol.udg, u, common.grid.npe, common.components.nc, common.meshsizes.ne, 0, common.grid.npe, 
            0, common.components.ncu, 0, common.meshsizes.ne1);

    if (common.spatialScheme == 0) {
        // LDG computes q through the model flux kernels.
        ComputeQ<M>(sol, res, app, master, mesh, tmp, common, common.cublasHandle, backend);
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


template <class M>
void CResidual<M>::updateUDG(dstype* u, Int backend)
{
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common; [[maybe_unused]] auto& driver_abi = disc.driver_abi;
    // insert u into udg
    ArrayInsert(sol.udg, u, common.grid.npe, common.components.nc, common.meshsizes.ne, 0, common.grid.npe, 
            0, common.components.ncu, 0, common.meshsizes.ne1);

    if (common.components.ncq>0)
        // compute the flux q
        ComputeQ<M>(sol, res, app, master, mesh, tmp, common, common.cublasHandle, backend);
}

template <class M>
void CResidual<M>::updateU(dstype* u, Int backend)
{
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common; [[maybe_unused]] auto& driver_abi = disc.driver_abi;
    // insert u into udg
    ArrayInsert(sol.udg, u, common.grid.npe, common.components.nc, common.meshsizes.ne, 0, common.grid.npe, 
            0, common.components.ncu, 0, common.meshsizes.ne1);
}

template <class M>
void CResidual<M>::evalAVfield(dstype* avField, dstype* u, Int backend)
{
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common; [[maybe_unused]] auto& driver_abi = disc.driver_abi;
    // insert u into udg
    ArrayInsert(sol.udg, u, common.grid.npe, common.components.nc, common.meshsizes.ne, 0, common.grid.npe, 
            0, common.components.ncu, 0, common.meshsizes.ne);
    
    // compute the flux q
    if (common.components.ncq>0)        
        ComputeQ<M>(sol, res, app, master, mesh, tmp, common, common.cublasHandle, backend);

    // compute the av field
    EXASIM_DRIVER_CALL(AvfieldDriver, avField, sol.xdg, sol.udg, sol.odg, sol.wdg, mesh, master, app, sol, tmp, common, backend);    
}

template <class M>
void CResidual<M>::evalAVfield(dstype* avField, Int backend)
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
    EXASIM_DRIVER_CALL(AvfieldDriver, avField, sol.xdg, sol.udg, sol.odg, sol.wdg, mesh, master, app, sol, tmp, common, backend);
}

// Compute the model initial conditions (layer A) for the owned discretization's solution.
// Thin entry over the free initializeSolution(); relocated out of the CDiscretization ctor so
// the operator drives its own initialization. The free function reads disc.common's sizes and
// dispatches the model init drivers in the solution's execution space (host on CPU, device on GPU).
template <class M>
void CResidual<M>::initializeSolution()
{
    if constexpr (std::is_same_v<M, exasim::detail::AbiAdapter>) {
        // Runtime-ABI build: the free initializeSolution dispatches the model init drivers
        // through driver_abi (the legacy/global ::Init*Driver). Byte-identical to before.
        ::initializeSolution(disc.sol, disc.app, disc.driver_abi, disc.common);
    } else {
        // Concrete-model build (GAP3, C3): route the model initial conditions through the templated
        // exasim::Init*Driver<M> kernels (which inline M::initu / M::initwdg / ...), with NO driver_abi.
        // Mirrors the free initializeSolution() in setstructs.cpp (same needs/wave guards); the templated
        // drivers read sizes from common, so no explicit ncx/nc/npe/ne arguments.
        auto& sol = disc.sol; auto& app = disc.app; auto& common = disc.common;
        if (sol.needudginit) {
            if (common.timeparams.wave == 0)
                exasim::InituDriver<M>(sol.udg, sol.xdg, app, common, common.backend);
            else
                exasim::InitudgDriver<M>(sol.udg, sol.xdg, app, common, common.backend);
        }
        if (sol.needodginit)
            exasim::InitodgDriver<M>(sol.odg, sol.xdg, app, common, common.backend);
        if (sol.needwdginit)
            exasim::InitwdgDriver<M>(sol.wdg, sol.xdg, app, common, common.backend);
    }
}

// Recover the initial operator state (q, uh, q-matrices) from the initialized u.
// Extracted verbatim from the CDiscretization constructor (option 2): the q/uh recovery is
// the operator applying itself, so it belongs to the operator (CResidual), not the function
// space. Members are bound as disc.* references; the kernel calls are unchanged.
template <class M>
void CResidual<M>::recoverInitialState(Int backend, bool postprocessOnly)
{
    [[maybe_unused]] auto& sol = disc.sol; [[maybe_unused]] auto& res = disc.res; [[maybe_unused]] auto& app = disc.app;
    [[maybe_unused]] auto& master = disc.master; [[maybe_unused]] auto& mesh = disc.mesh; [[maybe_unused]] auto& tmp = disc.tmp;
    [[maybe_unused]] auto& common = disc.common;

    if (common.spatialScheme == 0) {  // LDG: recover the auxiliary flux q from the initial u
        if ((common.components.ncq>0) && (common.timeparams.wave==0)) {
            GetUhat<M>(sol, res, app, master, mesh, tmp, common, common.cublasHandle, 0, common.meshsizes.nbf, backend);
            GetQ(sol, res, app, master, mesh, tmp, common, common.cublasHandle, 0, common.meshsizes.nbe, 0, common.meshsizes.nbf, backend);
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
