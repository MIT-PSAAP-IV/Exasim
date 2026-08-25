/*
 * Discretization Module
 * =====================
 * This file implements the core routines for initializing and managing the discretization structures
 * used in the Exasim backend for both CPU and GPU architectures. It supports multiple spatial schemes
 * (LDG, HDG) and various preconditioners, and is designed for parallel execution with MPI and GPU acceleration.

 * Main Components:
 * ----------------
 * - crs_init: Initializes the compressed row storage (CRS) structures for superelement-based preconditioning.
 * - CDiscretization: Main class encapsulating all discretization data and operations.
 *   - Constructor: Initializes all data structures, allocates memory, and sets up geometry and solution fields.
 *   - Destructor: Releases all allocated resources and handles for both CPU and GPU.
 *   - compGeometry: Computes and stores geometric quantities for elements and faces.
 *   - compMassInverse: Computes and stores the inverse of the mass matrix.
 *   - hdgAssembleLinearSystem: Assembles the HDG linear system and applies the selected preconditioner.
 *   - hdgAssembleResidual: Assembles the HDG residual vector.
 *   - evalResidual: Evaluates the residual vector for the current solution.
 *   - evalQ: Computes the flux q for the current solution.
 *   - evalQSer: Serial evaluation of flux q for non-wave problems.
 *   - evalMatVec: Computes matrix-vector products for Jacobian-vector operations.
 *   - updateUDG/updateU: Updates the solution fields with new values.
 *   - evalAVfield: Computes artificial viscosity fields, with MPI support for distributed domains.
 *   - evalOutput: Computes output quantities, with MPI support for distributed domains.
 *   - evalMonitor: Computes monitoring quantities for the solution.
 *   - DG2CG/DG2CG2/DG2CG3: Converts DG fields to CG fields using various mapping strategies.

 * Features:
 * ---------
 * - Supports both CPU and GPU backends (CUDA/HIP).
 * - MPI parallelization for distributed memory architectures.
 * - Flexible memory management for host and device.
 * - Multiple preconditioners: Block Jacobi, Elemental Additive Schwarz, Superelement Additive Schwarz (ILU0).
 * - Handles both LDG and HDG spatial schemes.
 * - Modular design for geometry, solution, and output computations.

 * Usage:
 * ------
 * Instantiate CDiscretization with appropriate input files and parallelization parameters.
 * Use member functions to assemble systems, evaluate residuals, compute fluxes, and manage solution fields.

 * Note:
 * -----
 * This file includes several implementation files (.cpp) directly for modularity and to support template-based
 * memory management and device/host operations.
 */
#ifndef __DISCRETIZATION
#define __DISCRETIZATION

#include <cstring>
#include <cstdlib>

#ifdef HAVE_CUDA
#include "gpuDeviceInfo.cpp"
#endif

#include "discretization.h"
#include "interfacesampler.h"   // CInterfaceSampler decl, needed early by the wall-model sampling
#include "wallmodelbuild.h"     // CWallModel decl, used by the constructor
#include "ioutilities.cpp"
#include "../PointLocator/pointlocator.h"

// #ifdef HAVE_TEXT2CODE
// #include "../Model/Text2codeGenerated/ModelDrivers.cpp"
// #elif defined(HAVE_BUILTINMODEL)
// #include "../Model/BuiltIn/BuiltinModelDrivers.cpp"
// #else
// #include "../Model/FrontendGenerated/KokkosDrivers.cpp"
// #endif

#include "connectivity.cpp"
#include "readbinaryfiles.cpp"
#include "setstructs.cpp"
#include "residual.hpp"  // unified (U)
//#include "ldgjactest.cpp"
#include "ldgblockjacobian.cpp"
#include "matvec.hpp"  // unified (U)
#include "qoicalculation.hpp"  // unified templated QoI; production instantiates <AbiAdapter>
#include "wallmodel.cpp"

template <class T = ::dstype, class I = ::Int>
void crs_init(commonstructT<T,I>& common, meshstructT<T,I>& mesh, int *elem, int nse, int nese)
{
    using dstype = T; using Int = I;
    common.nse = nse;
    common.nese = nese;
    
    int *row_ptr = NULL; 
    int *col_ind = NULL; 
    int *face = NULL; 
    int *f2eelem = NULL; 
    int *f2e = NULL; 
    TemplateMalloc(&f2e, 4*common.meshsizes.nf, 0);
    TemplateCopytoHost(f2e, mesh.f2e, 4*common.meshsizes.nf, common.backend); 
    
    int nfelem = crs_faceordering(&row_ptr, &col_ind, &face, &f2eelem, elem, f2e, common.nse, common.nese, common.meshsizes.nfe, common.meshsizes.nf);

    common.nfse = nfelem;
    common.nnz = row_ptr[common.nfse];      
        
    int n = 2*(common.meshsizes.nfe-1);
    TemplateMalloc(&common.bjindex.ind_ii, nfelem, 0);
    TemplateMalloc(&common.bjindex.ind_ji, nfelem*n, 0);
    TemplateMalloc(&common.bjindex.ind_jl, nfelem*n*n, 0);
    TemplateMalloc(&common.bjindex.ind_il, nfelem*n*n, 0);
    TemplateMalloc(&common.bjindex.num_ji, nfelem, 0);
    TemplateMalloc(&common.bjindex.num_jl, nfelem*n, 0);
    TemplateMalloc(&common.bjindex.Lind_ji, nfelem*n*2, 0);
    TemplateMalloc(&common.bjindex.Uind_ji, nfelem*n*2, 0);
    TemplateMalloc(&common.bjindex.Lnum_ji, nfelem*2, 0);
    TemplateMalloc(&common.bjindex.Unum_ji, nfelem*3, 0);
    for (int i=0; i<nfelem; i++) common.bjindex.ind_ii[i] = -1;
    for (int i=0; i<nfelem*n; i++) common.bjindex.ind_ji[i] = -1;
    for (int i=0; i<nfelem*n*n; i++) common.bjindex.ind_jl[i] = -1;
    for (int i=0; i<nfelem*n*n; i++) common.bjindex.ind_il[i] = -1;
    for (int i=0; i<nfelem; i++) common.bjindex.num_ji[i] = 0;
    for (int i=0; i<nfelem*n; i++) common.bjindex.num_jl[i] = 0;
    for (int i=0; i<nfelem*n*2; i++) common.bjindex.Lind_ji[i] = -1;
    for (int i=0; i<nfelem*n*2; i++) common.bjindex.Uind_ji[i] = -1;
    for (int i=0; i<nfelem*2; i++) common.bjindex.Lnum_ji[i] = 0;
    for (int i=0; i<nfelem*3; i++) common.bjindex.Unum_ji[i] = 0;    
    
    crs_indexingilu0(common.bjindex.ind_ii, common.bjindex.ind_ji, common.bjindex.ind_jl, common.bjindex.ind_il, common.bjindex.num_ji, common.bjindex.num_jl, 
            common.bjindex.Lind_ji, common.bjindex.Uind_ji, common.bjindex.Lnum_ji, common.bjindex.Unum_ji, row_ptr, col_ind, common.meshsizes.nfe, nfelem);

//     print2iarray(f2eelem, common.meshsizes.nfe, nfelem);
//     print2iarray(row_ptr, 1, nfelem+1);
//     print2iarray(col_ind, 1, row_ptr[nfelem]);    
//     print2iarray(common.bjindex.ind_ii, 1, nfelem);
//     print2iarray(common.bjindex.ind_ji, n, nfelem);
//     print2iarray(common.bjindex.ind_jl, n*n, nfelem);
//     print2iarray(common.bjindex.ind_il, n*n, nfelem);
//     print2iarray(common.bjindex.num_ji, 1, nfelem);
//     print2iarray(common.bjindex.num_jl, n, nfelem);
//     print2iarray(common.bjindex.Lind_ji, n*2, nfelem);
//     print2iarray(common.bjindex.Uind_ji, n*2, nfelem);
//     print2iarray(common.bjindex.Lnum_ji, 2, nfelem);
//     print2iarray(common.bjindex.Unum_ji, 3, nfelem);

    TemplateMalloc(&mesh.row_ptr, nfelem+1, common.backend);
    TemplateMalloc(&mesh.col_ind, row_ptr[nfelem],common.backend);
    TemplateMalloc(&mesh.face, nse*nfelem, common.backend);
    TemplateCopytoDevice(mesh.row_ptr, row_ptr, nfelem+1, common.backend);                       
    TemplateCopytoDevice(mesh.col_ind, col_ind, row_ptr[nfelem], common.backend);    
    TemplateCopytoDevice(mesh.face, face, nse*nfelem, common.backend);      
    
//     writearray2file(common.fileout + "elem.bin", elem, nse*nese, 0);
//     writearray2file(common.fileout + "f2e.bin", f2e, 4*common.meshsizes.nf, 0);
//     
//     writearray2file(common.fileout + "ind_ii.bin", common.bjindex.ind_ii, nfelem, 0);
//     writearray2file(common.fileout + "ind_ji.bin", common.bjindex.ind_ji, n*nfelem, 0);
//     writearray2file(common.fileout + "ind_jl.bin", common.bjindex.ind_jl, n*n*nfelem, 0);
//     writearray2file(common.fileout + "ind_il.bin", common.bjindex.ind_il, n*n*nfelem, 0);
//     writearray2file(common.fileout + "num_ji.bin", common.bjindex.num_ji, nfelem, 0);
//     writearray2file(common.fileout + "num_jl.bin", common.bjindex.num_jl, n*nfelem, 0);
//     writearray2file(common.fileout + "Lind_ji.bin", common.bjindex.Lind_ji, 2*n*nfelem, 0);
//     writearray2file(common.fileout + "Uind_ji.bin", common.bjindex.Uind_ji, 2*n*nfelem, 0);
//     writearray2file(common.fileout + "Lnum_ji.bin", common.bjindex.Lnum_ji, 2*nfelem, 0);
//     writearray2file(common.fileout + "Unum_ji.bin", common.bjindex.Unum_ji, 3*nfelem, 0);
//     
//     writearray2file(common.fileout + "row_ptr.bin", mesh.row_ptr, nfelem+1, common.backend);
//     writearray2file(common.fileout + "col_ind.bin", mesh.col_ind, row_ptr[nfelem], common.backend);
//     writearray2file(common.fileout + "face.bin", mesh.face, nse*nfelem, common.backend);
    
    CPUFREE(row_ptr);
    CPUFREE(col_ind);
    CPUFREE(face);
    CPUFREE(f2eelem);
    CPUFREE(f2e);
}
      
template <class T = ::dstype, class I = ::Int>
void BuildElementBlockBoundaryFaces(commonstructT<T,I>& common, meshstructT<T,I>& mesh, ::Int backend)
{
    using dstype = T; using Int = I;
    Int nfe = common.meshsizes.nfe;
    Int ne = common.meshsizes.ne;
    Int nbe = common.meshsizes.nbe;

    int nboufaces = 0;
    int maxbc = 0;
    for (int i = 0; i < nfe*ne; i++) {
        if (mesh.bf[i] > 0)
            nboufaces++;
        maxbc = max(maxbc, mesh.bf[i]);
    }
    common.meshsizes.maxnbc = maxbc;

    CPUFREE(common.nboufaces);
    TemplateFree(mesh.boufaces, backend);
    common.nboufaces = nullptr;
    mesh.boufaces = nullptr;
    mesh.szboufaces = 0;

    TemplateMalloc(&common.nboufaces, 1 + maxbc*nbe, 0);
    if (nboufaces > 0) {
        int *boufaces = nullptr;
        TemplateMalloc(&boufaces, nboufaces, 0);
        getboundaryfaces(common.nboufaces, boufaces, mesh.bf, common.eblks,
                         nbe, nfe, maxbc, nboufaces);
        TemplateMalloc(&mesh.boufaces, nboufaces, backend);
        TemplateCopytoDevice(mesh.boufaces, boufaces, nboufaces, backend);
        mesh.szboufaces = nboufaces;
        CPUFREE(boufaces);
    }
    else if (common.nboufaces != nullptr) {
        common.nboufaces[0] = 0;
    }
}

template <class T = ::dstype, class I = ::Int>
void AllocateLDGBlockJacobianMemory(resstructT<T,I>& res, commonstructT<T,I>& common, ::Int backend, scratcharenastructT<T,I>& scratch)
{
    using dstype = T; using Int = I;
    Int npe = common.grid.npe;
    Int npf = common.grid.npf;
    Int nfe = common.meshsizes.nfe;
    Int ncu = common.components.ncu;
    Int ncq = common.components.ncq;
    Int nd = common.grid.nd;
    Int ne = common.meshsizes.ne;
    Int neb = common.meshsizes.neb;

    Int n = npe*ncu;
    Int m = npf*nfe*ncu;
    Int nq = npe*ncq;
    Int ndofu = npe*ncu*common.meshsizes.ne1;
    Int M = max(common.solverparams.gmresRestart+1, common.solverparams.RBdim);

    Int tempn_uface = npf*npf*nfe*neb*ncu*(2*ncu + ncq);
    Int tempn_schur = max(n*n*neb, n*m*neb);
    Int szFq = common.grid.ngf*nd*ncu*ncu*common.meshsizes.nfb;
    Int szBufq = npf*npf*nd*ncu*ncu*common.meshsizes.nfb;
    Int szEf = npf*nd*npf*common.meshsizes.nfb;
    Int szAf = npf*npf*ncu*ncu*common.meshsizes.nfb;
    Int tempn_cross = max(szFq, szBufq) + max(szBufq, szEf) + szAf;
    Int hSize = max(max(tempn_schur, tempn_uface), tempn_cross);

    Int kInvSize = n*n*common.meshsizes.ne1;
    Int dSize = n*n*neb;
    Int bSize = n*nq*neb;
    Int fSize = m*n*neb;
    Int kSize = kInvSize + max(dSize + bSize + 2*fSize + hSize, M*ndofu);
    res.szP = kInvSize;

    res.K = scratch.allocate(kSize, backend); res.szK = kSize;  // K owned by the arena (S5 step 3)
    EnsureTemplateAllocation(&res.ipiv, res.szipiv, n*neb, backend);
    if (ncq > 0) {
        EnsureTemplateAllocation(&res.Mass2, res.szMass2, npe*npe*ne, backend);
        EnsureTemplateAllocation(&res.Minv2, res.szMinv2, npe*npe*ne, backend);
        EnsureTemplateAllocation(&res.C, res.szC, npe*npe*ne*nd, backend);
        EnsureTemplateAllocation(&res.E, res.szE, npe*npf*nfe*ne*nd, backend);
    }

    // Assembly views reserved sequentially from the K arena (the shared tail after the
    // preconditioner region szP==kInvSize). Replaces the inline &K[kInvSize + dSize + ...] math.
    res.resetKArena(kInvSize);
    res.D = res.reserveView(dSize);
    res.B = res.reserveView(bSize);
    res.F = res.reserveView(fSize);
    res.G = res.reserveView(fSize);
    res.H = res.reserveView(hSize);
    res.fhAliasesK = 1;  // F and H alias into K here; freememory must not free them (see resstruct)
}

// Both CPU and GPU constructor
template <class T, class I>
CDiscretizationT<T, I>::CDiscretizationT(string filein, string fileout, string exasimpath, Int mpiprocs, Int mpirank, 
        Int fileoffset, Int omprank, Int backend, Int builtinmodelID,
        const ExasimDriverABI& abi, Int nsca, Int nvec, Int nten, Int nsurf, Int nvqoi,
        ExasimExecutionMode mode, const std::vector<dstype>* physicsparamOverride,
        Int saveParaview)
{
    driver_abi = abi;
    common.driver_abi = &driver_abi;  // expose the ABI to the unified templated FEM code (AbiAdapter path)
    common.backend = backend;
    common.exasimpath = exasimpath;
    common.builtinmodelID = builtinmodelID;
    app.builtinmodelID = builtinmodelID;

//     if (mpirank==0) {      
// #ifdef HAVE_TEXT2CODE
//       cout<< "Model Driver = ../Model/ModelDrivers.cpp"<<endl;
// #elif defined(HAVE_BUILTINMODEL)
//       cout<< "Model Driver = ../Model/BuiltIn/BuiltinModelDrivers.cpp"<<endl;
// #else
//       cout<< "Model Driver = ../Model/FrontendGenerated/KokkosDrivers.cpp"<<endl;
// #endif      
//     }

    if (backend>1) { // GPU
#ifdef HAVE_GPU        
        // host structs
        solstruct hsol;
        resstruct hres;
        appstruct happ;
        masterstruct hmaster; 
        meshstruct hmesh;
        tempstruct htmp;    
        commonstruct hcommon;     

        hcommon.backend = backend;
        // The GPU path stages the read into host structs (happ/hcommon) and then
        // copies to device. cpuInit() computes the initial solution on the host
        // (cpuInituDriver), which for the built-in model provider dispatches on
        // builtinmodelID — so the staging structs need it too, otherwise it is 0
        // and the dispatch aborts ("Unknown builtinmodelID=0"). Mirror the member
        // assignment above (the CPU path uses the member app/common directly).
        hcommon.builtinmodelID = builtinmodelID;
        happ.builtinmodelID = builtinmodelID;
        // allocate data for structs in CPU memory
        cpuInit(hsol, hres, happ, driver_abi, hmaster, hmesh, htmp, hcommon, filein, fileout,
                mpiprocs, mpirank, fileoffset, omprank,
                physicsparamOverride);
                
        // copy data from cpu memory to gpu memory
        gpuInit(sol, res, app, driver_abi, master, mesh, tmp, common,
            hsol, hres, happ, hmaster, hmesh, htmp, hcommon);
        app.read_uh = happ.read_uh;
        // Mirror the builtin/multi-model dispatch id onto the device structs: the Kokkos init
        // (and solve) drivers select the model via app.modelnumber (== builtinmodelID for builtin
        // models, see readbinaryfiles.cpp). gpuInit does not copy it, so without this the device
        // dispatch sees modelnumber=0 and aborts ("Unknown builtinmodelID=0 in KokkosInitu").
        app.modelnumber = happ.modelnumber;
        common.modelnumber = hcommon.modelnumber;
        // carry the needs-init signal onto the device solution; the model IC is computed below
        // (post-copy) in the device execution space, so the same Kokkos init path serves GPU too.
        sol.needudginit = hsol.needudginit;
        sol.needodginit = hsol.needodginit;
        sol.needwdginit = hsol.needwdginit;
        if (hmesh.bf != nullptr) {
          TemplateMalloc(&mesh.bf, hcommon.meshsizes.nfe*hcommon.meshsizes.ne, 0);
          for (int i=0; i<hcommon.meshsizes.nfe*hcommon.meshsizes.ne; i++) mesh.bf[i] = hmesh.bf[i];   
        }

       // copy hsol.xcg to sol.xcg for paraview visualization
        sol.szxcg = hsol.szxcg;
        TemplateMalloc(&sol.xcg, sol.szxcg, 0);
        TemplateCopytoHost(sol.xcg, hsol.xcg, sol.szxcg, 0);
        if (common.mpiRank==0) printf("free CPU memory \n");
          
        // release CPU memory
        happ.freememory(1);        
        hmaster.freememory(1);        
        hmesh.freememory(1);        
        hsol.freememory(1);        
        htmp.freememory(1);        
        hres.freememory(1);        
        hcommon.freememory();             
#endif        
    }
    else  {// CPU
        cpuInit(sol, res, app, driver_abi, master, mesh, tmp, common, filein, fileout,
                mpiprocs, mpirank, fileoffset, omprank,
                physicsparamOverride);
    }
    finalizeConstruction(backend, mode, nsca, nvec, nten, nsurf, nvqoi, saveParaview);
}

// Post-init construction tail: derive read_uh, apply vis-count/saveParaview overrides, compute
// geometry + (LDG) mass inverse, and set up the HDG/coupling discretization. Factored out of the
// file constructor so the in-memory (Preprocessed) constructor reuses the identical finalization.
template <class T, class I>
void CDiscretizationT<T, I>::finalizeConstruction(Int backend, ExasimExecutionMode mode,
        Int nsca, Int nvec, Int nten, Int nsurf, Int nvqoi, Int saveParaview)
{
    common.read_uh = app.read_uh;

    // (model initial conditions moved to CResidual::initializeSolution, driven by CSolution
    //  after construction -- the operator initializes its own solution; see solution.h)
    const bool postprocessOnly = (mode == ExasimExecutionMode::Postprocess);
    // Apply caller-supplied visualization field counts whenever provided (>0). The
    // postprocess path passes these from CLI args; the solve path passes them from the
    // pdeapp nsca/nvec/nten keys so external/builtin-library models (gendatain=0, which
    // do not bake the vis counts into datain) can still write ParaView vis inline.
    if (nsca > 0) common.qoiparams.nsca = nsca;
    if (nvec > 0) common.qoiparams.nvec = nvec;
    if (nten > 0) common.qoiparams.nten = nten;
    if (nsurf > 0) common.qoiparams.nsurf = nsurf;
    if (nvqoi > 0) common.qoiparams.nvqoi = nvqoi;
    // Likewise honor the pdeapp saveParaview key on the solve path (external models do
    // not bake app.flag[17] into datain). Only force-enable; never disable a datain that
    // already requested vis.
    if (!postprocessOnly && saveParaview != 0)
        common.qoiparams.saveParaview = saveParaview;

    const bool needsVisualizationConnectivity =
        (common.qoiparams.saveParaview != 0) &&
        (common.qoiparams.nsca + common.qoiparams.nvec + common.qoiparams.nten > 0) &&
        (sol.szxcg == 0 || mesh.szcgelcon == 0 || mesh.szrowent2elem == 0 ||
         mesh.szcgent2dgent == 0 || mesh.szcolent2elem == 0);
    if (needsVisualizationConnectivity) {
        const Int npe = common.grid.npe;
        const Int nd = common.grid.nd;
        const Int ne = common.meshsizes.ne;

        dstype* xcg = nullptr;
        TemplateMalloc(&xcg, npe * nd * ne, 0);
        TemplateMalloc(&mesh.cgelcon, npe * ne, 0);
        const Int ncgnodes = mkelconcg_hashgrid(xcg, mesh.cgelcon, sol.xdg, npe, nd, ne);
        const Int ncgdof = mkent2elem(mesh.rowent2elem, mesh.colent2elem, mesh.cgelcon, npe, ne);
        map_cgent2dgent(mesh.cgent2dgent, mesh.rowent2elem, mesh.colent2elem, xcg, sol.xdg, npe, nd, ncgdof);

        TemplateMalloc(&sol.xcg, nd * ncgnodes, 0);
        for (Int i = 0; i < nd * ncgnodes; i++)
            sol.xcg[i] = xcg[i];
        sol.szxcg = nd * ncgnodes;

        mesh.szcgelcon = mesh.nsize[11] = npe * ne;
        mesh.szrowent2elem = mesh.nsize[12] = ncgdof + 1;
        mesh.szcgent2dgent = mesh.nsize[13] = mesh.rowent2elem[ncgdof];
        mesh.szcolent2elem = mesh.nsize[14] = mesh.rowent2elem[ncgdof];

        CPUFREE(xcg);
    }

    // compute the geometry quantities
    if (common.mpiRank==0) printf("start compGeometry... \n");
    compGeometry(backend);        
    if (common.mpiRank==0) printf("finish compGeometry... \n");        

    // compute the inverse of the mass matrix
    if (common.spatialScheme == 0) {
        if (common.mpiRank==0) printf("start compMassInverse... \n");
        compMassInverse(backend);    
        if (common.mpiRank==0) printf("finish compMassInverse... \n");
        if (!postprocessOnly && common.solverparams.preconditioner == 1) {
            if (common.mpiRank==0) printf("start qEquation... \n");
            BuildElementBlockBoundaryFaces(common, mesh, backend);        
            AllocateLDGBlockJacobianMemory(res, common, backend, scratch);
            qEquation(sol, res, app, master, mesh, tmp, common, backend);
            TemplateFree(res.Mass2, backend);
            TemplateFree(res.Minv2, backend);
            res.szMass2 = 0;
            res.szMinv2 = 0;
            if (common.mpiRank==0) printf("finish qEquation... \n");
        }
        else if (!postprocessOnly) {
            res.szP = 0;
            Int ndofu = common.grid.npe*common.components.ncu*common.meshsizes.ne1;
            Int M = max(common.solverparams.gmresRestart+1, common.solverparams.RBdim);
            res.K = scratch.allocate(M*ndofu, backend); res.szK = M*ndofu;  // K owned by the arena (S5 step 3)
        }
        else {
            res.szP = 0;
        }
    }
    
    // (LDG initial-q recovery moved to CResidual::recoverInitialState, called by CSolution)

    // Optional: validate the batched DGProjection path on this backend/rank.
    if (getenv("EXASIM_TEST_PROJECTION") != nullptr)
        projectionSelfTest(backend);
    // Optional: validate the 2D->3D extrusion kernels on this backend/rank.
    if (getenv("EXASIM_TEST_EXTRUDE") != nullptr)
        extrusionSelfTest(backend);

    if (common.spatialScheme > 0)  { // HDG
      Int neb = common.meshsizes.neb; // maximum number of elements per block
      Int npe = common.grid.npe; // number of nodes on master element
      Int npf = common.grid.npf; // number of nodes on master face
      Int nfe = common.meshsizes.nfe; // number of faces on master element
      Int ne = common.meshsizes.ne; // number of elements in this subdomain
      Int nf = common.meshsizes.nf; // number of faces in this subdomain
      Int ncx = common.components.ncx; // number of compoments of (xdg)
      Int nc = common.components.nc; // number of compoments of (u, q)
      Int ncu = common.components.ncu; // number of compoments of (u)
      Int ncq = common.components.ncq; // number of compoments of (q)      
      Int nbe = common.meshsizes.nbe; // number of blocks for elements
      int ncu12 = common.szinterfacefluxmap;
      
      if (common.mpiRank==0) 
        printf("Init HDG Discretization ... \n");        
      
      int nboufaces = 0; // number of boundary faces
      int maxbc = 0; // maximum number of boundary conditions
      for (int i=0; i<nfe*ne; i++) {
        if (mesh.bf[i] > 0) nboufaces++;
        maxbc = max(maxbc, mesh.bf[i]);
      }
      common.meshsizes.maxnbc = maxbc;      
      
      if (common.couplingparams.coupledboundarycondition>0) {
        //common.couplingparams.nintfaces = getinterfacefaces(mesh.bf, common.eblks, nbe-1, nfe, common.couplingparams.coupledboundarycondition);
        common.couplingparams.nintfaces = getinterfacefaces(mesh.bf, nfe, common.meshsizes.ne1, common.couplingparams.coupledboundarycondition);
        int *intfaces = nullptr; // store interface faces
        TemplateMalloc(&intfaces, common.couplingparams.nintfaces, 0);
        //getinterfacefaces(intfaces, mesh.bf, common.eblks, nbe-1, nfe, common.couplingparams.coupledboundarycondition, common.couplingparams.nintfaces);
        getinterfacefaces(intfaces, mesh.bf, nfe, common.meshsizes.ne1, common.couplingparams.coupledboundarycondition, common.couplingparams.nintfaces);
        TemplateMalloc(&mesh.intfaces, common.couplingparams.nintfaces, common.backend);
        TemplateCopytoDevice(mesh.intfaces, intfaces, common.couplingparams.nintfaces, common.backend);                       
        mesh.szintfaces = common.couplingparams.nintfaces;
        
        CPUFREE(intfaces);
        
        TemplateMalloc(&sol.xdgint, ncx*npf*common.couplingparams.nintfaces, common.backend);
        GetBoudaryNodes(sol.xdgint, sol.xdg, mesh.intfaces, mesh.perm, nfe, npf, npe, ncx, ncx, common.couplingparams.nintfaces);
        sol.szxdgint = ncx*npf*common.couplingparams.nintfaces;                 
      }
      
      // GetBoudaryNodes(xdgb.data(), &sol.xdg[0], &mesh.boufaces[start], mesh.perm, nfe, npf, npe, ncx, ncx, nfaces);

      if (common.mpiRank==0) 
        printf("Maximum number of boundary conditions = %d \n", maxbc);        

      // print2iarray(mesh.bf, nfe, ne);
      // print2iarray(common.eblks, 3, nbe);

      int *boufaces = nullptr; // store boundary faces
      TemplateMalloc(&common.nboufaces, 1 + maxbc*nbe, 0);
      TemplateMalloc(&boufaces, nboufaces, 0);
      getboundaryfaces(common.nboufaces, boufaces, mesh.bf, common.eblks, nbe, nfe, maxbc, nboufaces);
      TemplateMalloc(&mesh.boufaces, nboufaces, common.backend);
      TemplateCopytoDevice(mesh.boufaces, boufaces, nboufaces, common.backend);                       
      mesh.szboufaces = nboufaces;

      CPUFREE(boufaces);
      //CPUFREE(mesh.bf);            
                          
      if (!postprocessOnly && (common.solverparams.preconditioner==2) && (common.szcartgridpart > 0)) {              
        if (common.cartgridpart[0]==2) {          
          int *elem = NULL;                
          int nse  = gridpartition2d(&elem, common.cartgridpart[1], common.cartgridpart[2], common.cartgridpart[3], common.cartgridpart[4], common.cartgridpart[5]);       
          int nese = common.cartgridpart[3]*common.cartgridpart[4];    
          crs_init(common, mesh, elem, nse, nese);
          CPUFREE(elem);
        }
        else if (common.cartgridpart[0]==3) {
          int *elem = NULL;   
          int nse  = gridpartition3d(&elem, common.cartgridpart[1], common.cartgridpart[2], common.cartgridpart[3], common.cartgridpart[4], common.cartgridpart[5], common.cartgridpart[6], common.cartgridpart[7]);       
          int nese = common.cartgridpart[4]*common.cartgridpart[5]*common.cartgridpart[6];      
          crs_init(common, mesh, elem, nse, nese);
          CPUFREE(elem);
        }                               
      }
      
      if (!postprocessOnly) {
        res.szH = npf*nfe*ncu*npf*nfe*ncu*common.meshsizes.ne; // HDG elemental matrices     
        res.szK = (npe*ncu*npe*ncu + npe*ncu*npe*ncq + npf*nfe*ncu*npe*ncq + npf*nfe*ncu*npe*ncu)*neb;                        
        if (common.solverparams.preconditioner==0)      // Block Jacobition preconditioner
          res.szP = ncu*npf*ncu*npf*nf;
        else if (common.solverparams.preconditioner==1) // Elemental additive Schwarz preconditioner
          res.szP = npf*nfe*ncu*npf*nfe*ncu*common.meshsizes.ne;        
        else if (common.solverparams.preconditioner==2) // Superelement additive Schwarz preconditioner
          res.szP = npf*ncu*npf*ncu*common.nse*common.nnz;        
        res.szV = ncu*npf*nf*(common.solverparams.gmresRestart+1); // Krylov vectors in GMRES
        res.szK = max(res.szK, res.szP + res.szV);              
        res.szF = npe*ncu*npf*nfe*ncu*common.meshsizes.ne;      
        res.szipiv = max(max(npf*nfe,npe)*ncu*neb, ncu*npf*common.meshsizes.nfb);
              
        TemplateMalloc(&res.H, res.szH, backend);
        res.K = scratch.allocate(res.szK, backend);  // K owned by the arena (S5 step 3)
        TemplateMalloc(&res.F, res.szF, backend);
        TemplateMalloc(&res.ipiv, res.szipiv, backend); // fix big here
        res.fhAliasesK = 0;  // HDG: H/F/K each owned -> freememory frees all three
              
        // B, D, G share the K block (and it also holds the preconditioner matrix + sys.v
        // Krylov vectors -- see resstruct). Assembly views reserved sequentially after the
        // leading npf*nfe*ncu*npe*ncu*neb block; replaces the inline &K[off + ...] math.
        res.resetKArena(npf*nfe*ncu*npe*ncu*neb);
        res.D = res.reserveView(npe*ncu*npe*ncu*neb);
        res.B = res.reserveView(npe*ncu*npe*ncq*neb);
        res.G = res.reserveView(npf*nfe*ncu*npe*ncq*neb);
        
        if (common.couplingparams.coupledinterface>0) {
          res.szRi = npf*ncu12*common.couplingparams.ncie;
          res.szKi = npf*ncu12*npe*ncu*common.couplingparams.ncie;
          res.szHi = npf*ncu12*npf*nfe*ncu*common.couplingparams.ncie;
          TemplateMalloc(&res.Ri, res.szRi, backend);
          TemplateMalloc(&res.Ki, res.szKi, backend);
          TemplateMalloc(&res.Hi, res.szHi, backend);
        }           
      }

      if (common.mpiRank==0) 
        printf("Memory allocation ...\n");        

      // (HDG operator-state recovery -- uh via GetFaceNodes, q-matrices via qEquation, q via
      //  hdgGetQ -- moved to CResidual::recoverInitialState, called by CSolution post-init)
    }

    // Wall-model build is a host-only, double-geometry concern (CWallModel + CPointLocator run on
    // the double mesh). It is dead for non-default precision (a float solve never uses it), so guard
    // the whole branch on T==dstype so CWallModel is not instantiated at float.
    if constexpr (std::is_same_v<T, ::dstype> && std::is_same_v<I, ::Int>) {
      if (common.wallmodelparams.nwm == 1) {
        if (common.mpiRank==0)
          printf("Build wall-model data for boundary condition %d ... \n", common.wallmodelparams.wmBoundaries[0]);
        CWallModel(*this).build(common.wallmodelparams.wmBoundaries[0], common.wallmodelparams.wmDistances[0]);
      }
      else if (common.wallmodelparams.nwm > 1) {
        error("Multiple wall-model configurations are not supported by the backend wallmodelstruct yet.");
      }
    }

    if (common.mpiRank==0) {
      if (common.outputparams.debugMode==1) {
        common.printinfo();
        app.printinfo();
        res.printinfo();
        tmp.printinfo();
        sol.printinfo();
        mesh.printinfo();
        master.printinfo();
      }
      
      printf("finish CDiscretization constructor... \n");        
    }
}

 
// (BuildWallModelData moved to CWallModel::build -- see wallmodelbuild.cpp)

// destructor
template <class T, class I>
CDiscretizationT<T, I>::~CDiscretizationT()
{        
    app.freememory(common.backend);
    if (common.mpiRank==0) printf("CDiscretization destructor: app memory is freed successfully.\n");
    master.freememory(common.backend);
    if (common.mpiRank==0) printf("CDiscretization destructor: master memory is freed successfully.\n");
    mesh.freememory(common.backend);
    if (common.mpiRank==0) printf("CDiscretization destructor: mesh memory is freeed successfully.\n");
    sol.freememory(common.backend);
    if (common.mpiRank==0) printf("CDiscretization destructor: sol memory is freed successfully.\n");
    tmp.freememory(common.backend);
    if (common.mpiRank==0) printf("CDiscretization destructor: tmp memory is freed successfully.\n");
    res.freememory(common.backend);
    if (common.mpiRank==0) printf("CDiscretization destructor: res memory is freed successfully.\n");
    scratch.freememory(common.backend);  // owns the K backing buffer (res.K was a non-owning view) -- S5 step 3
    wallmodel.freememory(common.backend);
    if (common.mpiRank==0) printf("CDiscretization destructor: wallmodel memory is freed successfully.\n");
    common.freememory();
    if (common.mpiRank==0) printf("CDiscretization destructor: common memory is freed successfully.\n");

#ifdef HAVE_CUDA    
    if (common.backend==2) {
        CHECK(cudaEventDestroy(common.eventHandle));
        CHECK_CUBLAS(cublasDestroy(common.cublasHandle));
    }
#endif    
    
#ifdef HAVE_HIP    
    if (common.backend==3) {
        CHECK(hipEventDestroy(common.eventHandle));
        CHECK_HIPBLAS(hipblasDestroy(common.cublasHandle));
    }
#endif        
}

// Compute and store the geometry
template <class T, class I>
void CDiscretizationT<T, I>::compGeometry(Int backend) {
    if (common.mpiRank==0) printf("start ElemGeom... \n");
    ElemGeom(sol, master, mesh, tmp, common, common.cublasHandle, backend);   
    if (common.mpiRank==0) printf("Finish ElemGeom... \n");
    FaceGeom(sol, master, mesh, tmp, common, common.cublasHandle, backend);   

    ElemFaceGeom(sol, master, mesh, tmp, common, common.cublasHandle, backend);
}

// Compute and store the inverse of the mass matrix
template <class T, class I>
void CDiscretizationT<T, I>::compMassInverse(Int backend) {
    ComputeMinv(sol, res, app, master, mesh, tmp, common, common.cublasHandle, backend);
}

// Batched L2 projection between nodal bases -- thin wrapper over the backend
// primitive so callers get the CPU/CUDA/HIP + MPI (element-local) path.
template <class T, class I>
void CDiscretizationT<T, I>::projectField(dstype* U1, dstype* U, dstype* shapegs,
        Int npe_s, Int nc, Int backend) {
    DGProjection(U1, U, shapegs, npe_s, nc, sol, res, app, master, mesh, tmp, common, common.cublasHandle, backend);
}

// On-device validation of DGProjection: an identity projection (source basis ==
// target basis, so shapegs is the target master's Gauss-point shape values) must
// reproduce a real field to ~machine precision. This exercises the whole batched
// pipeline -- geometry/jac, ShapJac, Gauss2Node, Inverse, ArrayGemmBatch1, the
// straight/curved dispatch, device memory and the blas handle -- on the active
// backend, and independently on every MPI rank (the projection is element-local).
// The C != M *math* is validated separately on the host analytic oracle
// (backend/Discretization/dgprojection_backend_test.cpp).
template <class T, class I>
void CDiscretizationT<T, I>::projectionSelfTest(Int backend) {
    Int npe = common.grid.npe;
    Int ncx = common.components.ncx;
    Int ne  = common.meshsizes.ne;   // rank-local element count
    if (ne <= 0 || npe <= 0 || ncx <= 0) return;
    Int N = npe * ne;                // nc = 1 (project the x-coordinate field)

    dstype *U=nullptr, *U1=nullptr, *hd=nullptr, *hu=nullptr;
    TemplateMalloc(&U,  N, backend);
    TemplateMalloc(&U1, N, backend);
    // U = component 0 of xdg  ->  a real, spatially varying field on the target basis
    ArrayExtract(U, sol.xdg, npe, ncx, ne, 0, npe, 0, 1, 0, ne);

    // identity projection: source basis == target basis (shapegs = master.shapegt values block)
    projectField(U1, U, master.shapegt, npe, 1, backend);

    // U1 <- U1 - U   (should be ~0 elementwise)
    ArrayAXPBY(U1, U1, U, (dstype)1.0, (dstype)(-1.0), N);

    // check per rank on the host (validates each rank's element-local projection)
    TemplateMalloc(&hd, N, 0);
    TemplateMalloc(&hu, N, 0);
    TemplateCopytoHost(hd, U1, N, backend);
    TemplateCopytoHost(hu, U,  N, backend);
    dstype emax = (dstype)0, umax = (dstype)0;
    for (Int i = 0; i < N; i++) {
        dstype a = hd[i] < 0 ? -hd[i] : hd[i];
        dstype b = hu[i] < 0 ? -hu[i] : hu[i];
        if (a > emax) emax = a;
        if (b > umax) umax = b;
    }
    dstype relerr = (umax > (dstype)0) ? emax / umax : emax;
    // For identity (source basis == target basis) C == M exactly, so
    // U1 = M^{-1}(M U) and the residual is bounded by the mass-matrix
    // conditioning kappa(M)*eps -- machine precision on straight elements, but
    // meaningfully larger on curved high-order elements where kappa(M) is large.
    // Gate the straight path tightly (a real stride/dispatch bug shows up as an
    // O(1) error there); allow the curved path the conditioning slack. A real
    // bug is O(1) and trips either gate.
    dstype tol = (common.grid.curvedMesh == 0) ? (dstype)1e-9 : (dstype)1e-3;
    printf("[rank %d] DGProjection identity self-test: elems=%d relerr=%.3e (curvedMesh=%d backend=%d) -> %s\n",
           (int)common.mpiRank, (int)ne, (double)relerr, (int)common.grid.curvedMesh, (int)backend,
           (relerr < tol) ? "PASS" : "FAIL");

    TemplateFree(hd, 0);
    TemplateFree(hu, 0);
    TemplateFree(U,  backend);
    TemplateFree(U1, backend);

    if (!(relerr < tol))
        error("DGProjection self-test FAILED (relerr exceeds tolerance)");
}

// On-device validation of the 2D->3D extrusion kernels. Extrusion is a pure
// data-parallel index/gather op with no mesh dependency, so this fabricates a
// synthetic 2D field, extrudes it on the active backend (CPU/CUDA/HIP), and
// checks the result on the host, independently per MPI rank:
//   (1) ExtrudeSolution gather: fill U2d[k]=k, then every 3D entry must equal
//       the 2D flat index it gathers from (exact).
//   (2) ExtrudeVelocity rotation: vr==1 => vx^2+vy^2==1 everywhere (exercises
//       the device cos/sin + coordinate path).
template <class T, class I>
void CDiscretizationT<T, I>::extrusionSelfTest(Int backend) {
    const Int np2d = 6, nc = 2, ne2d = 4, porder = 2, nz = 3;
    const Int np1d = porder + 1;
    const Int N2 = np2d * nc * ne2d;
    const Int N3nodes = np2d * np1d;
    const Int NE3 = ne2d * nz;
    const Int N3 = N3nodes * nc * NE3;
    const double PI = 3.14159265358979323846;

    // host inputs
    dstype *hu2=nullptr, *hvr=nullptr, *htt=nullptr, *hplc=nullptr;
    TemplateMalloc(&hu2, N2, 0);
    TemplateMalloc(&hvr, N2, 0);
    TemplateMalloc(&htt, nz + 1, 0);
    TemplateMalloc(&hplc, np1d, 0);
    for (Int k = 0; k < N2; k++) { hu2[k] = (dstype)k; hvr[k] = (dstype)1; }
    for (Int e = 0; e <= nz; e++) htt[e] = (dstype)(e * (PI / 2));           // 0, pi/2, pi, 3pi/2
    for (Int d = 0; d < np1d; d++) hplc[d] = (dstype)d / (dstype)(np1d - 1); // 0, .5, 1

    // device buffers
    dstype *U2=nullptr, *U3=nullptr, *vr=nullptr, *tt=nullptr, *plc=nullptr, *vx=nullptr, *vy=nullptr;
    TemplateMalloc(&U2, N2, backend);  TemplateCopytoDevice(U2, hu2, N2, backend);
    TemplateMalloc(&vr, N2, backend);  TemplateCopytoDevice(vr, hvr, N2, backend);
    TemplateMalloc(&tt, nz + 1, backend); TemplateCopytoDevice(tt, htt, nz + 1, backend);
    TemplateMalloc(&plc, np1d, backend);  TemplateCopytoDevice(plc, hplc, np1d, backend);
    TemplateMalloc(&U3, N3, backend);
    TemplateMalloc(&vx, N3, backend);
    TemplateMalloc(&vy, N3, backend);

    ExtrudeSolution(U3, U2, (int)np2d, (int)nc, (int)ne2d, (int)np1d, (int)nz);
    ExtrudeVelocity(vx, vy, vr, tt, plc, (int)np2d, (int)nc, (int)ne2d, (int)np1d, (int)nz);

    // pull back
    dstype *h3=nullptr, *hvx=nullptr, *hvy=nullptr;
    TemplateMalloc(&h3, N3, 0);
    TemplateMalloc(&hvx, N3, 0);
    TemplateMalloc(&hvy, N3, 0);
    TemplateCopytoHost(h3, U3, N3, backend);
    TemplateCopytoHost(hvx, vx, N3, backend);
    TemplateCopytoHost(hvy, vy, N3, backend);

    // (1) gather map must be exact
    Int gmax = 0;
    for (Int idx = 0; idx < N3; idx++) {
        Int n3 = idx % N3nodes, r = idx / N3nodes, b = r % nc, e3 = r / nc;
        Int a = n3 % np2d, c = e3 % ne2d;
        Int expect = a + np2d * (b + nc * c);
        Int diff = (Int)h3[idx] - expect; if (diff < 0) diff = -diff;
        if (diff > gmax) gmax = diff;
    }
    // (2) rotation: vx^2 + vy^2 == 1 (vr == 1)
    dstype vmax = (dstype)0;
    for (Int idx = 0; idx < N3; idx++) {
        dstype e = hvx[idx] * hvx[idx] + hvy[idx] * hvy[idx] - (dstype)1;
        if (e < 0) e = -e; if (e > vmax) vmax = e;
    }

    // Gather must be exact (integer index copy). The rotation identity
    // vx^2+vy^2==1 is a cos/sin round-off check, so its tolerance must scale with
    // the build precision: ~1e-16 in double, ~1e-7 in single (USE_FLOAT) -- the
    // builtin consumer runs both a double and a single-precision model.
    const dstype rot_tol = (sizeof(dstype) < 8) ? (dstype)1e-4 : (dstype)1e-10;
    bool ok = (gmax == 0) && (vmax < rot_tol);
    printf("[rank %d] Extrude self-test: 3Delems=%d gather_maxmiss=%d rot_err=%.3e prec=%dB (backend=%d) -> %s\n",
           (int)common.mpiRank, (int)NE3, (int)gmax, (double)vmax, (int)sizeof(dstype), (int)backend, ok ? "PASS" : "FAIL");

    TemplateFree(hu2, 0); TemplateFree(hvr, 0); TemplateFree(htt, 0); TemplateFree(hplc, 0);
    TemplateFree(h3, 0); TemplateFree(hvx, 0); TemplateFree(hvy, 0);
    TemplateFree(U2, backend); TemplateFree(U3, backend); TemplateFree(vr, backend);
    TemplateFree(tt, backend); TemplateFree(plc, backend); TemplateFree(vx, backend); TemplateFree(vy, backend);

    if (!ok) error("Extrude self-test FAILED");
}

// ComputeLDGPreconditioner re-homed to CPreconditioner (C4).

// (hdgAssembleLinearSystem / hdgAssembleResidual moved to CAssembler -- see assembler.cpp)

// residual evaluation
// (evalResidual / evalQ / evalQSer / evalAVfield / updateUDG / updateU moved to CResidual -- see residualeval.cpp)

// evalOutput re-homed to CSolution (S4).


// evalMonitor re-homed to CSolution (S4).

template <class T, class I>
void CDiscretizationT<T, I>::DG2CG(dstype* ucg, dstype* udg, dstype *utm, Int ncucg, Int ncudg, Int ncu, Int backend)
{
    for (Int i=0; i<ncu; i++) {
        // extract the ith component of udg and store it in utm
        ArrayExtract(utm, udg, common.grid.npe, ncudg, common.meshsizes.ne, 0, common.grid.npe, i, i+1, 0, common.meshsizes.ne);
        
        // make it a CG field and store in res.Ru
        ArrayDG2CG(res.Ru, utm, mesh.cgent2dgent, mesh.rowent2elem, common.sizes.ndofucg);
        
        // convert CG field to DG field
        GetArrayAtIndex(utm, res.Ru, mesh.cgelcon, common.grid.npe*common.meshsizes.ne);
        
        // insert utm into ucg
        ArrayInsert(ucg, utm, common.grid.npe, ncucg, common.meshsizes.ne, 0, common.grid.npe, i, i+1, 0, common.meshsizes.ne);
    }
}

template <class T, class I>
void CDiscretizationT<T, I>::DG2CG2(dstype* ucg, dstype* udg, dstype *utm, Int ncucg, Int ncudg, Int ncu, Int backend)
{
    for (Int i=0; i<ncu; i++) {
        // extract the ith component of udg and store it in utm
        ArrayExtract(utm, udg, common.grid.npe, ncudg, common.meshsizes.ne, 0, common.grid.npe, i, i+1, 0, common.meshsizes.ne);

        // make it a CG field and store in res.Ru
        ArrayDG2CG2(res.Ru, utm, mesh.colent2elem, mesh.rowent2elem, common.sizes.ndofucg, common.grid.npe);
        
        // convert CG field to DG field
        GetArrayAtIndex(utm, res.Ru, mesh.cgelcon, common.grid.npe*common.meshsizes.ne);
        
        // insert utm into ucg
        ArrayInsert(ucg, utm, common.grid.npe, ncucg, common.meshsizes.ne, 0, common.grid.npe, i, i+1, 0, common.meshsizes.ne);
    }
}

template <class T, class I>
void CDiscretizationT<T, I>::DG2CG3(dstype* ucg, dstype* udg, dstype *utm, Int ncucg, Int ncudg, Int ncu, Int backend)
{
    for (Int i=0; i<ncu; i++) {
        // extract the ith component of udg and store it in utm
        ArrayExtract(utm, udg, common.grid.npe, ncudg, common.meshsizes.ne, 0, common.grid.npe, i, i+1, 0, common.meshsizes.ne);
        
        // make it a CG field and store in res.Ru
        ArrayDG2CG(&ucg[i*common.sizes.ndofucg], utm, mesh.cgent2dgent, mesh.rowent2elem, common.sizes.ndofucg);
    }
}

// interface/boundary sampling methods (moved out of CDiscretization)
#include "interfacesampler.cpp"
// wall-model build (moved out of CDiscretization)
#include "wallmodelbuild.cpp"

// ---- Phase 2: explicit instantiation of the backend-defined members (default precision) ----------
// These members (file ctor, dtor, geometry/mass/DG2CG) are DEFINED in this TU (part of the unity
// ExasimSolver.cpp build). Other separately-compiled TUs -- notably main.cpp, whose CSolution<M>
// inline ctor/dtor construct and destroy a `CDiscretization disc` by value -- use them through the
// (now templated) CDiscretization alias but never see these out-of-line definitions. Instantiate
// them once here; the matching `extern template` declarations in discretization.h suppress implicit
// instantiation in those other TUs so they link to this single definition. See
// docs/internals/precision-threading.md (Phase 2).
template CDiscretizationT<::dstype, ::Int>::CDiscretizationT(
    std::string, std::string, std::string, Int, Int, Int, Int, Int, Int, const ExasimDriverABI&,
    Int, Int, Int, Int, Int, ExasimExecutionMode, const std::vector<dstype>*, Int);
template CDiscretizationT<::dstype, ::Int>::~CDiscretizationT();
template void CDiscretizationT<::dstype, ::Int>::finalizeConstruction(
    Int, ExasimExecutionMode, Int, Int, Int, Int, Int, Int);
template void CDiscretizationT<::dstype, ::Int>::compGeometry(Int);
template void CDiscretizationT<::dstype, ::Int>::compMassInverse(Int);
template void CDiscretizationT<::dstype, ::Int>::projectField(dstype*, dstype*, dstype*, Int, Int, Int);
template void CDiscretizationT<::dstype, ::Int>::projectionSelfTest(Int);
template void CDiscretizationT<::dstype, ::Int>::extrusionSelfTest(Int);
template void CDiscretizationT<::dstype, ::Int>::DG2CG(dstype*, dstype*, dstype*, Int, Int, Int, Int);
template void CDiscretizationT<::dstype, ::Int>::DG2CG2(dstype*, dstype*, dstype*, Int, Int, Int, Int);
template void CDiscretizationT<::dstype, ::Int>::DG2CG3(dstype*, dstype*, dstype*, Int, Int, Int, Int);

#endif
