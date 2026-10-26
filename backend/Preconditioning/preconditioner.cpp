/*
    preconditioner.cpp

    This file implements the CPreconditioner class and related functions for preconditioning linear systems
    in the Exasim backend. The preconditioner is used to accelerate the convergence of iterative solvers
    by transforming the original system into a more favorable form.

    Main Components:
    ----------------
    - CPreconditioner::CPreconditioner(CDiscretization& disc, Int backend)
        Constructor: Initializes the preconditioner structure and prints info if in debug mode.

    - CPreconditioner::~CPreconditioner()
        Destructor: Frees memory allocated for the preconditioner.

    - CPreconditioner::ComputeInitialGuessAndPreconditioner(sysstruct& sys, solverstatestruct& state, CAssembler<M, T, I>& assembler, CDiscretization& disc, Int backend)
        Computes the initial guess and sets up the preconditioner using reduced basis techniques.
        Involves matrix-vector multiplications and small matrix inversions.

    - CPreconditioner::ApplyPreconditioner(dstype* x, sysstruct& sys, CDiscretization& disc, Int backend)
        Applies the preconditioner to the input vector x, storing the result in disc.res.Ru.

    - CPreconditioner::ComputeInitialGuessAndPreconditioner(sysstruct& sys, solverstatestruct& state, CAssembler<M, T, I>& assembler, CDiscretization& disc, Int N, Int spatialScheme, Int backend)
        Variant of the initial guess and preconditioner computation supporting different spatial schemes.

    - ApplyBlockILU0(double* x, double* A, double* b, double *B, double *C, commonstruct& common)
        Implements block ILU(0) factorization and solve for super-element additive Schwarz preconditioner.
        Performs forward and backward solves with block matrices.

    - CPreconditioner::ApplyPreconditioner(dstype* x, sysstruct& sys, CDiscretization& disc, Int spatialScheme, Int backend)
        Applies the preconditioner based on the spatial scheme and preconditioner type:
            - Block Jacobi
            - Elemental additive Schwarz
            - Super-element additive Schwarz with block ILU(0)

    Dependencies:
    -------------
    - setprecondstruct.cpp: Sets up the preconditioner structure.
    - applymatrix.cpp: Applies matrix operations.
    - Various BLAS-like routines (PGEMTV, PGEMTM, PGEMNV, PGEMNMStridedBached).
    - Utility functions (ArrayCopy, ArraySetValue, SubtractColumns, etc.).

    Notes:
    ------
    - The code supports both CPU and GPU backends.
    - Uses Kokkos for synchronization and parallelism.
    - Handles both sequential and non-sequential block solves for efficiency.
    - Designed for high-performance scientific computing applications.

*/
#ifndef __PRECONDITIONER
#define __PRECONDITIONER

#include "preconditioner.h"
#include "setprecondstruct.cpp"
#include "applymatrix.cpp"
#include "../Discretization/stagetables.hpp"

// Solve the small RB normal system without a fatal inverse. Equilibration
// makes the rank test independent of stored update magnitudes. All arrays are
// host-accessible, and work needs n entries.
template <class T, class I>
bool SolveReducedBasisGram(T* H, T* rhs, T* work, I n)
{
    if (n <= 0) return false;
    for (I i=0; i<n; ++i) {
        const T d = H[i+n*i];
        if (!is_finite_bitwise(d) || d <= T(0) || !is_finite_bitwise(rhs[i]))
            return false;
        work[i] = std::sqrt(d);
    }
    for (I j=0; j<n; ++j) {
        rhs[j] /= work[j];
        for (I i=j; i<n; ++i) {
            H[i+n*j] = (H[i+n*j]/work[i])/work[j];
            if (!is_finite_bitwise(H[i+n*j])) return false;
        }
    }
    // Normal equations square the condition number. Reject directions that
    // have lost roughly half the working precision to linear dependence.
    const T pivotTolerance = std::sqrt(std::numeric_limits<T>::epsilon());
    for (I j=0; j<n; ++j) {
        T pivot = H[j+n*j];
        for (I k=0; k<j; ++k) pivot -= H[j+n*k]*H[j+n*k];
        if (!is_finite_bitwise(pivot) || pivot <= pivotTolerance) return false;
        H[j+n*j] = std::sqrt(pivot);
        for (I i=j+1; i<n; ++i) {
            T a = H[i+n*j];
            for (I k=0; k<j; ++k) a -= H[i+n*k]*H[j+n*k];
            H[i+n*j] = a/H[j+n*j];
        }
    }
    for (I i=0; i<n; ++i) {
        for (I j=0; j<i; ++j) rhs[i] -= H[i+n*j]*rhs[j];
        rhs[i] /= H[i+n*i];
    }
    for (I k=n; k>0; --k) {
        const I i = k-1;
        for (I j=i+1; j<n; ++j) rhs[i] -= H[j+n*i]*rhs[j];
        rhs[i] /= H[i+n*i];
    }
    for (I i=0; i<n; ++i) {
        rhs[i] /= work[i];
        if (!is_finite_bitwise(rhs[i])) return false;
    }
    return true;
}

template <class T, class I>
bool ReducedBasisCoefficients(sysstructT<T,I>& sys, T* coefficients, T* H,
        T* rhs, solverstatestruct& state, commonstructT<T,I>& common, I N,
        Int backend)
{
    const I m = state.RBcurrentdim;
    T* hostH = sys.tempmem;
    T* hostRhs = hostH + m*m;
    T* scales = hostRhs + m;
    ArrayCopy(common.cublasHandle, hostH, H, m*m, backend);
    ArrayCopy(common.cublasHandle, hostRhs, rhs, m, backend);
    Kokkos::fence();
    int usable = SolveReducedBasisGram(hostH, hostRhs, scales, m) ? 1 : 0;
#ifdef HAVE_MPI
    // Ranks must take the same fallback, including on heterogeneous hosts.
    int allUsable;
    MPI_Allreduce(&usable, &allUsable, 1, MPI_INT, MPI_MIN, EXASIM_COMM_WORLD);
    usable = allUsable;
#endif
    if (!usable) {
        state.RBcurrentdim = 0;
        state.RBremovedind = 0;
        ArraySetValue(sys.x, T(0), N);
        if (common.mpiRank == 0)
            std::cout << "Rejected singular or non-finite RB basis; using a zero GMRES guess." << std::endl;
        return false;
    }
    // Complete this small transfer before GMRES may reuse the host workspace.
    TemplateCopytoDevice(coefficients, hostRhs, m, backend);
    return true;
}


// constructor
template <class M, class T, class I>
CPreconditioner<M, T, I>::CPreconditioner(CDiscretization& disc, Int backend, ExasimExecutionMode mode)
{
    mpiRank = disc.common.mpiRank;
    precond.backend = backend;
    if (mode == ExasimExecutionMode::Postprocess) {
        if (disc.common.mpiRank == 0) printf("finish CPreconditioner constructor (postprocess mode)... \n");
        return;
    }

    setprecondstruct(precond, disc, backend);    
    if ((disc.common.mpiRank==0) && (disc.common.outputparams.debugMode==1)) precond.printinfo();
    if (disc.common.mpiRank == 0) printf("finish CPreconditioner constructor... \n");    
}

// destructor
template <class M, class T, class I>
CPreconditioner<M, T, I>::~CPreconditioner()
{            
    precond.freememory(precond.backend);
    if (mpiRank==0) printf("CPreconditioner destructor: precond memory is freed successfully.\n");
}

template <class M, class T, class I>
void CPreconditioner<M, T, I>::ComputeInitialGuessAndPreconditioner(sysstruct& sys, solverstatestruct& state, CAssembler<M, T, I>& assembler, CDiscretization& disc, Int backend)
{       
    // RB supplies a warm start, not a modification of ApplyPreconditioner.
    // Older J*W columns are deliberately recycled as an approximate predictor.
    // linearSolve checks its current nonlinear residual before GMRES uses it.
    
    Int N = disc.common.sizes.ndof1;
    Int RBdim = state.RBcurrentdim;
    dstype *RBcoef = &disc.tmp.tempn[0];
    dstype *RBcoef_tmp = &disc.tmp.tempn[RBdim];
    dstype *H = &disc.tmp.tempg[0];
    dstype *H_tmp = &disc.tmp.tempg[RBdim*RBdim];
    
    // compute V = J(u)*W        
    //Int i = RBdim-1;
    int i = state.RBremovedind - 1;
    if (state.RBremovedind==0) i = RBdim-1;
    assembler.evalMatVecWithNorm(&precond.U[i*N], &precond.W[i*N], sys.u, sys.b, one, backend);
    
    //assembler.evalMatVec(view_1d Jv, view_1d v, view_1d u, view_1d Ru, Int backend);
    
    /* RBcoef_tmp = V^T b */
    PGEMTV(disc.common.cublasHandle, N, RBdim, &one, precond.U, N, sys.b, inc1, 
            &zero, RBcoef_tmp, inc1, H_tmp, backend);
    
    /* H = V^T V */
    PGEMTM(disc.common.cublasHandle, RBdim, RBdim, N, &one, precond.U, N, precond.U, N, &zero, 
            H, RBdim, H_tmp, backend);    
    
    if (!ReducedBasisCoefficients(sys, RBcoef, H, RBcoef_tmp, state, disc.common, N, backend))
        return;
        
    // compute x = -W*RBcoef
    PGEMNV(disc.common.cublasHandle, N, RBdim, &minusone, precond.W, N, RBcoef, 
            inc1, &zero, sys.x, inc1, backend);                                                 
}

template <class M, class T, class I>
void CPreconditioner<M, T, I>::ApplyPreconditioner(dstype* x, sysstruct& sys, CDiscretization& disc, Int backend)
{        
    Int N = disc.common.sizes.ndof1;        

    if ((disc.common.spatialScheme == 0) && (disc.common.solverparams.preconditioner == 1) && (disc.res.K != nullptr)) {
        const PrecondStageTable* t = PrecondStagesOf(disc.common);
        if (t != nullptr && t->apply != nullptr) {
            ResidualStageContext c{disc.sol, disc.res, disc.app, disc.master, disc.mesh, disc.tmp, disc.common,
                                   disc.common.cublasHandle, backend};
            t->apply(c, x);
            return;
        }
    }
    ArrayCopy(disc.common.cublasHandle, disc.res.Ru, x, N, backend);
    if ((disc.common.spatialScheme == 0) && (disc.common.solverparams.preconditioner == 1) && (disc.res.K != nullptr)) {
        Int n = disc.common.grid.npe*disc.common.components.ncu;
        Int ne = disc.common.meshsizes.ne1;
        PGEMNMStridedBached(disc.common.cublasHandle, n, 1, n, one,
                disc.res.K, n, disc.res.Ru, n, zero, x, n, ne, backend);
    }
    else 
        ApplyMatrix(disc.common.cublasHandle, x, disc.res.Minv, disc.res.Ru,
            disc.common.grid.npe, disc.common.components.ncu, disc.common.meshsizes.ne1,
            disc.common.solverparams.preconditioner, disc.common.grid.curvedMesh, backend);
}

template <class M, class T, class I>
void CPreconditioner<M, T, I>::ComputeInitialGuessAndPreconditioner(sysstruct& sys, solverstatestruct& state, CAssembler<M, T, I>& assembler, CDiscretization& disc, Int N, Int spatialScheme, Int backend)
{     
    if (spatialScheme == 0) {
        ComputeInitialGuessAndPreconditioner(sys, state, assembler, disc, backend);
        return;
    }
    Int RBdim = state.RBcurrentdim;
    dstype *RBcoef = &disc.tmp.tempn[0];
    dstype *RBcoef_tmp = &disc.tmp.tempn[RBdim];
    dstype *H = &disc.tmp.tempg[0];
    dstype *H_tmp = &disc.tmp.tempg[RBdim*RBdim];
    
    // compute V = J(u)*W        
    //Int i = RBdim-1;
    int i = state.RBremovedind - 1;
    if (state.RBremovedind==0) i = RBdim-1;
    assembler.evalMatVec(&precond.U[i*N], &precond.W[i*N], sys.u, sys.b, spatialScheme, backend);
    
    //assembler.evalMatVec(view_1d Jv, view_1d v, view_1d u, view_1d Ru, Int backend);
    
    /* RBcoef_tmp = V^T b */
    PGEMTV(disc.common.cublasHandle, N, RBdim, &one, precond.U, N, sys.b, inc1, 
            &zero, RBcoef_tmp, inc1, H_tmp, backend);
    
    /* H = V^T V */
    PGEMTM(disc.common.cublasHandle, RBdim, RBdim, N, &one, precond.U, N, precond.U, N, &zero, 
            H, RBdim, H_tmp, backend);    
    
    if (!ReducedBasisCoefficients(sys, RBcoef, H, RBcoef_tmp, state, disc.common, N, backend))
        return;
        
    // compute x = alpha * W * RBcoef
    dstype alpha = (spatialScheme == 0) ? minusone : one;
    PGEMNV(disc.common.cublasHandle, N, RBdim, &alpha, precond.W, N, RBcoef, 
            inc1, &zero, sys.x, inc1, backend);                                                 
}

void ApplyBlockILU0(double* x, double* A, double* b, double *B, double *C, commonstruct& common) 
{    
    Int nfe = common.meshsizes.nfe; 
    Int ncu = common.components.ncu;// number of compoments of (u)
    Int npf = common.grid.npf; // number of nodes on master face           
    Int ncf = ncu*npf;    
    Int nse = common.nse;
    Int nfse = common.nfse;
    Int Q = ncf*nse;
    Int N = ncf*ncf*nse;    
    Int M = 2 * (nfe - 1);
    
    // Forward solve: L*y = b (unit diagonal)
    for (int i = 0; i < nfse; ++i) {
//         double *yi = &b[Q*i];
//         int k = common.bjindex.Lnum_ji[0 + i*2];
//         for (int l = 0; l < k; ++l) {
//             int ptr = common.bjindex.Lind_ji[l + 0*M + i*2*M];
//             int j   = common.bjindex.Lind_ji[l + 1*M + i*2*M];                
//             PGEMNMStridedBached(common.cublasHandle, ncf, 1, ncf, minusone, &A[ptr*N], ncf, &b[Q*j], ncf, one, yi, ncf, nse, common.backend);         
//         }
      
        double *yi = &b[Q*i];                        
        int k = common.bjindex.Lnum_ji[0 + i*2];
        int flag_seq = common.bjindex.Lnum_ji[1 + i*2];
        
        if (flag_seq == 1 && k > 1) {
                      
            for (int l = 0; l < k; ++l) {
              int ji = common.bjindex.Lind_ji[l + 1*M + i*2*M];
              ArrayCopy(&B[Q*l], &b[Q*ji], Q);
            }                        
            
            int p1 = common.bjindex.Lind_ji[0 + 0*M + i*2*M];
            PGEMNMStridedBached(common.cublasHandle, ncf, 1, ncf, one, &A[N*p1], ncf, B, ncf, zero, C, ncf, nse*k, common.backend);         
            
            SubtractColumns(yi, C, Q, k);                        
        } else {          
            // Non-sequential: direct pointer access to y(:,:,j)
            for (int l = 0; l < k; ++l) {
                int ptr = common.bjindex.Lind_ji[l + 0*M + i*2*M];
                int j   = common.bjindex.Lind_ji[l + 1*M + i*2*M];                
                PGEMNMStridedBached(common.cublasHandle, ncf, 1, ncf, minusone, &A[ptr*N], ncf, &b[Q*j], ncf, one, yi, ncf, nse, common.backend);         
            }
        }
    }

    // Backward solve: U*x = y
    for (int i = nfse-1; i >= 0; --i) {
//         double *yi = &b[Q*i];
//         int k = common.bjindex.Unum_ji[0 + i*3];
//         int rstart = common.bjindex.Unum_ji[1 + i*3];
//         for (int l = 0; l < k; ++l) {
//             int ptr = common.bjindex.Uind_ji[l + 0*M + i*2*M];
//             int j   = common.bjindex.Uind_ji[l + 1*M + i*2*M];                                
//             PGEMNMStridedBached(common.cublasHandle, ncf, 1, ncf, minusone, &A[ptr*N], ncf, &x[Q*j], ncf, one, yi, ncf, nse, common.backend);         
//         }
      
        double *yi = &b[Q*i];
        int k = common.bjindex.Unum_ji[0 + i*3];
        int rstart = common.bjindex.Unum_ji[1 + i*3];
        int flag_seq = common.bjindex.Unum_ji[2 + i*3];

        if (flag_seq == 1 && k > 1) {
                        
            for (int l = 0; l < k; ++l) {
              int ji = common.bjindex.Uind_ji[l + 1*M + i*2*M];
              ArrayCopy(&B[Q*l], &x[Q*ji], Q);
            }                        
            
            int p1 = common.bjindex.Uind_ji[0 + 0*M + i*2*M];
            PGEMNMStridedBached(common.cublasHandle, ncf, 1, ncf, one, &A[N*p1], ncf, B, ncf, zero, C, ncf, nse*k, common.backend);         
            
            SubtractColumns(yi, C, Q, k);                  
        } else {
            // Non-sequential: direct pointer access to x(:,:,j)
            for (int l = 0; l < k; ++l) {
                int ptr = common.bjindex.Uind_ji[l + 0*M + i*2*M];
                int j   = common.bjindex.Uind_ji[l + 1*M + i*2*M];                                
                PGEMNMStridedBached(common.cublasHandle, ncf, 1, ncf, minusone, &A[ptr*N], ncf, &x[Q*j], ncf, one, yi, ncf, nse, common.backend);         
            }
        }
        
        // Final diagonal solve for node i: x(:,n,i) = A(:,:,n,rstart) * yi(:,n) for all n
        PGEMNMStridedBached(common.cublasHandle, ncf, 1, ncf, one, &A[rstart*N], ncf, yi, ncf, zero, &x[i*Q], ncf, nse, common.backend);         
    }
}

template <class M, class T, class I>
void CPreconditioner<M, T, I>::ApplyPreconditioner(dstype* x, sysstruct& sys, CDiscretization& disc, Int spatialScheme, Int backend)
{                
    if (spatialScheme==0) {
      Int N = disc.common.sizes.ndof1;        
      ArrayCopy(disc.common.cublasHandle, disc.res.Ru, x, N, backend);
      ApplyMatrix(disc.common.cublasHandle, x, disc.res.Minv, disc.res.Ru, disc.common.grid.npe, disc.common.components.ncu, 
          disc.common.meshsizes.ne1, disc.common.solverparams.precMatrixType, disc.common.grid.curvedMesh, backend);                
    }
    else if (spatialScheme==1) {      
      if (disc.common.solverparams.preconditioner==0) { // Block Jacobi preconditioner
        Int nf = disc.common.meshsizes.nf; // number of faces in this subdomain
        Int ncu = disc.common.components.ncu;// number of compoments of (u)
        Int npf = disc.common.grid.npf; // number of nodes on master face           
        Int ncf = ncu*npf;  

        ArrayCopy(disc.common.cublasHandle, disc.res.Rh, x, ncf*nf, backend);

        // (ncf)  * (ncf) * nf x (ncf) * nf -> (ncf) * nf
        PGEMNMStridedBached(disc.common.cublasHandle, ncf, 1, ncf, one, disc.res.K, ncf, disc.res.Rh, ncf, zero, x, ncf, nf, backend);         
      }
      else if (disc.common.solverparams.preconditioner==1) { // Elemental additive Schwarz preconditioner        
        hdgMatVec(x, disc.res.K, x, disc.res.Rh, disc.res.Rq, disc.res, disc.app, disc.mesh, disc.common, disc.tmp, disc.common.cublasHandle, backend,
                  disc.getNonmatchingInteractions()); // [nonmatching] same coupling as the conformal preconditioner (decision D1)
      }
      else if (disc.common.solverparams.preconditioner==2) { // super-element additive Schwarz preconditioner with BLIU0
        Int nf = disc.common.meshsizes.nf; // number of faces in this subdomain
        Int ncu = disc.common.components.ncu;// number of compoments of (u)
        Int npf = disc.common.grid.npf; // number of nodes on master face           
        Int ncf = ncu*npf;          
        Int nfe = disc.common.meshsizes.nfe;
        Int nse = disc.common.nse;
        Int nfse = disc.common.nfse;
        
        //writearray2file(disc.common.fileout + "b.bin", x, ncf*nf, backend);  
        
        GetCollumnAtIndex(disc.res.Rq, x, disc.mesh.face, ncf, nse*nfse);
        
//         writearray2file(disc.common.fileout + "b1.bin", disc.res.Rq, ncf*nse*nfse, backend);  
//         
//         writearray2file(disc.common.fileout + "K.bin", disc.res.K, ncf*ncf*nse*disc.common.nnz, backend);  
//         
        ApplyBlockILU0(disc.tmp.tempn, disc.res.K, disc.res.Rq, disc.tmp.tempg, &disc.tmp.tempn[ncf*nse*nfse], disc.common); 
        
//        writearray2file(disc.common.fileout + "b2.bin", disc.tmp.tempn, ncf*nse*nfse, backend);  
        
        ArraySetValue(x, zero, ncf*nf);
        PutCollumnAtIndexAtomicAdd(x, disc.tmp.tempn, disc.mesh.face, ncf, nse*nfse);

#ifdef HAVE_MPI
        if (disc.common.mpiProcs > 1) {
          GetElementFaceNodesAtFaces(disc.tmp.buffsend, x, disc.mesh.elemcon, disc.mesh.bilufacesend,
                  npf, nfe, ncu, disc.common.nbilufacesend);

#ifdef HAVE_CUDA
          cudaDeviceSynchronize();
#endif
#ifdef HAVE_HIP
          hipDeviceSynchronize();
#endif

          Int neighbor, nsend, nrecv, psend = 0, precv = 0, request_counter = 0;
          for (int n=0; n<disc.common.nnbsd; n++) {
            neighbor = disc.common.nbsd[n];
            nsend = disc.common.bilufacesendpts[n]*ncf;
            if (nsend > 0) {
              MPI_Isend(&disc.tmp.buffsend[psend], nsend, mpi_type<dstype>(), neighbor, 0,
                    EXASIM_COMM_LOCAL, &disc.common.requests[request_counter]);
              psend += nsend;
              request_counter += 1;
            }
          }

          for (int n=0; n<disc.common.nnbsd; n++) {
            neighbor = disc.common.nbsd[n];
            nrecv = disc.common.bilufacerecvpts[n]*ncf;
            if (nrecv > 0) {
              MPI_Irecv(&disc.tmp.buffrecv[precv], nrecv, mpi_type<dstype>(), neighbor, 0,
                    EXASIM_COMM_LOCAL, &disc.common.requests[request_counter]);
              precv += nrecv;
              request_counter += 1;
            }
          }

          if (request_counter > 0)
            MPI_Waitall(request_counter, disc.common.requests, disc.common.statuses);

          PutElementFaceNodesAtFaces(x, disc.tmp.buffrecv, disc.mesh.elemcon, disc.mesh.bilufacerecv,
                  npf, nfe, ncu, disc.common.nbilufacerecv);
        }
#endif       
      }
    }
}

// Re-homed from CDiscretization (C4): the LDG block-Jacobi preconditioner matrix is computed
// from the discretization's structs but is a preconditioner responsibility. BlockJacobianLDG /
// mpiBlockJacobianLDG are visible here via the unity build (discretization.cpp, which includes
// ldgblockjacobian.cpp, precedes preconditioner.cpp in ExasimSolver.cpp).
template <class M, class T, class I>
void CPreconditioner<M, T, I>::ComputeLDGPreconditioner(CDiscretization& disc, dstype* K, dstype* u, Int backend)
{
#ifdef HAVE_MPI
    if (disc.common.mpiProcs > 1) {
        mpiBlockJacobianLDG(K, u, disc.sol, disc.res, disc.app, disc.driver_abi, disc.master, disc.mesh, disc.tmp,
                disc.common, disc.common.cublasHandle, backend);
        return;
    }
#endif
    BlockJacobianLDG(K, u, disc.sol, disc.res, disc.app, disc.driver_abi, disc.master, disc.mesh, disc.tmp, disc.common,
            disc.common.cublasHandle, backend);
}

// Build the HDG preconditioner matrix K from the assembled global system H.
// (Re-homed from CAssembler::hdgAssembleLinearSystem: assembly builds H, the preconditioner
//  builds K from H -- two separate concerns. Called after the assembler, before GMRES.)
template <class M, class T, class I>
void CPreconditioner<M, T, I>::ComputeHDGPreconditioner(CDiscretization& disc, Int backend)
{
    auto& res = disc.res; auto& mesh = disc.mesh; auto& tmp = disc.tmp; auto& common = disc.common;

    if (common.solverparams.preconditioner==0) {
      // fix bug here: tmp.tempn is not enough memory to store ncu*npf*ncu*npf*nf
      hdgBlockJacobi(res.K, res.H, res, mesh, tmp, common, common.cublasHandle, backend);
    }
    else if (common.solverparams.preconditioner==1) {
      hdgElementalAdditiveSchwarz(res.K, res.H, res, mesh, tmp, common, common.cublasHandle, backend);
    }
    else if (common.solverparams.preconditioner==2) {
      hdgBlockILU0(res.K, res.H, res, mesh, tmp, common, common.cublasHandle, backend);
    }
}

#endif
