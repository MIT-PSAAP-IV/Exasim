/**
 * @file pblas.h
 * @brief Parallel BLAS (Basic Linear Algebra Subprograms) interface for CPU, CUDA, HIP, and MPI backends.
 *
 * This header provides a unified interface for linear algebra operations across multiple hardware backends,
 * including CPU (BLAS/LAPACK), CUDA (cuBLAS), HIP (hipBLAS), and distributed memory (MPI).
 *
 * Supported operations include:
 *  - Matrix-vector and matrix-matrix multiplications (GEMM, GEMV)
 *  - Dot products and norms
 *  - Array copy, scaling, and AXPY/AXPBY operations
 *  - Batched matrix inversion
 *  - Node-to-Gauss and Gauss-to-Node transformations for finite element methods
 *
 * Backend selection:
 *  - backend <= 1: CPU BLAS/LAPACK
 *  - backend == 2: CUDA cuBLAS
 *  - backend == 3: HIP hipBLAS
 *
 * Precision selection:
 *  - USE_FLOAT: single precision (float)
 *  - otherwise: double precision (double)
 *
 * MPI support:
 *  - If HAVE_MPI is defined, global reductions are performed using MPI_Allreduce.
 *
 * CUDA/HIP support:
 *  - If HAVE_CUDA or HAVE_HIP is defined, GPU routines are enabled.
 *
 * Function naming conventions:
 *  - Functions prefixed with 'cpu' are CPU-specific.
 *  - Functions prefixed with 'gpu' or 'hip' are CUDA/HIP-specific.
 *  - Functions without prefix are generic and dispatch to the appropriate backend.
 *
 * Thread safety:
 *  - Most functions are stateless and thread-safe, except for those using global handles.
 *
 * Error handling:
 *  - CUDA/HIP routines use CHECK_CUBLAS or CHECK_HIPBLAS macros for error checking.
 *
 * @author Exasim project contributors
 * @date 2024
 */
#ifndef __PBLAS_H__
#define __PBLAS_H__

// ------------------------------------------------------------------------------------------------
// blas<T>: precision-dispatched CPU BLAS/LAPACK (Phase 3 of dstype->template threading, see
// docs/internals/precision-threading.md). Replaces the per-function `#ifdef USE_FLOAT S.. #else D..`
// branches with a by-value, type-dispatched interface so the pblas wrappers below can be templated
// on the scalar type T. The primary template is left undefined -> an unsupported precision is a
// compile error; `float`/`double` map to the s*/d* LAPACK symbols (the SGEMM/DGEMM/... macros from
// common.h). Under the default T=dstype this selects EXACTLY the routine the old #ifdef did, so the
// emitted call is identical (byte-for-byte numerics). The GPU (cublas/hipblas) branches are still
// selected by the compile-time USE_FLOAT macro for now -- correct under the default T=dstype;
// trait-ifying them for non-default GPU precision is the remote-verified tail of this phase.
template <class T> struct blas;   // primary: unsupported precision -> compile error

template <> struct blas<double> {
    static void gemm(char ta, char tb, Int m, Int n, Int k, double al, const double* A, Int lda,
                     const double* B, Int ldb, double be, double* C, Int ldc)
        { DGEMM(&ta, &tb, &m, &n, &k, &al, const_cast<double*>(A), &lda, const_cast<double*>(B), &ldb, &be, C, &ldc); }
    static void gemv(char t, Int m, Int n, double al, const double* A, Int lda, const double* x,
                     Int incx, double be, double* y, Int incy)
        { DGEMV(&t, &m, &n, &al, const_cast<double*>(A), &lda, const_cast<double*>(x), &incx, &be, y, &incy); }
    static void getrf(Int m, Int n, double* A, Int lda, Int* ipiv, Int& info) { DGETRF(&m, &n, A, &lda, ipiv, &info); }
    static void getri(Int n, double* A, Int lda, const Int* ipiv, double* work, Int lwork, Int& info)
        { DGETRI(&n, A, &lda, const_cast<Int*>(ipiv), work, &lwork, &info); }
};
template <> struct blas<float> {
    static void gemm(char ta, char tb, Int m, Int n, Int k, float al, const float* A, Int lda,
                     const float* B, Int ldb, float be, float* C, Int ldc)
        { SGEMM(&ta, &tb, &m, &n, &k, &al, const_cast<float*>(A), &lda, const_cast<float*>(B), &ldb, &be, C, &ldc); }
    static void gemv(char t, Int m, Int n, float al, const float* A, Int lda, const float* x,
                     Int incx, float be, float* y, Int incy)
        { SGEMV(&t, &m, &n, &al, const_cast<float*>(A), &lda, const_cast<float*>(x), &incx, &be, y, &incy); }
    static void getrf(Int m, Int n, float* A, Int lda, Int* ipiv, Int& info) { SGETRF(&m, &n, A, &lda, ipiv, &info); }
    static void getri(Int n, float* A, Int lda, const Int* ipiv, float* work, Int lwork, Int& info)
        { SGETRI(&n, A, &lda, const_cast<Int*>(ipiv), work, &lwork, &info); }
};

template <class T = dstype>
static void cpuNode2Gauss(T *ug, T *un, T *shapt, Int ng, Int np, Int nn)
{
    blas<T>::gemm('N', 'N', ng, nn, np, T(1), shapt, ng, un, np, T(0), ug, ng);
}


template <class T = dstype>
static void Node2Gauss(cublasHandle_t handle, T *ug, T *un, T *shapt, Int ng, Int np, Int nn, Int backend)
{
    if (backend <= 1)
        blas<T>::gemm('N', 'N', ng, nn, np, T(1), shapt, ng, un, np, T(0), ug, ng);

#ifdef HAVE_CUDA
#ifdef USE_FLOAT  
    if (backend == 2)     
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ng, nn, np, 
            cublasOne, shapt, ng, un, np, cublasZero, ug, ng);    
#else            
    if (backend == 2)  
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ng, nn, np, 
            cublasOne, shapt, ng, un, np, cublasZero, ug, ng);
#endif        
#endif        
    
#ifdef HAVE_HIP
#ifdef USE_FLOAT  
    if (backend == 3) 
        hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, ng, nn, np, 
            &one, shapt, ng, un, np, &zero, ug, ng);    
#else
    if (backend == 3) 
        hipblasDgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, ng, nn, np, 
            &one, shapt, ng, un, np, &zero, ug, ng);
#endif        
#endif        
}

template <class T = dstype>
static void Gauss2Node(cublasHandle_t handle, T *un, T *ug, T *shapg, Int ng, Int np, Int nn, Int backend)
{
    if (backend <= 1)
        blas<T>::gemm('N', 'N', np, nn, ng, T(1), shapg, np, ug, ng, T(0), un, np);

#ifdef HAVE_CUDA
#ifdef USE_FLOAT  
    if (backend == 2)     
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, np, nn, ng, 
            cublasOne, shapg, np, ug, ng, cublasZero, un, np);    
#else            
    if (backend == 2)  
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, np, nn, ng, 
            cublasOne, shapg, np, ug, ng, cublasZero, un, np);
#endif        
#endif             
    
#ifdef HAVE_HIP
#ifdef USE_FLOAT
    if (backend == 3)
        hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, np, nn, ng, 
            &one, shapg, np, ug, ng, &zero, un, np);
#else
    if (backend == 3)
        hipblasDgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, np, nn, ng, 
            &one, shapg, np, ug, ng, &zero, un, np);
#endif        
#endif
}

template <class T = dstype>
static void Gauss2Node1(cublasHandle_t handle, T *un, T *ug, T *shapg, Int ng, Int np, Int nn, Int backend)
{
    if (backend <= 1)
        blas<T>::gemm('N', 'N', np, nn, ng, T(1), shapg, np, ug, ng, T(1), un, np);

#ifdef HAVE_CUDA          
#ifdef USE_FLOAT  
    if (backend == 2)     
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, np, nn, ng, 
            cublasOne, shapg, np, ug, ng, cublasOne, un, np);    
#else            
    if (backend == 2)  
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, np, nn, ng, 
            cublasOne, shapg, np, ug, ng, cublasOne, un, np);
#endif        
#endif             
    
#ifdef HAVE_HIP
#ifdef USE_FLOAT
    if (backend == 3)
        hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, np, nn, ng, 
            &one, shapg, np, ug, ng, &one, un, np);
#else
    if (backend == 3)
        hipblasDgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, np, nn, ng, 
            &one, shapg, np, ug, ng, &one, un, np);
#endif        
#endif    
}

#ifdef HAVE_CUDA       
static void gpuComputeInverse(cublasHandle_t handle, dstype* A, dstype *C, Int n, Int batchSize)
{
    Int *ipiv, *info;
    cudaMalloc(&ipiv, n * batchSize * sizeof(Int));
    cudaMalloc(&info,  batchSize * sizeof(Int));     
    
    //dstype *C;
    //CHECK(cudaMalloc(&C,  n*n*batchSize * sizeof(dstype)));        
    
    dstype **Ap_h = (dstype **)malloc(batchSize*sizeof(dstype *));
    dstype **Ap_d;
    cudaMalloc(&Ap_d,batchSize*sizeof(dstype *));
    Ap_h[0] = A;
    for (Int i = 1; i < batchSize; i++)
      Ap_h[i] = Ap_h[i-1]+(n*n);
    cudaMemcpy(Ap_d,Ap_h,batchSize*sizeof(dstype *),cudaMemcpyHostToDevice);
    
    dstype **Cp_h = (dstype **)malloc(batchSize*sizeof(dstype *));
    dstype **Cp_d;
    cudaMalloc(&Cp_d,batchSize*sizeof(dstype *));
    Cp_h[0] = C;
    for (Int i = 1; i < batchSize; i++)
      Cp_h[i] = Cp_h[i-1] + (n*n);
    cudaMemcpy(Cp_d,Cp_h,batchSize*sizeof(dstype *),cudaMemcpyHostToDevice);
        
#ifdef USE_FLOAT        
    cublasSgetrfBatched(handle,n,Ap_d,n,ipiv,info,batchSize);    
    cublasSgetriBatched(handle,n,(const dstype **)Ap_d,n,ipiv,Cp_d,n,info,batchSize);
#else            
    cublasDgetrfBatched(handle,n,Ap_d,n,ipiv,info,batchSize);    
    cublasDgetriBatched(handle,n,(const dstype **)Ap_d,n,ipiv,Cp_d,n,info,batchSize);    
#endif            
        
    // copy C to A
    ArrayCopy(A, C, n*n*batchSize);
            
    cudaFree(Ap_d); free(Ap_h);
    cudaFree(Cp_d); free(Cp_h);
    cudaFree(ipiv); cudaFree(info); 
}
#endif     

#ifdef HAVE_HIP
static void hipComputeInverse(cublasHandle_t handle, dstype* A, dstype* C, Int n, Int batchSize)
{    
    Int *ipiv, *info;
    CHECK(hipMalloc(&ipiv, n * batchSize * sizeof(Int)));
    CHECK(hipMalloc(&info, batchSize * sizeof(Int)));     
    
    // Allocate host and device pointer arrays
    dstype **Ap_h = (dstype **)malloc(batchSize * sizeof(dstype *));
    dstype **Ap_d;
    CHECK(hipMalloc(&Ap_d, batchSize * sizeof(dstype *)));
    Ap_h[0] = A;
    for (Int i = 1; i < batchSize; i++)
        Ap_h[i] = Ap_h[i-1] + (n * n);
    CHECK(hipMemcpy(Ap_d, Ap_h, batchSize * sizeof(dstype *), hipMemcpyHostToDevice));
    
    dstype **Cp_h = (dstype **)malloc(batchSize * sizeof(dstype *));
    dstype **Cp_d;
    CHECK(hipMalloc(&Cp_d, batchSize * sizeof(dstype *)));
    Cp_h[0] = C;
    for (Int i = 1; i < batchSize; i++)
        Cp_h[i] = Cp_h[i-1] + (n * n);
    CHECK(hipMemcpy(Cp_d, Cp_h, batchSize * sizeof(dstype *), hipMemcpyHostToDevice));
            
#ifdef USE_FLOAT        
    hipblasSgetrfBatched(handle, n, Ap_d, n, ipiv, info, batchSize);    
    hipblasSgetriBatched(handle, n, Ap_d, n, ipiv, Cp_d, n, info, batchSize);
#else            
    CHECK_HIPBLAS(hipblasDgetrfBatched(handle, n, Ap_d, n, ipiv, info, batchSize));    
    CHECK_HIPBLAS(hipblasDgetriBatched(handle, n, Ap_d, n, ipiv, Cp_d, n, info, batchSize));    
#endif            
            
    // Copy C back to A
    ArrayCopy(A, C, n * n * batchSize);
                
    // Cleanup
    CHECK(hipFree(Ap_d)); free(Ap_h);
    CHECK(hipFree(Cp_d)); free(Cp_h);
    CHECK(hipFree(ipiv)); CHECK(hipFree(info));     
}
#endif  

template <class T = dstype>
static void cpuComputeInverse(T* A, T* work, Int* ipiv, Int n)
{
    // LAPACK GETRI requires workspace of at least max(1,n).  Some callers,
    // such as the polynomial-preconditioner setup, only provide O(n) scratch.
    // Advertising n*n workspace here lets GETRI write past the supplied buffer
    // and corrupt neighboring solver memory.
    Int lwork = (n > 0) ? n : 1;
    Int info;
    blas<T>::getrf(n, n, A, n, ipiv, info);
    if (info != 0) {
        printf("GETRF failed in cpuComputeInverse with info = %d and n = %d\n", (int) info, (int) n);
        error("cpuComputeInverse failed during LU factorization.");
    }
    blas<T>::getri(n, A, n, ipiv, work, lwork, info);
    if (info != 0) {
        printf("GETRI failed in cpuComputeInverse with info = %d and n = %d\n", (int) info, (int) n);
        error("cpuComputeInverse failed during matrix inversion.");
    }
}
   
static void Inverse(cublasHandle_t handle, dstype* A, dstype *C, Int *ipiv, Int n, Int batchSize, Int backend)
{    
#ifdef HAVE_CUDA        
    if (backend == 2)   
        gpuComputeInverse(handle, A, C, n, batchSize);
#endif       
    
#ifdef HAVE_HIP        
    if (backend == 3)   
        hipComputeInverse(handle, A, C, n, batchSize);    
#endif           
    
    if (backend <= 1) {        
        for (int k=0; k<batchSize; k++) 
            cpuComputeInverse(&A[n*n*k], C, ipiv, n);                
    }    
}

static void PDOT(cublasHandle_t handle, Int m, dstype* x, Int incx, dstype* y, Int incy, 
        dstype *global_dot, Int backend) 
{           
    dstype local_dot=zero;

#ifdef USE_FLOAT    
    if (backend <= 1) 
        local_dot = SDOT(&m, x, &incx, y, &incy);
#else   
    if (backend <= 1) 
        local_dot = DDOT(&m, x, &incx, y, &incy);
#endif        
    
#ifdef HAVE_CUDA  
#ifdef USE_FLOAT  
    if (backend == 2)     
        cublasSdot(handle, m, x, incx, y, incy, &local_dot);    
#else            
    if (backend == 2)  
        cublasDdot(handle, m, x, incx, y, incy, &local_dot);    
#endif
#endif             
    
#ifdef HAVE_HIP  
#ifdef USE_FLOAT  
    if (backend == 3)     
        hipblasSdot(handle, m, x, incx, y, incy, &local_dot);    
#else            
    if (backend == 3)  
        hipblasDdot(handle, m, x, incx, y, incy, &local_dot);    
#endif
#endif  
    
#ifdef HAVE_MPI        
    MPI_Allreduce(&local_dot, global_dot, 1, mpi_type<dstype>(), MPI_SUM, EXASIM_COMM_WORLD);
#else    
    //ArrayCopy(global_dot, local_dot, 1, backend);
    *global_dot = local_dot;
#endif    
}

static dstype PNORM(cublasHandle_t handle, Int m, dstype* x, Int backend) 
{            
    dstype nrm;
    PDOT(handle, m, x, inc1, x, inc1, &nrm, backend); 
    return sqrt(nrm);    
}

static dstype PNORM(cublasHandle_t handle, Int m, Int n, dstype* x, Int backend) 
{            
    dstype nrm1=0.0, nrm;        
    if (n>0) PDOT(handle, n, x, inc1, x, inc1, &nrm1, backend); 
    PDOT(handle, m-n, &x[n], inc1, &x[n], inc1, &nrm, backend); 
    nrm = nrm + 0.5*nrm1;
    return sqrt(nrm);    
}

static void DOT(cublasHandle_t handle, Int m, dstype* x, Int incx, dstype* y, Int incy, dstype *dot, Int backend) 
{               
#ifdef USE_FLOAT    
    if (backend <= 1) 
        *dot = SDOT(&m, x, &incx, y, &incy);
#else   
    if (backend <= 1) 
        *dot = DDOT(&m, x, &incx, y, &incy);
#endif        
    
#ifdef HAVE_CUDA          
#ifdef USE_FLOAT  
    if (backend == 2)     
        cublasSdot(handle, m, x, incx, y, incy, dot);    
#else            
    if (backend == 2)  
        cublasDdot(handle, m, x, incx, y, incy, dot);    
#endif        
#endif                 
    
#ifdef HAVE_HIP  
#ifdef USE_FLOAT  
    if (backend == 3)     
        hipblasSdot(handle, m, x, incx, y, incy, dot);    
#else            
    if (backend == 3)  
        hipblasDdot(handle, m, x, incx, y, incy, dot);    
#endif
#endif      
}

static void ArrayCopy(cublasHandle_t handle, dstype* y, dstype* x, Int m, Int backend) 
{               
#ifdef USE_FLOAT    
    if (backend <= 1) 
        SCOPY(&m, x, &inc1, y, &inc1);
#else   
    if (backend <= 1) 
        DCOPY(&m, x, &inc1, y, &inc1);
#endif        
    
#ifdef HAVE_CUDA          
#ifdef USE_FLOAT  
    if (backend == 2)     
        cublasScopy(handle, m, x, inc1, y, inc1);    
#else            
    if (backend == 2)  
        cublasDcopy(handle, m, x, inc1, y, inc1);    
#endif        
#endif              
    
#ifdef HAVE_HIP          
#ifdef USE_FLOAT  
    if (backend == 3)     
        hipblasScopy(handle, m, x, inc1, y, inc1);    
#else            
    if (backend == 3)  
        hipblasDcopy(handle, m, x, inc1, y, inc1);    
#endif        
#endif                  
}

static void ArrayMultiplyScalar(cublasHandle_t handle, dstype* x, dstype alpha, Int m, Int backend) 
{               
#ifdef USE_FLOAT    
    if (backend <= 1) 
        SSCAL(&m, &alpha, x, &inc1);
#else   
    if (backend <= 1) 
        DSCAL(&m, &alpha, x, &inc1);
#endif        
    
#ifdef HAVE_CUDA          
#ifdef USE_FLOAT  
    if (backend == 2)     
        cublasSscal(handle, m, &alpha, x, inc1);    
#else            
    if (backend == 2)  
        cublasDscal(handle, m, &alpha, x, inc1);    
#endif        
#endif                 
    
#ifdef HAVE_HIP
#ifdef USE_FLOAT  
    if (backend == 3)     
        hipblasSscal(handle, m, &alpha, x, inc1);    
#else            
    if (backend == 3)  
        hipblasDscal(handle, m, &alpha, x, inc1);    
#endif        
#endif                     
}

//    cublasDaxpy(handle, n, &a, x, 1, z, 1);
static void ArrayAXPY(cublasHandle_t handle, dstype* z, dstype* x, dstype a, Int m, Int backend) 
{
#ifdef USE_FLOAT    
    if (backend <= 1) 
        SAXPY(&m, &a, x, &inc1, z, &inc1);
#else
    if (backend <= 1) 
        DAXPY(&m, &a, x, &inc1, z, &inc1);
#endif

#ifdef HAVE_CUDA
#ifdef USE_FLOAT  
    if (backend == 2)     
        cublasSaxpy(handle, m, &a, x, inc1, z, inc1);
#else
    if (backend == 2)  
        cublasDaxpy(handle, m, &a, x, inc1, z, inc1);
#endif
#endif
    
#ifdef HAVE_HIP
#ifdef USE_FLOAT  
    if (backend == 3)     
        hipblasSaxpy(handle, m, &a, x, inc1, z, inc1);
#else
    if (backend == 3)  
        hipblasDaxpy(handle, m, &a, x, inc1, z, inc1);
#endif
#endif    
}

static void ArrayAXPBY(cublasHandle_t handle, dstype* z, dstype* x, dstype* y, dstype a, dstype b, Int m, Int backend) 
{
    ArrayCopy(handle, z, y, m, backend);
    ArrayMultiplyScalar(handle, z, b, m, backend);
    ArrayAXPY(handle, z, x, a, m, backend);
}

static void ArrayAX(cublasHandle_t handle, dstype* z, dstype* x, dstype a, Int m, Int backend) 
{
    ArrayCopy(handle, z, x, m, backend);
    ArrayMultiplyScalar(handle, z, a, m, backend);    
}

static dstype NORM(cublasHandle_t handle, Int m, dstype* x, Int backend) 
{            
    dstype nrm;
    DOT(handle, m, x, inc1, x, inc1, &nrm, backend); 
    return sqrt(nrm);    
}

template <class T = dstype>
static void PGEMNV(cublasHandle_t handle, Int m, Int n, T* alpha, T* A, Int lda, 
        T* x, Int incx, T* beta, T* y, Int incy, Int backend) 
{
    /* y = alpha*A * x + beta y */
    if (backend <= 1)
        blas<T>::gemv('N', m, n, *alpha, A, lda, x, incx, *beta, y, incy);
    
#ifdef HAVE_CUDA          
#ifdef USE_FLOAT  
    if (backend == 2)     
        cublasSgemv(handle, CUBLAS_OP_N, m, n, alpha, A, lda, x, incx,
                             beta, y, incy);        
#else            
    if (backend == 2)  
        cublasDgemv(handle, CUBLAS_OP_N, m, n, alpha, A, lda, x, incx,
                             beta, y, incy);
#endif        
#endif                     
    
#ifdef HAVE_HIP          
#ifdef USE_FLOAT  
    if (backend == 3)     
        hipblasSgemv(handle, HIPBLAS_OP_N, m, n, alpha, A, lda, x, incx,
                             beta, y, incy);        
#else            
    if (backend == 3)  
        hipblasDgemv(handle, HIPBLAS_OP_N, m, n, alpha, A, lda, x, incx,
                             beta, y, incy);
#endif        
#endif     
}

template <class T = dstype>
static void PGEMTV(cublasHandle_t handle, Int m, Int n, T *alpha, T* A, Int lda, 
        T* x, Int incx, T *beta, T* y, Int incy, T* ylocal, Int backend) 
{
    /* y = alpha*A^T * x + beta y */
    if (backend <= 1)
        blas<T>::gemv('T', m, n, *alpha, A, lda, x, incx, *beta, ylocal, incy);
    
#ifdef HAVE_CUDA          
#ifdef USE_FLOAT  
    if (backend == 2)     
        cublasSgemv(handle, CUBLAS_OP_T, m, n, alpha, A, lda, x, incx,
                             beta, ylocal, incy);        
#else            
    if (backend == 2)  
        cublasDgemv(handle, CUBLAS_OP_T, m, n, alpha, A, lda, x, incx,
                             beta, ylocal, incy);
#endif        
#endif             
    
#ifdef HAVE_HIP          
#ifdef USE_FLOAT  
    if (backend == 3)     
        hipblasSgemv(handle, HIPBLAS_OP_T, m, n, alpha, A, lda, x, incx,
                             beta, ylocal, incy);        
#else            
    if (backend == 3)  
        hipblasDgemv(handle, HIPBLAS_OP_T, m, n, alpha, A, lda, x, incx,
                             beta, ylocal, incy);
#endif        
#endif  
    
#ifdef  HAVE_MPI          
#ifdef HAVE_CUDA
    cudaDeviceSynchronize();
#endif
#ifdef HAVE_HIP
    hipDeviceSynchronize();
#endif        
    MPI_Allreduce(ylocal, y, n, mpi_type<dstype>(), MPI_SUM, EXASIM_COMM_WORLD);
#else
    ArrayCopy(y, ylocal, n);
#endif    
}

// static void PGEMTV2(cublasHandle_t handle, Int m, Int n, dstype *alpha, dstype* A, Int lda, 
//         dstype* x, Int incx, dstype *beta, dstype* y, Int incy, dstype* ylocal, Int backend) 
// {
//     /* y = alpha*A^T * x + beta y */
// #ifdef USE_FLOAT     
//     if (backend <= 1)
//         SGEMM(&chn, &chn, &incx, &n, &m, alpha, x, &incx, A, &lda, beta, ylocal, &incy);    
//         //SGEMV(&cht, &m, &n, alpha, A, &lda, x, &incx, beta, ylocal, &incy);
// #else   
//     if (backend <= 1) 
//         DGEMM(&chn, &chn, &incx, &n, &m, alpha, x, &incx, A, &lda, beta, ylocal, &incy);  
//         //DGEMV(&cht, &m, &n, alpha, A, &lda, x, &incx, beta, ylocal, &incy);
// #endif
//     
// #ifdef HAVE_CUDA          
// #ifdef USE_FLOAT  
//     if (backend == 2)     
//         cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, incx, n, m, 
//             alpha, x, incx, A, lda, beta, ylocal, incy);                    
//         //CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_T, m, n, alpha, A, lda, x, incx,
//         //                     beta, ylocal, incy));        
// #else            
//     if (backend == 2)  
//         cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, incx, n, m, 
//             alpha, x, incx, A, lda, beta, ylocal, incy);                    
//         //CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_T, m, n, alpha, A, lda, x, incx,
//         //                     beta, ylocal, incy));
// #endif        
// #endif             
//     
// #ifdef  HAVE_MPI          
// #ifdef USE_FLOAT         
//     MPI_Allreduce(ylocal, y, n, MPI_FLOAT, MPI_SUM, EXASIM_COMM_WORLD);
// #else            
//     MPI_Allreduce(ylocal, y, n, MPI_DOUBLE, MPI_SUM, EXASIM_COMM_WORLD);
// #endif    
// #else
//     ArrayCopy(y, ylocal, n);
// #endif    
// }

// static void PGEMNM(cublasHandle_t handle, Int m, Int n, Int k, dstype *alpha, dstype* A, Int lda, 
//         dstype* B, Int ldb, dstype *beta, dstype* C, Int ldc, Int backend) 
// {
//     /* C = alpha*A * B + beta C */
// #ifdef USE_FLOAT     
//     if (backend <= 1) 
//         SGEMM(&chn, &chn, &m, &n, &k, alpha, A, &lda, B, &ldb, beta, C, &ldc);
// #else   
//     if (backend <= 1) 
//         DGEMM(&chn, &chn, &m, &n, &k, alpha, A, &lda, B, &ldb, beta, C, &ldc);
// #endif
//     
// #ifdef HAVE_CUDA          
// #ifdef USE_FLOAT  
//     if (backend == 2)     
//         cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 
//             alpha, A, lda, B, ldb, beta, C, ldc);                    
// #else            
//     if (backend == 2)  
//         cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 
//             alpha, A, lda, B, ldb, beta, C, ldc);
// #endif        
// #endif                     
// }

template <class T = dstype>
static void PGEMTM(cublasHandle_t handle, Int m, Int n, Int k, T *alpha, T* A, Int lda, 
        T* B, Int ldb, T *beta, T* C, Int ldc, T* Clocal, Int backend) 
{
    /* C = alpha*A^T * B + beta C */
    if (backend <= 1)
        blas<T>::gemm('T', 'N', m, n, k, *alpha, A, lda, B, ldb, *beta, Clocal, ldc);
    
#ifdef HAVE_CUDA          
#ifdef USE_FLOAT  
    if (backend == 2)     
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, 
            alpha, A, lda, B, ldb, beta, Clocal, ldc);                    
#else            
    if (backend == 2)  
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, 
            alpha, A, lda, B, ldb, beta, Clocal, ldc);
#endif        
#endif                     
    
#ifdef HAVE_HIP          
#ifdef USE_FLOAT  
    if (backend == 3)     
        hipblasSgemm(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, m, n, k, 
            alpha, A, lda, B, ldb, beta, Clocal, ldc);                    
#else            
    if (backend == 3)  
        hipblasDgemm(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, m, n, k, 
            alpha, A, lda, B, ldb, beta, Clocal, ldc);
#endif        
#endif     
    
    Int p = m*n;
    
#ifdef  HAVE_MPI          
#ifdef HAVE_CUDA
    cudaDeviceSynchronize();
#endif
#ifdef HAVE_HIP
    hipDeviceSynchronize();
#endif            
    MPI_Allreduce(Clocal, C, p, mpi_type<dstype>(), MPI_SUM, EXASIM_COMM_WORLD);
#else
    ArrayCopy(C, Clocal, p);
#endif        
}

static void PGEMNMStridedBached(cublasHandle_t handle, Int m, Int n, Int k, dstype alpha, dstype* A, Int lda, 
        dstype* B, Int ldb, dstype beta, dstype* C, Int ldc, Int batchCount, Int backend) 
{
#ifdef USE_FLOAT     
    if (backend <= 1)       
      for (int i=0; i<batchCount; i++)
        SGEMM(&chn, &chn, &m, &n, &k, &alpha, &A[m*k*i], &lda, &B[k*n*i], &ldb, &beta, &C[m*n*i], &ldc);        
#else
    if (backend <= 1) 
      for (int i=0; i<batchCount; i++)
        DGEMM(&chn, &chn, &m, &n, &k, &alpha, &A[m*k*i], &lda, &B[k*n*i], &ldb, &beta, &C[m*n*i], &ldc);
#endif

#ifdef HAVE_CUDA          
#ifdef USE_FLOAT  
    if (backend == 2)     
        CHECK_CUBLAS(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 
            &alpha, A, lda, m*k, B, ldb, k*n, &beta, C, ldc, m*n, batchCount));                    
#else            
    if (backend == 2)  
        CHECK_CUBLAS(cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 
            &alpha, A, lda, m*k, B, ldb, k*n, &beta, C, ldc, m*n, batchCount));
#endif        
#endif                     
    
#ifdef HAVE_HIP          
#ifdef USE_FLOAT  
    if (backend == 3)     
        CHECK_HIPBLAS(hipblasSgemmStridedBatched(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, 
            &alpha, A, lda, m * k, B, ldb, k * n, &beta, C, ldc, m * n, batchCount));                    
#else            
    if (backend == 3)  
        CHECK_HIPBLAS(hipblasDgemmStridedBatched(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, 
            &alpha, A, lda, m * k, B, ldb, k * n, &beta, C, ldc, m * n, batchCount));
#endif        
#endif     
}

#endif  
