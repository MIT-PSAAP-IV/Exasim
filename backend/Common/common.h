/**
 * @file common.h
 * @brief Common definitions, macros, and data structures for Exasim backend.
 *
 * This header provides:
 * - Precision and integer type selection via macros (`USE_FLOAT`, `USE_LONG`).
 * - BLAS/LAPACK function name mappings for single/double precision.
 * - CUDA/HIP/CPU backend abstraction macros and memory management templates.
 * - Utility macros for error checking, memory allocation, and timing.
 * - Bitwise NaN detection functions for float/double.
 * - Data structures for application, master, mesh, solution, residual, temporary, system, preconditioner, and common simulation parameters.
 * - Each struct contains pointers to simulation data, size information, printinfo and freememory methods.
 * - MPI, Kokkos, and optional mutation++/enzyme support.
 *
 * Main structs:
 * - appstruct: Application-level parameters and data arrays.
 * - masterstruct: Master element/face shape functions and quadrature data.
 * - meshstruct: Mesh connectivity and partitioning information.
 * - solstruct: Solution vectors and auxiliary arrays.
 * - resstruct: Residuals, matrices, and pivot arrays.
 * - tempstruct: Temporary buffers for communication and computation.
 * - sysstruct: System vectors for solvers and time-stepping.
 * - precondstruct: Preconditioner matrices and pivots.
 * - commonstruct: Global simulation parameters, arrays, and MPI/CUDA handles.
 *
 * Memory management:
 * - TemplateMalloc/TemplateFree/TemplateCopytoDevice/TemplateCopytoHost for backend-agnostic allocation and transfer.
 * - CPUFREE, GPUFREE, HIPFREE macros for safe pointer deallocation.
 *
 * Error handling:
 * - CHECK, CHECK_CUBLAS, CHECK_HIPBLAS, CHECK_ROCBLAS macros for backend error reporting.
 *
 * Timing:
 * - Macros for timing code blocks, optionally synchronized with CUDA.
 *
 * Usage:
 * - Include this header in backend modules to access common types, macros, and data structures.
 * - Select precision and backend via compile-time macros.
 * - Use provided structs for organizing simulation data and managing memory.
 */
#ifndef __COMMON_H__
#define __COMMON_H__

// Standard headers used directly below (filesystem path/dir helpers, string,
// containers). Included here so common.h is self-contained for consumers that
// reach it without run.hpp's preamble (e.g. <exasim/model.hpp>).
#include "boundscheck.h"   // rank-aware bounds checks (no-op unless EXASIM_BOUNDS_CHECK)
#include <string>
#include <vector>
#include <filesystem>
#include <iomanip>
#include <ostream>
#include <cstdint>
#include <cstring>

#define SCOPY scopy_
#define SSCAL sscal_
#define SAXPY saxpy_
#define SDOT sdot_
#define SGEMV sgemv_
#define SGEMM sgemm_
#define SGETRF sgetrf_
#define SGETRI sgetri_
#define SGEEV sgeev_

#define DCOPY dcopy_
#define DSCAL dscal_
#define DAXPY daxpy_
#define DDOT ddot_
#define DGEMV dgemv_
#define DGEMM dgemm_
#define DGEEV dgeev_
#define DGETRF dgetrf_
#define DGETRI dgetri_

#ifdef USE_FLOAT
typedef float dstype;
#else
typedef double dstype; //  double is default precision 
#endif

#ifdef USE_LONG
typedef long Int;
#else
typedef int Int;
#endif

// Non-deduced type wrapper. A scalar function parameter spelled `noDeduce_t<T>` deduces T ONLY from
// the buffer arguments, so a double literal/constant (0.0, `zero`, `minusone`) passed alongside a
// float* buffer converts to float instead of forcing a conflicting T=double deduction. This is the
// pblas/array-op fix for the precision-template cutover (drivers/kernels/geometry pass dstype
// constants next to T buffers).
template <class U> struct noDeduce_ { using type = U; };
template <class U> using noDeduce_t = typename noDeduce_<U>::type;

#ifdef HAVE_MPI
// mpi_type<T>(): the MPI_Datatype for a scalar/index type (Phase 3 -- the MPI companion to blas<T>).
// Replaces hardcoded MPI_DOUBLE and `#ifdef USE_FLOAT MPI_FLOAT/DOUBLE` branches so MPI halo exchange
// + reductions carry the element type by T. Spelled mpi_type<dstype>() today (byte-identical for the
// default double build); becomes mpi_type<T>() once the comm helpers are templated. NB this also
// FIXES a latent bug: the halo-exchange MPI_Isend/Irecv hardcoded MPI_DOUBLE while the buffers are
// dstype*, so a single-precision (USE_FLOAT) build mis-typed every exchange (float* sent as double).
template <class T> inline MPI_Datatype mpi_type();
template <> inline MPI_Datatype mpi_type<double>() { return MPI_DOUBLE; }
template <> inline MPI_Datatype mpi_type<float>()  { return MPI_FLOAT;  }
template <> inline MPI_Datatype mpi_type<int>()    { return MPI_INT;    }
template <> inline MPI_Datatype mpi_type<long>()   { return MPI_LONG;   }
#endif

#ifndef HAVE_CUDA
#ifdef HAVE_HIP    
#define cublasHandle_t hipblasHandle_t
#define cudaEvent_t hipEvent_t
#else
#define cublasHandle_t int
#define cudaEvent_t int
#endif        
#endif

// #ifdef HAVE_ENZYME                
// template <typename... Args>
// void __enzyme_autodiff(void*, Args... args);
// void __enzyme_fwddiff(void*, ...);
// int enzyme_const, enzyme_dup;
// #endif

typedef Kokkos::View<int*, Kokkos::HostSpace> view_1ih;
typedef Kokkos::View<dstype*, Kokkos::HostSpace> view_1dh;
typedef Kokkos::View<int*> view_1i;
typedef Kokkos::View<dstype*> view_1d;

#ifdef HAVE_MPP
#include <mutation++.h>
#endif

#include "../Model/ModelDispatch/driver_abi.h"

#define MKL_INT int

#define CPUFREE(x)                                                           \
{                                                                         \
    if (x != nullptr) {                                                      \
        free(x);                                                          \
        x = nullptr;                                                         \
    }                                                                     \
}

extern "C" {
    double DNRM2(Int*,double*,Int*);
    double DDOT(Int*,double*,Int*,double*,Int*);
    void DCOPY(Int*,double*,Int*,double*,Int*);    
    void DSCAL(Int*,double*,double*,Int*);
    void DAXPY(Int*,double*,double*,Int*,double*,Int*);
    void DGEMV(char*,Int*,Int*,double*,double*,Int*,double*,Int*,double*,double*,Int*);  
    void DGEMM(char*,char*,Int*,Int*,Int*,double*,double*,Int*,
             double*,Int*,double*,double*,Int*);        
    void DGETRF(Int*,Int*,double*,Int*,Int*,Int*);
    void DGETRI(Int*,double*,Int*,Int*,double*,Int*,Int*);
    void DTRSM(char *, char*, char*, char *, Int *, Int *, double*, double*, Int*,
             double*, Int*);
    void DGEEV( char* jobvl, char* jobvr, int* n, double* a,
                int* lda, double* wr, double* wi, double* vl, int* ldvl,
                double* vr, int* ldvr, double* work, int* lwork, int* info );    
    
    float SNRM2(Int*,float*,Int*);  
    float SDOT(Int*,float*,Int*,float*,Int*);
    void SCOPY(Int*,float*,Int*,float*,Int*);
    void SSCAL(Int*,float*,float*,Int*);
    void SAXPY(Int*,float*,float*,Int*,float*,Int*);
    void SGEMM(char*,char*,Int*,Int*,Int*,float*,float*,Int*,
             float*,Int*,float*,float*,Int*);  
    void SGEMV(char*,Int*,Int*,float*,float*,Int*,float*,Int*,float*,float*,Int*);      
    void SGETRF(Int*,Int*,float*,Int*,Int*,Int*);    
    void SGETRI(Int*,float*,Int*,Int*,float*,Int*,Int*);
    void STRSM(char *, char*, char*, char *, Int *, Int*, float*, float*, Int*,
             float*, Int*);        
    void SGEEV( char* jobvl, char* jobvr, Int* n, float* a,
                Int* lda, float* wr, float* wi, float* vl, Int* ldvl,
                float* vr, Int* ldvr, float* work, Int* lwork, Int* info );    
}

template <typename T>
bool is_nan_bitwise(T x);

// Specialization for double
template <> bool is_nan_bitwise<double>(double x) {
    uint64_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return ((bits & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL) &&  // exponent all 1s
           ((bits & 0x000fffffffffffffULL) != 0);                         // mantissa nonzero
}

// Specialization for float
template <> bool is_nan_bitwise<float>(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return ((bits & 0x7f800000U) == 0x7f800000U) &&                       // exponent all 1s
           ((bits & 0x007fffffU) != 0);                                   // mantissa nonzero
}

// Optional macro for quick usage
#define IS_NAN(x) is_nan_bitwise<decltype(x)>(x)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// global constants for BLAS. `inline` (C++17) so this header can be included in more than
// one TU without multiple-definition link errors -- e.g. an external driver's main.cpp plus
// the Exasim library both pull common.h (the MPI preprocessing path drags in ExasimSolver.o,
// which also defines these). Without inline they are strong symbols and collide.
inline dstype one = 1.0;
inline dstype minusone = -1.0;
inline dstype zero = 0.0;
inline char chn = 'N';
inline char cht = 'T';
inline char chl = 'L';
inline char chu = 'U';
inline char chr = 'R';
inline char chv = 'V';
inline Int inc1 = 1;

// global variables for CUBLAS  
// dstype *cublasOne;
// dstype *cublasMinusone;
// dstype *cublasZero;
dstype cublasOne[1] = {one};
dstype cublasMinusone[1] = {minusone};
dstype cublasZero[1] = {zero};

#ifdef HAVE_CUDA       
   #define CUDA_SYNC cudaDeviceSynchronize();  
#else 
   #define CUDA_SYNC
#endif                      

#ifdef TIMING    
    #define INIT_TIMING auto begin = std::chrono::high_resolution_clock::now(); auto end = std::chrono::high_resolution_clock::now();
#else
    #define INIT_TIMING
#endif

#ifdef TIMING
   #define TIMING_START  begin = std::chrono::high_resolution_clock::now();   
#else 
   #define TIMING_START     
#endif       

#ifdef TIMING
   #define TIMING_END    end = std::chrono::high_resolution_clock::now();   
#else 
   #define TIMING_END     
#endif       

#ifdef TIMING       
   #define TIMING_GET(num) common.timing[num] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;        
#else 
   #define TIMING_GET(num)  
#endif                      

#ifdef TIMING
   #define START_TIMING {CUDA_SYNC; TIMING_START;}       
#else 
   #define START_TIMING
#endif       

#ifdef TIMING
   #define END_TIMING(num) {CUDA_SYNC; TIMING_END; TIMING_GET(num)}   
#else 
   #define END_TIMING(num)
#endif       

#ifdef TIMING       
   #define TIMING_GET1(num) disc.common.timing[num] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;        
#else 
   #define TIMING_GET1(num)  
#endif                      

#ifdef TIMING
   #define END_TIMING_DISC(num) {CUDA_SYNC; TIMING_END; TIMING_GET1(num)}   
#else 
   #define END_TIMING_DISC(num)
#endif       
                
#ifdef HAVE_CUDA     

#ifdef USE_FLOAT
#define cublasNRM2 cublasSnorm2
#define cublasDOT cublasSdot
#define cublasAXPY cublasSaxpy
#define cublasGEMV cublasSgemv
#define cublasGEMM cublasSgemm
#define cublasGEMVBatched cublasSgemvBatched
#define cublasGEMMBatched cublasSgemmBatched
#define cublasGEMVStridedBatched cublasSgemvStridedBatched
#define cublasGEMMStridedBatched cublasSgemmStridedBatched
#define cublasGETRF cublasSgetrf
#define cublasGETRI cublasSgetri
#define cublasGETRFBatched cublasSgetrfBatched
#define cublasGETRIBatched cublasSgetriBatched
#define cublasTRSM cublasStrsm 
#else
#define cublasNRM2 cublasDnorm2
#define cublasDOT cublasDdot
#define cublasAXPY cublasDaxpy
#define cublasGEMV cublasDgemv
#define cublasGEMM cublasDgemm
#define cublasGEMVBatched cublasDgemvBatched
#define cublasGEMMBatched cublasDgemmBatched
#define cublasGEMVStridedBatched cublasDgemvStridedBatched
#define cublasGEMMStridedBatched cublasDgemmStridedBatched
#define cublasGETRF cublasDgetrf
#define cublasGETRI cublasDgetri
#define cublasGETRFBatched cublasDgetrfBatched
#define cublasGETRIBatched cublasDgetriBatched
#define cublasTRSM cublasDtrsm 
#endif

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUBLAS(call)                                                     \
{                                                                              \
    cublasStatus_t err;                                                        \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define GPUFREE(x)                                                       \
{                                                                         \
    if (x != nullptr) {                                                      \
        cudaTemplateFree(x);                                              \
        x = nullptr;                                                         \
    }                                                                     \
}

template <typename T> static void cudaTemplateMalloc(T **d_data, Int n)
{
    // allocate the memory on the GPU            
    CHECK( cudaMalloc( (void**)d_data, n * sizeof(T) ) );
}

template <typename T> static void cudaTemplateMallocManaged(T **d_data, Int n)
{
    // allocate unified memory 
    CHECK( cudaMallocManaged( (void**)d_data, n * sizeof(T) ) );        
}

template <typename T> static void cudaTemplateHostAlloc(T **h_data, Int n, unsigned int flags)
{
    // allocate zero-copy memory on host    
    CHECK(cudaHostAlloc((void **)h_data, n * sizeof(T), flags));                
}

template <typename T> static void cudaTemplateHostAllocMappedMemory(T **h_data, Int n)
{
    // allocate zero-copy memory on host    
    CHECK(cudaHostAlloc((void **)h_data, n * sizeof(T), cudaHostAllocMapped));                
}

template <typename T> static void cudaTemplateHostAllocPinnedMemory(T **h_data, Int n)
{
    // allocate pinned memory on host        
    CHECK(cudaHostAlloc((void **)h_data, n * sizeof(T), cudaHostAllocDefault));                
}

template <typename T> static void cudaTemplateFree(T *d_data)
{
    // free the memory on the GPU            
    CHECK( cudaFree( d_data ) );    
}

template <typename T> static void cudaCopytoDevice(T *d_data, T *h_data, Int n)
{
    // copy data from CPU to GPU
    CHECK( cudaMemcpy( d_data, h_data, n * sizeof(T), cudaMemcpyHostToDevice ) );    
}

template <typename T> static void cudaCopytoHost(T *h_data, T *d_data, Int n)
{
    // copy data from GPU to CPU
    CHECK( cudaMemcpy( h_data, d_data, n * sizeof(T), cudaMemcpyDeviceToHost ) );    
}

#endif

#ifdef HAVE_HIP

#define CHECK(call)                                                            \
{                                                                              \
    const hipError_t error = call;                                             \
    if (error != hipSuccess)                                                   \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                hipGetErrorString(error));                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_HIPBLAS(call)                                                    \
{                                                                              \
    hipblasStatus_t err;                                                       \
    if ((err = (call)) != HIPBLAS_STATUS_SUCCESS)                              \
    {                                                                          \
        fprintf(stderr, "Got hipBLAS error %d at %s:%d\n", err, __FILE__,      \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_ROCBLAS(call)                                                    \
{                                                                              \
    rocblas_status err;                                                        \
    if ((err = (call)) != rocblas_status_success)                              \
    {                                                                          \
        fprintf(stderr, "Got rocBLAS error %d at %s:%d\n", err, __FILE__,      \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define HIPFREE(x)                                                       \
{                                                                         \
    if (x != nullptr) {                                                      \
        CHECK( hipFree(x) );                                              \
        x = nullptr;                                                         \
    }                                                                     \
}


template <typename T> static void hipTemplateHostMalloc(T **h_data, Int n, unsigned int flags)
{
    // allocate zero-copy memory on host    
    CHECK(hipHostMalloc((void **)h_data, n * sizeof(T), flags));                
}

#endif

template <typename T> static void TemplateMalloc(T **data, Int n, Int backend)
{    
    if ((backend <= 1) && (n>0))              
        *data = (T *) malloc(n*sizeof(T));      

#ifdef HAVE_CUDA            
    if ((backend == 2) && (n>0)) // CUDA C                
        // allocate the memory on the GPU            
        CHECK( cudaMalloc( (void**)data, n * sizeof(T) ) );
#endif                 
    
#ifdef HAVE_HIP
    if ((backend == 3) && (n > 0)) // HIP
    {
        // Allocate memory on the GPU using HIP
        CHECK( hipMalloc( (void**)data, n * sizeof(T) ) );
    }
#endif    
}

template <typename T> static void TemplateFree(T *data,  Int backend)
{
    if (backend <= 1)  CPUFREE(data);
        
#ifdef HAVE_CUDA            
    if (backend == 2)  GPUFREE(data);
#endif                  
    
#ifdef HAVE_HIP            
    if (backend == 3)  HIPFREE(data);
#endif                      
}

template <typename T> static void TemplateReallocate(T **data, Int n, Int backend)
{
    TemplateFree(*data,  backend);
    TemplateMalloc(data, n, backend);
}

template <typename T> static void TemplateCopytoDevice(T *d_data, const T *h_data, Int n, Int backend)
{
    if (backend <= 1)  {
        for (Int i=0; i<n; i++)
            d_data[i] = h_data[i];
    }
    
#ifdef HAVE_CUDA            
    // copy data from CPU to GPU
    if ((backend == 2) && (n>0)) CHECK( cudaMemcpy( d_data, h_data, n * sizeof(T), cudaMemcpyHostToDevice ) );            
#endif    
    
#ifdef HAVE_HIP
    // Copy data from CPU to GPU using HIP
    if ((backend == 3) && (n > 0)) {
        CHECK( hipMemcpy(d_data, h_data, n * sizeof(T), hipMemcpyHostToDevice) );
    }
#endif    
}

template <typename T> static void TemplateMallocCopytoDevice(T **d_data, const T *h_data, Int n, Int backend)
{
    if (n <= 0) {
        *d_data = nullptr;
        return;
    }

    TemplateMalloc(d_data, n, backend);
    TemplateCopytoDevice(*d_data, h_data, n, backend);
}

template <typename T> static void TemplateCopytoHost(T *h_data, T *d_data, Int n, Int backend)
{
    if (backend <= 1)  {
        for (Int i=0; i<n; i++)
            h_data[i] = d_data[i];
    }

#ifdef HAVE_CUDA
    // copy data from GPU to CPU
    if (backend == 2) CHECK( cudaMemcpy( h_data, d_data, n * sizeof(T), cudaMemcpyDeviceToHost ) );
#endif    
    
#ifdef HAVE_HIP
    // Copy data from GPU to CPU using HIP
    if (backend == 3) {
        CHECK( hipMemcpy(h_data, d_data, n * sizeof(T), hipMemcpyDeviceToHost) );
    }
#endif    
}

// static void PrintErrorAndExit(const char* errmsg, const char *file, int line ) 
// {    
//     printf( "%s in %s at line %d\n", errmsg, file, line );
// 
// #ifdef  HAVE_MPI       
//     MPI_Finalize();    
// #endif
// 
//     exit( 1 );    
// }
// 
// static void PrintErrorAndExit(string errmsg, const char *file, int line ) 
// {    
//     printf( "%s in %s at line %d\n", errmsg.c_str(), file, line );
// 
// #ifdef  HAVE_MPI       
//     MPI_Finalize();    
// #endif
// 
//     exit( 1 );    
// }
// 
// #define error( errmsg ) (PrintErrorAndExit( errmsg, __FILE__, __LINE__ ))

// -----------------------------------------------------------------------------
// Print an error message (with file/line info) and terminate the program.
// If MPI is enabled, aborts all ranks immediately using MPI_Abort().
// -----------------------------------------------------------------------------

static inline void PrintErrorAndExit(const std::string& errmsg, const char* file, int line)
{
    int rank = 0;

#ifdef HAVE_MPI
    MPI_Comm_rank(EXASIM_COMM_WORLD, &rank);
#endif
    
    fprintf(stderr,
            "\n==============================================\n"
            "[Rank %d] ERROR: %s\n"
            "  Location: %s:%d\n"
            "==============================================\n\n",
            rank, errmsg.c_str(), file, line);
    fflush(stderr);

#ifdef HAVE_MPI
    // Abort the entire MPI job instead of trying to finalize gracefully.
    // MPI_Finalize() is unsafe after a runtime error and can hang.
    MPI_Abort(EXASIM_COMM_WORLD, EXIT_FAILURE);
#else
    exit(EXIT_FAILURE);
#endif
}

// -----------------------------------------------------------------------------
// Macro for convenient error reporting
// -----------------------------------------------------------------------------
#define error(msg)  PrintErrorAndExit((msg), __FILE__, __LINE__)

std::string trim_dir(const std::string& s) {
    return std::filesystem::path{s}.parent_path().string();   // use .native() if you want OS-preferred slashes
}

bool ensure_dir(const std::string& dir) {
    std::filesystem::path p(dir);
    if (std::filesystem::exists(p)) return std::filesystem::is_directory(p);  // false if it's a file
    return std::filesystem::create_directories(p);               // creates parents as needed
}

std::string make_path(const std::string& str1, const std::string& str2) {
    std::filesystem::path base = str1;
    std::filesystem::path tail = str2;

    // If tail is absolute, strip its root so it becomes relative    
    tail = tail.relative_path();

    std::filesystem::path full = base / tail;
    return full.lexically_normal().string();
}

std::string trimToSubstringAtFirstOccurence(const std::string& fullPath, const std::string& keyword) {
    std::size_t pos = fullPath.find(keyword);  // Use find to get the first occurrence
    if (pos != std::string::npos) {
        return fullPath.substr(0, pos + keyword.length());
    }
    else {      
      return "";
    }
}

std::string trimToSubstringAtFirstOccurence(const std::filesystem::path& fullPath, const std::string& keyword) {
    const std::string s = fullPath.generic_string();
    std::size_t pos = s.find(keyword);  // Use find to get the first occurrence
    if (pos != std::string::npos) {
        return s.substr(0, pos + keyword.length());
    }
    else {      
      return "";
    }
}

std::string trimToSubstringAtLastOccurence(const std::string& fullPath, const std::string& keyword) {
    std::size_t pos = fullPath.rfind(keyword);  // Use rfind to get the last occurrence
    if (pos != std::string::npos) {
        return fullPath.substr(0, pos + keyword.length());
    }
    else {      
      return "";
    }
}

std::string trimToSubstringAtLastOccurence(const std::filesystem::path& fullPath,
                                           const std::string& keyword)
{
    // generic_string uses forward slashes on all platforms (nice for substring ops)
    const std::string s = fullPath.generic_string();
    const auto pos = s.rfind(keyword);
    if (pos != std::string::npos)
        return s.substr(0, pos + keyword.size());
    return "";
}

// Named offsets into appstruct::ndims -- the app.bin wire format written by the (frozen) Matlab/
// Python/Julia frontends. The packed layout must stay fixed to keep reading app.bin; these constants
// only make the decode self-documenting: app.ndims[AppNdims::nc] instead of a bare app.ndims[5].
// Indices 2..4 are unused padding. Meanings mirror the decode in setstructs.cpp / buildstructs.hpp.
struct AppNdims {
    enum {
        mpiprocs = 0,  // number of MPI ranks
        nd       = 1,  // spatial dimension
        nc       = 5,  // components of (u, q)
        ncu      = 6,  // components of u
        ncq      = 7,  // components of q
        ncp      = 8,  // components of p (mostly unused)
        nco      = 9,  // components of o (auxiliary)
        nch      = 10, // components of uhat (trace)
        ncx      = 11, // components of xdg (coordinates)
        nce      = 12, // components of output fields
        ncw      = 13, // components of w (wave/auxiliary)
        nsca     = 14, // scalar vis fields
        nvec     = 15, // vector vis fields
        nten     = 16, // tensor vis fields
        nsurf    = 17, // surface vis/storage/QoI fields
        nvqoi    = 18  // volume quantities of interest
    };
};

template <class T = ::dstype, class I = ::Int>
struct appstructT {
    using dstype = T; using Int = I;
    Int *lsize=nullptr;
    Int *nsize=nullptr;  // data size
    Int *ndims=nullptr;  // dimensions
    Int *flag=nullptr;   // flag parameters
    Int *problem=nullptr;// problem parameters    
    Int *comm=nullptr;   // communication parameters 
    Int *porder=nullptr; // polymnomial degrees
    Int *stgib=nullptr;
    Int *vindx=nullptr;
    Int *interfacefluxmap=nullptr;
    Int *wmModelIDs=nullptr;
    Int *wmBoundaries=nullptr;
    
    dstype *uinf=nullptr;    // boundary data
    dstype *dt=nullptr;      // time steps       
    dstype *dae_dt=nullptr;  // dual time steps       
    dstype *factor=nullptr;  // factors      
    dstype *physicsparam=nullptr; // physical parameters
    dstype *solversparam=nullptr; // solvers parameters
    dstype *tau=nullptr; // stabilization parameters
    dstype *stgdata=nullptr; 
    dstype *stgparam=nullptr;
    dstype *avparam=nullptr;
    dstype *wmDistances=nullptr;
    
    //dstype time=nullptr;     /* current time */
    dstype *fc_u=nullptr;    /* factor when discretizing the time derivative of the U equation. Allow scalar field for local time stepping in steady problems? */
    dstype *fc_q=nullptr;    /* factor when discretizing the time derivative of the Q equation. Allow scalar field for local time stepping in steady problems? */
    dstype *fc_w=nullptr;    /* factor when discretizing the time derivative of the P equation. Allow scalar field for local time stepping in steady problems? */    
    
    dstype *dtcoef_u=nullptr;    /* factor when discretizing the time derivative of the U equation. Allow scalar field for local time stepping in steady problems? */
    dstype *dtcoef_q=nullptr;    /* factor when discretizing the time derivative of the Q equation. Allow scalar field for local time stepping in steady problems? */
    dstype *dtcoef_w=nullptr;    /* factor when discretizing the time derivative of the P equation. Allow scalar field for local time stepping in steady problems? */    
    
    Int szflag=0, szproblem=0, szcomm=0, szporder=0, szstgib=0, szvindx=0, szinterfacefluxmap=0;
    Int szwmModelIDs=0, szwmBoundaries=0;
    Int szuinf=0, szdt=0, szdae_dt=0, szfactor=0, szphysicsparam=0, szsolversparam=0;
    Int sztau=0, szstgdata=0, szstgparam=0, szfc_u=0, szfc_q=0, szfc_w=0;
    Int szdtcoef_u=0, szdtcoef_q=0, szdtcoef_w=0, szavparam=0, szwmDistances=0;
    Int read_uh = 0;
    Int modelnumber = 0;
    Int builtinmodelID = 0;
    Int frontendgenerated = 0;

    int sizeofint() {
      int sz = szflag + szproblem + szcomm + szporder + szstgib + szvindx + szinterfacefluxmap
             + szwmModelIDs + szwmBoundaries;
      return sz;
    }
    int sizeoffloat() {
      int sz = szuinf+szdt+szdae_dt+szfactor+szphysicsparam+szsolversparam+
               sztau+szstgdata+szstgparam+szfc_u+szfc_q+szfc_w+szdtcoef_u+
               szdtcoef_q+szdtcoef_w+szavparam+szwmDistances;
      return sz;        
    }

    void printinfo()
    {    
      printf("--------------- App Struct Information ----------------\n");
      printf("size of flag: %d\n", szflag);
      printf("size of problem: %d\n", szproblem);
      printf("size of comm: %d\n", szcomm);
      printf("size of porder: %d\n", szporder);
      printf("size of stgib: %d\n", szstgib);
      printf("size of vindx: %d\n", szvindx);
      printf("size of interfacefluxmap: %d\n", szinterfacefluxmap);
      printf("size of wmModelIDs: %d\n", szwmModelIDs);
      printf("size of wmBoundaries: %d\n", szwmBoundaries);
      printf("size of uinf: %d\n", szuinf);
      printf("size of dt: %d\n", szdt);
      printf("size of dae_dt: %d\n", szdae_dt);
      printf("size of factor: %d\n", szfactor);
      printf("size of physicsparam: %d\n", szphysicsparam);
      printf("size of solversparam: %d\n", szsolversparam);
      printf("size of tau: %d\n", sztau);
      printf("size of stgdata: %d\n", szstgdata);
      printf("size of stgparam: %d\n", szstgparam);
      printf("size of avparam: %d\n", szavparam);
      printf("size of wmDistances: %d\n", szwmDistances);
      printf("size of fc_u: %d\n", szfc_u);
      printf("size of fc_q: %d\n", szfc_q);
      printf("size of fc_w: %d\n", szfc_w);
      printf("size of dtcoef_u: %d\n", szdtcoef_u);
      printf("size of dtcoef_q: %d\n", szdtcoef_q);
      printf("size of dtcoef_w: %d\n", szdtcoef_w);
      printf("size of int: %d\n", sizeofint());
      printf("size of float: %d\n", sizeoffloat());
    }

    #ifdef HAVE_MPP
    Mutation::Mixture *mix=nullptr;
    #endif

    // custom destructor
    void freememory(Int backend)
    {
        TemplateFree(lsize, backend);
        TemplateFree(nsize, backend);
        TemplateFree(ndims, backend);   
        TemplateFree(comm, backend);   
        TemplateFree(porder, backend);               
        TemplateFree(flag, backend);    
        TemplateFree(problem, backend);
        TemplateFree(stgib, backend);
        TemplateFree(vindx, backend);
        TemplateFree(interfacefluxmap, backend);
        TemplateFree(wmModelIDs, backend);
        TemplateFree(wmBoundaries, backend);
        TemplateFree(uinf, backend);
        TemplateFree(dt, backend);
        TemplateFree(dae_dt, backend);
        TemplateFree(factor, backend);
        TemplateFree(physicsparam, backend);
        TemplateFree(solversparam, backend);
        TemplateFree(tau, backend);
        TemplateFree(stgdata, backend);
        TemplateFree(stgparam, backend);
        TemplateFree(avparam, backend);
        TemplateFree(wmDistances, backend);
        TemplateFree(fc_u, backend);
        TemplateFree(fc_q, backend);
        TemplateFree(fc_w, backend);
        TemplateFree(dtcoef_u, backend);
        TemplateFree(dtcoef_q, backend);
        TemplateFree(dtcoef_w, backend);
    }
};
using appstruct = appstructT<::dstype, ::Int>;

struct wallmodelstruct {
    Int initialized = 0;

    Int ibc = -1;
    Int nd = 0;
    Int ncx = 0;
    Int npe = 0;
    Int npf = 0;
    Int ngf = 0;
    Int nfe = 0;
    Int nfaces = 0;
    Int npoints = 0;
    Int nbe1 = 0;
    Int bfwmDepth = 4;
    Int bfwmWidth = 15;
    dstype y1 = 0.0;

    Int* faces = nullptr;
    Int* nextfaces = nullptr;  // always allocated in CPU memory
    Int* elems = nullptr;
    Int* elemsx1 = nullptr;
    dstype* xw = nullptr;
    dstype* nw = nullptr;
    dstype* x1 = nullptr;
    dstype* xi1 = nullptr;
    dstype* shap1 = nullptr;
    dstype* bfwmTauwCoeffs = nullptr;
    dstype* bfwmQwCoeffs = nullptr;

    Int szfaces = 0;
    Int sznextfaces = 0;
    Int szelems = 0;
    Int szelemsx1 = 0;
    Int szxw = 0;
    Int sznw = 0;
    Int szx1 = 0;
    Int szxi1 = 0;
    Int szshap1 = 0;
    Int szbfwmTauwCoeffs = 0;
    Int szbfwmQwCoeffs = 0;

    int sizeofint()
    {
        return szfaces + sznextfaces + szelems + szelemsx1;
    }

    int sizeoffloat()
    {
        return szxw + sznw + szx1 + szxi1 + szshap1 + szbfwmTauwCoeffs + szbfwmQwCoeffs;
    }

    void freememory(Int backend)
    {
        TemplateFree(faces, backend);
        TemplateFree(nextfaces, 0);
        TemplateFree(elems, backend);
        TemplateFree(elemsx1, backend);
        TemplateFree(xw, backend);
        TemplateFree(nw, backend);
        TemplateFree(x1, backend);
        TemplateFree(xi1, backend);
        TemplateFree(shap1, backend);
        TemplateFree(bfwmTauwCoeffs, backend);
        TemplateFree(bfwmQwCoeffs, backend);

        initialized = 0;
        ibc = -1;
        nd = 0;
        ncx = 0;
        npe = 0;
        npf = 0;
        ngf = 0;
        nfe = 0;
        nfaces = 0;
        npoints = 0;
        nbe1 = 0;
        bfwmDepth = 4;
        bfwmWidth = 15;
        y1 = 0.0;

        szfaces = 0;
        sznextfaces = 0;
        szelems = 0;
        szelemsx1 = 0;
        szxw = 0;
        sznw = 0;
        szx1 = 0;
        szxi1 = 0;
        szshap1 = 0;
        szbfwmTauwCoeffs = 0;
        szbfwmQwCoeffs = 0;
    }
};

template <class T = ::dstype, class I = ::Int>
struct masterstructT {
    using dstype = T; using Int = I;
    
    Int *lsize=nullptr;
    Int *nsize=nullptr;  // data size
    Int *ndims=nullptr;  // dimensions
    
    dstype *shapegwdotshapeg=nullptr;
    dstype *shapfgwdotshapfg=nullptr;
    dstype *shapegt=nullptr; // element shape functions at Gauss points (transpose)
    dstype *shapegw=nullptr; // element shape functions at Gauss points multiplied by Gauss weights
    dstype *shapfgt=nullptr; // face shape functions at Gauss points (transpose)
    dstype *shapfgw=nullptr; // face shape functions at Gauss points multiplied by Gauss weights    
    dstype *shapent=nullptr; // element shape functions at nodes (transpose)
    dstype *shapen=nullptr;  // element shape functions at nodes        
    dstype *shapfnt=nullptr; // element shape functions at nodes (transpose)
    dstype *shapfn=nullptr;  // element shape functions at nodes        
    dstype *xpe=nullptr; // nodal points on master element
    dstype *gpe=nullptr; // gauss points on master element
    dstype *gwe=nullptr; // gauss weighs on master element
    dstype *xpf=nullptr; // nodal points on master face
    dstype *gpf=nullptr; // gauss points on master face
    dstype *gwf=nullptr; // gauss weighs on master face
    
    dstype *shap1dgt=nullptr; 
    dstype *shap1dgw=nullptr; 
    dstype *shap1dnt=nullptr; 
    dstype *shap1dnl=nullptr; 
    dstype *xp1d=nullptr; // node points on 1D element
    dstype *gp1d=nullptr; // gauss points on 1D element
    dstype *gw1d=nullptr; // gauss weights on 1D element    
    
    Int szshapegwdotshapeg=0, szshapfgwdotshapfg=0, szshapegt=0, szshapegw=0;
    Int szshapfgt=0, szshapfgw=0, szshapent=0, szshapen=0, szshapfnt=0;
    Int szshapfn=0, szxpe=0, szgpe=0, szgwe=0, szxpf=0, szgpf=0, szgwf=0;
    Int szshap1dgt=0, szshap1dgw=0, szshap1dnt=0, szshap1dnl=0, szxp1d=0;
    Int szgp1d=0, szgw1d=0;

    int sizeofint() { return 0;} 

    int sizeoffloat()
    {
      int sz = szshapegwdotshapeg+szshapfgt + szshapfgw + szshapent + szshapen +
               szshapfnt + szshapfn + szxpe + szgpe + szgwe + szxpf + szgpf + 
               szgwf + szshap1dgt + szshap1dgw + szshap1dnt + szshap1dnl + 
               szxp1d + szgp1d + szgw1d;
      return sz;         
    }

    void printinfo()
    {
      printf("--------------- Master Struct Information ----------------\n");
      printf("size of shapegwdotshapeg: %d\n", szshapegwdotshapeg);
      printf("size of shapfgwdotshapfg: %d\n", szshapfgwdotshapfg);
      printf("size of shapegt: %d\n", szshapegt);
      printf("size of shapegw: %d\n", szshapegw);
      printf("size of shapfgt: %d\n", szshapfgt);
      printf("size of shapfgw: %d\n", szshapfgw);
      printf("size of shapent: %d\n", szshapent);
      printf("size of shapen: %d\n", szshapen);
      printf("size of shapfnt: %d\n", szshapfnt);
      printf("size of shapfn: %d\n", szshapfn);
      printf("size of xpe: %d\n", szxpe);
      printf("size of gpe: %d\n", szgpe);
      printf("size of gwe: %d\n", szgwe);
      printf("size of xpf: %d\n", szxpf);
      printf("size of gpf: %d\n", szgpf);
      printf("size of gwf: %d\n", szgwf);
      printf("size of shap1dgt: %d\n", szshap1dgt);
      printf("size of shap1dgw: %d\n", szshap1dgw);
      printf("size of shap1dnt: %d\n", szshap1dnt);
      printf("size of shap1dnl: %d\n", szshap1dnl);
      printf("size of xp1d: %d\n", szxp1d);
      printf("size of gp1d: %d\n", szgp1d);
      printf("size of gw1d: %d\n", szgw1d);
      printf("size of int: %d\n", sizeofint());
      printf("size of float: %d\n", sizeoffloat());
    }

    void freememory(Int backend)
    {
        TemplateFree(lsize, backend);
        TemplateFree(nsize, backend);
        TemplateFree(ndims, backend);    
        TemplateFree(shapegt, backend); // element shape functions at Gauss points (transpose)
        TemplateFree(shapegw, backend); // element shape functions at Gauss points multiplied by Gauss weights
        TemplateFree(shapfgt, backend); // face shape functions at Gauss points (transpose)
        TemplateFree(shapfgw, backend); // face shape functions at Gauss points multiplied by Gauss weights    
        TemplateFree(shapent, backend); // element shape functions at nodes (transpose)
        TemplateFree(shapen, backend);  // element shape functions at nodes        
        TemplateFree(shapfnt, backend); // element shape functions at nodes (transpose)
        TemplateFree(shapfn, backend);  // element shape functions at nodes        
        TemplateFree(xpe, backend); // nodal points on master element
        TemplateFree(gpe, backend); // gauss points on master element
        TemplateFree(gwe, backend); // gauss weighs on master element
        TemplateFree(xpf, backend); // nodal points on master face
        TemplateFree(gpf, backend); // gauss points on master face
        TemplateFree(gwf, backend); // gauss weighs on master face            
        TemplateFree(shap1dgt, backend); 
        TemplateFree(shap1dgw, backend); 
        TemplateFree(shap1dnt, backend); 
        TemplateFree(shap1dnl, backend); 
        TemplateFree(xp1d, backend); 
        TemplateFree(gp1d, backend); 
        TemplateFree(gw1d, backend);            
        TemplateFree(shapegwdotshapeg, backend);
        TemplateFree(shapfgwdotshapfg, backend);             
    }            
};
using masterstruct = masterstructT<::dstype, ::Int>;
  
template <class T = ::dstype, class I = ::Int>
struct meshstructT {
    using dstype = T; using Int = I;
    Int *lsize=nullptr;
    Int *nsize=nullptr;  // data size
    Int *ndims=nullptr;  // dimensions
        
    Int *facecon=nullptr;    // face-to-element connectivities 
    Int *e2f=nullptr;        // element-to-face connectivities
    Int *f2e=nullptr;        // face-to-element connectivities
    Int *f2f=nullptr;        // face-to-face connectivities
    Int *f2l=nullptr;        // face-to-local connectivities
    Int *elemcon=nullptr;    // element-to-face connectivities
    Int *perm=nullptr;       // indices of element nodes on faces
    Int *bf=nullptr;         // boundary faces  
    Int *boufaces=nullptr;   // boundary faces
    Int *intfaces=nullptr;   // interface faces
    Int *eblks=nullptr;    // element blocks
    Int *fblks=nullptr;    // face blocks    
    Int *nbsd=nullptr;
    Int *elemsend=nullptr;
    Int *elemrecv=nullptr;
    Int *elemsendpts=nullptr;
    Int *elemrecvpts=nullptr;
    Int *elempart=nullptr;
    Int *elempartpts=nullptr;
    Int *cgelcon=nullptr;
    Int *rowent2elem=nullptr;
    Int *cgent2dgent=nullptr;
    Int *colent2elem=nullptr;
    Int *rowe2f1=nullptr;
    Int *cole2f1=nullptr;
    Int *ent2ind1=nullptr;
    Int *rowe2f2=nullptr;
    Int *cole2f2=nullptr;
    Int *ent2ind2=nullptr;
    Int *row_ptr=nullptr;
    Int *col_ind=nullptr;
    Int *face=nullptr;
    Int *cartgridpart=nullptr;
    Int *boundaryConditions=nullptr;
    Int *intepartpts=nullptr;
    
    Int *faceperm=nullptr;
    Int *nbintf=nullptr;
    Int *facesend=nullptr;
    Int *facerecv=nullptr;
    Int *facesendpts=nullptr;
    Int *facerecvpts=nullptr;
    
    Int *findxdg1=nullptr; 
    //Int *findxdg2=nullptr; 
    Int *findxdgp=nullptr; 
    Int *findudg1=nullptr; 
    Int *findudg2=nullptr; 
    Int *findudgp=nullptr;     
    Int *eindudg1=nullptr;     
    Int *eindudgp=nullptr;     
    Int *elemsendind=nullptr;
    Int *elemrecvind=nullptr;
    Int *elemsendodg=nullptr;
    Int *elemrecvodg=nullptr;
    Int *elemsendudg=nullptr;
    Int *elemrecvudg=nullptr;
    //Int *index=nullptr; 
    
    Int szfacecon=0, szf2e=0, szf2f=0, szf2l=0, szelemcon=0, szperm=0, szbf=0, szboufaces=0, szintfaces=0; 
    Int szeblks=0, sze2f=0, szfblks=0, sznbsd=0, szelemsend=0;
    Int szelemrecv=0, szelemsendpts=0, szelemrecvpts=0, szelempart=0;
    Int szelempartpts=0, szcgelcon=0, szrowent2elem=0, szcgent2dgent=0;
    Int szcolent2elem=0, szrowe2f1=0, szcole2f1=0, szent2ind1=0, szrowe2f2=0;
    Int szcole2f2=0, szent2ind2=0, szfindxdg1=0, szfindxdg2=0, szfindxdgp=0; 
    Int szfindudg1=0, szfindudg2=0, szfindudgp=0, szeindudg1=0, szeindudgp=0;
    Int szelemsendind=0, szelemrecvind=0, szelemsendodg=0, szelemrecvodg=0;
    Int szelemsendudg=0, szelemrecvudg=0, szindex=0, szcartgridpart=0;    
    Int szfaceperm=0, sznbintf=0, szfacesend=0, szfacerecv=0, szfacesendpts=0, szfacerecvpts=0;
    
    int sizeoffloat() {return 0;}
    int sizeofint() {
      int sz = szeblks+szfblks + sznbsd + szelemsend + szelemrecv + 
               szelemsendpts + szelemrecvpts + szelempart + szelempartpts + 
               szcgelcon + szrowent2elem + szcgent2dgent + szcolent2elem + 
               szrowe2f1 + szcole2f1 + szent2ind1 + szrowe2f2 + szcole2f2 + 
               szent2ind2 + szfindxdg1 + szfindxdgp + szfindudg1 + szfindudg2 + 
               szfindudgp + szeindudg1 + szeindudgp + szelemsendind + szelemrecvind + 
               szelemsendodg + szelemrecvodg + szelemsendudg + szelemrecvudg + szfaceperm +
               sznbintf + szfacesend + szfacerecv + szfacesendpts + szfacerecvpts +
               szfacecon + szf2e + sze2f + szf2f + szf2l + szelemcon + szperm + szbf + szboufaces + szintfaces;
      return sz;        
    }

    void printinfo()
    {
      printf("--------------- Mesh Struct Information ----------------\n");
      printf("size of facecon: %d\n", szfacecon);
      printf("size of e2f: %d\n", sze2f);
      printf("size of f2e: %d\n", szf2e);
      printf("size of f2f: %d\n", szf2f);
      printf("size of f2l: %d\n", szf2l);
      printf("size of elemcon: %d\n", szelemcon);
      printf("size of perm: %d\n", szperm);
      printf("size of bf: %d\n", szbf);
      printf("size of boufaces: %d\n", szboufaces);
      printf("size of intfaces: %d\n", szintfaces);      
      printf("size of eblks: %d\n", szeblks);
      printf("size of fblks: %d\n", szfblks);
      printf("size of nbsd: %d\n", sznbsd);
      printf("size of elemsend: %d\n", szelemsend);
      printf("size of elemrecv: %d\n", szelemrecv);
      printf("size of elemsendpts: %d\n", szelemsendpts);
      printf("size of elemrecvpts: %d\n", szelemrecvpts);
      printf("size of elempart: %d\n", szelempart);
      printf("size of elempartpts: %d\n", szelempartpts);
      printf("size of cgelcon: %d\n", szcgelcon);
      printf("size of rowent2elem: %d\n", szrowent2elem);
      printf("size of cgent2dgent: %d\n", szcgent2dgent);
      printf("size of colent2elem: %d\n", szcolent2elem);
      printf("size of rowe2f1: %d\n", szrowe2f1);
      printf("size of cole2f1: %d\n", szcole2f1);
      printf("size of ent2ind1: %d\n", szent2ind1);
      printf("size of rowe2f2: %d\n", szrowe2f2);
      printf("size of cole2f2: %d\n", szcole2f2);
      printf("size of ent2ind2: %d\n", szent2ind2);
      printf("size of findxdg1: %d\n", szfindxdg1);
      //printf("size of findxdg2: %d\n", szfindxdg2);
      printf("size of findxdgp: %d\n", szfindxdgp);
      printf("size of findudg1: %d\n", szfindudg1);
      printf("size of findudg2: %d\n", szfindudg2);
      printf("size of findudgp: %d\n", szfindudgp);
      printf("size of eindudg1: %d\n", szeindudg1);
      printf("size of eindudgp: %d\n", szeindudgp);
      printf("size of elemsendind: %d\n", szelemsendind);
      printf("size of elemrecvind: %d\n", szelemrecvind);
      printf("size of elemsendodg: %d\n", szelemsendodg);
      printf("size of elemrecvodg: %d\n", szelemrecvodg);
      printf("size of elemsendudg: %d\n", szelemsendudg);
      printf("size of elemrecvudg: %d\n", szelemrecvudg);      
      printf("size of faceperm: %d\n", szfaceperm);
      printf("size of nbintf: %d\n", sznbintf);
      printf("size of facesend: %d\n", szfacesend);
      printf("size of facerecv: %d\n", szfacerecv);
      printf("size of facesendpts: %d\n", szfacesendpts);
      printf("size of facerecvpts: %d\n", szfacerecvpts);      
      printf("size of int: %d\n", sizeofint());
      printf("size of float: %d\n", sizeoffloat());
    }

    void freememory(Int backend)
    {
        TemplateFree(lsize, backend);
        TemplateFree(nsize, backend);
        TemplateFree(ndims, backend);    
        TemplateFree(facecon, backend);    // face-to-element connectivities 
        TemplateFree(e2f, backend); 
        TemplateFree(f2e, backend);    // face-to-element connectivities 
        TemplateFree(f2f, backend); 
        TemplateFree(f2l, backend); 
        CPUFREE(bf); 
        TemplateFree(elemcon, backend);    // element-to-face connectivities
        TemplateFree(perm, backend);       // indices of element nodes on faces
        TemplateFree(boufaces, backend);   // boundary faces
        TemplateFree(intfaces, backend);   // interface faces
        TemplateFree(eblks, backend);    // element blocks
        TemplateFree(fblks, backend);    // face blocks    
        TemplateFree(nbsd, backend);
        TemplateFree(elemsend, backend);
        TemplateFree(elemrecv, backend);
        TemplateFree(elemsendpts, backend);
        TemplateFree(elemrecvpts, backend);            
        TemplateFree(elempart, backend);
        TemplateFree(elempartpts, backend);   
        TemplateFree(cgelcon, backend);            
        TemplateFree(rowent2elem, backend);
        TemplateFree(cgent2dgent, backend);
        TemplateFree(colent2elem, backend);
        TemplateFree(rowe2f1, backend);
        TemplateFree(cole2f1, backend);
        TemplateFree(ent2ind1, backend);
        TemplateFree(rowe2f2, backend);
        TemplateFree(cole2f2, backend);
        TemplateFree(ent2ind2, backend);
        TemplateFree(col_ind, backend);
        TemplateFree(row_ptr, backend);
        TemplateFree(face, backend);
        
        TemplateFree(findxdg1, backend);   
        TemplateFree(findxdgp, backend);   
        TemplateFree(findudg1, backend);   
        TemplateFree(findudg2, backend);   
        TemplateFree(findudgp, backend);               
        TemplateFree(eindudg1, backend);               
        TemplateFree(eindudgp, backend);  
        TemplateFree(elemsendind, backend);   
        TemplateFree(elemrecvind, backend); 
        TemplateFree(elemsendodg, backend);
        TemplateFree(elemrecvodg, backend);
        TemplateFree(elemsendudg, backend);
        TemplateFree(elemrecvudg, backend);
        TemplateFree(faceperm, backend);
        TemplateFree(nbintf, backend);
        TemplateFree(facesend, backend);
        TemplateFree(facerecv, backend);
        TemplateFree(facesendpts, backend);
        TemplateFree(facerecvpts, backend);            
    }            
};
using meshstruct = meshstructT<::dstype, ::Int>;

template <class T = ::dstype, class I = ::Int>
struct solstructT {
    using dstype = T; using Int = I;
    Int *lsize=nullptr;
    Int *nsize=nullptr;  // data size
    Int *ndims=nullptr;  // dimensions

    // needs-init signals: set by the file reader when a field was NOT supplied by the input
    // (fresh start, no restart data) and must be filled by the model's initial conditions.
    // The reader allocates+zeros the field and raises the flag; initializeSolution() consumes
    // them (so the reader does pure file-read, the model-IC is a separate "initialize a
    // solution" step that queries the discretization for which fields exist).
    Int needudginit=0;   // udg (u, or wave-packed u,q) must be computed from initu/initudg
    Int needodginit=0;   // odg must be computed from initodg
    Int needwdginit=0;   // wdg must be computed from initwdg

    dstype *xdg=nullptr; // spatial coordinates
    dstype *udg=nullptr; // solution (u, q) 
    dstype *sdg=nullptr; // source term due to the previous solution
    dstype *odg=nullptr; // auxilary term 
    dstype *wdg=nullptr; // dw/dt = u (wave problem)
    dstype *uh=nullptr;  // uhat      
    dstype *xcg=nullptr;  // xcg      

    #ifdef HAVE_ENZYME
        dstype *dudg=nullptr; // solution (du, dq, dp) 
        dstype *dwdg=nullptr; // dw/dt = u (wave problem)
        dstype *duh=nullptr; // duhat
        dstype *dodg=nullptr;
        dstype *dodgg=nullptr;
        dstype *dog1=nullptr;
        dstype *dog2=nullptr;
    #endif
    dstype *elemg=nullptr;
    dstype *faceg=nullptr;
    dstype *xdgint=nullptr;   
    dstype *uext = nullptr;   
    // dstype *udgint=nullptr;  
    // dstype *odgint=nullptr;  
    // dstype *wdgint=nullptr;   
    // dstype *uhint=nullptr;  
    // dstype *nlint=nullptr;  
    dstype *elemfaceg=nullptr;        
    //dstype *udgg=nullptr;
    dstype *sdgg=nullptr;
    dstype *odgg=nullptr;
    dstype *og1=nullptr;
    dstype *og2=nullptr;    
    dstype *udgavg=nullptr; // time-average solution (u, q) 
    dstype *bouudgavg=nullptr; 
    dstype *bouwdgavg=nullptr; 
    dstype *bouuhavg=nullptr; 
    dstype *wsrc=nullptr;   // source term due to the time derivative for DAE equations  
    dstype *wdual=nullptr;   // source term due to the dual time derivative for DAE equations  
    dstype** udgarray;    
    
    Int szxdg=0, szxcg=0, szudg=0, szsdg=0, szodg=0, szwdg=0, szuh=0;
    Int szelemg=0, szfaceg=0, szelemfaceg=0, szsdgg=0, szodgg=0, szog1=0, szog2=0;
    Int szudgavg=0, szwsrc=0, szwdual=0, szuext=0, szxdgint=0, szudgint=0, szwdgint=0, szodgint=0;

    int sizeofint() {return 0;}
    int sizeoffloat() {
      int sz = szxdg + szxcg + szudg + szsdg + szodg + szwdg + szuh + szelemg + szfaceg +
               szelemfaceg + szsdgg + szodgg + szog1 + szog2 + szudgavg + 
               szwsrc + szwdual + szuext + szxdgint + szudgint + szodgint + szwdgint;
      return sz;
    }

    void printinfo()
    {
      printf("--------------- Solution Struct Information ----------------\n");
      printf("size of xdg: %d\n", szxdg);
      printf("size of xcg: %d\n", szxcg);
      printf("size of udg: %d\n", szudg);
      printf("size of sdg: %d\n", szsdg);
      printf("size of odg: %d\n", szodg);
      printf("size of wdg: %d\n", szwdg);
      printf("size of uh: %d\n", szuh);
      printf("size of uext: %d\n", szuext);
      printf("size of xdgint: %d\n", szxdgint);
      // printf("size of udgint: %d\n", szudgint);
      // printf("size of wdgint: %d\n", szwdgint);
      // printf("size of odgint: %d\n", szodgint);
      printf("size of elemg: %d\n", szelemg);
      printf("size of faceg: %d\n", szfaceg);
      printf("size of elemfaceg: %d\n", szelemfaceg);
      printf("size of sdgg: %d\n", szsdgg);
      printf("size of odgg: %d\n", szodgg);
      printf("size of og1: %d\n", szog1);
      printf("size of og2: %d\n", szog2);
      printf("size of udgavg: %d\n", szudgavg);
      printf("size of wsrc: %d\n", szwsrc);
      printf("size of wdual: %d\n", szwdual);     
      printf("size of int: %d\n", sizeofint());
      printf("size of float: %d\n", sizeoffloat());   
    } 

    void freememory(Int backend)
    {
        TemplateFree(lsize, backend);
        TemplateFree(nsize, backend);
        TemplateFree(ndims, backend);    
        TemplateFree(xdg, backend); // spatial coordinates
        TemplateFree(xcg, 0);       // spatial coordinates
        TemplateFree(udg, backend); // solution (u, q, p) 
        TemplateFree(sdg, backend); // source term due to the previous solution
        TemplateFree(odg, backend); // auxilary term 
        TemplateFree(wdg, backend); // wave problem
        TemplateFree(uext, backend); 
        TemplateFree(xdgint, backend); // spatial coordinates
      #ifdef HAVE_ENZYME                   
        TemplateFree(dudg, backend); // solution (u, q, p) 
        TemplateFree(dwdg, backend); // wave problem
        TemplateFree(duh, backend);
        TemplateFree(dodg, backend);
        TemplateFree(dodgg, backend);
        TemplateFree(dog1, backend);
        TemplateFree(dog2, backend);
      #endif            
        TemplateFree(uh, backend);  // uhat      
        TemplateFree(elemg, backend); 
        TemplateFree(faceg, backend); 
        TemplateFree(elemfaceg, backend);
        TemplateFree(sdgg, backend); 
        TemplateFree(odgg, backend); 
        TemplateFree(wsrc, backend);
        TemplateFree(wdual, backend);
        TemplateFree(og1, backend);
        TemplateFree(og2, backend);     
        TemplateFree(udgavg, backend); // time-average solution (u, q, p) 
        TemplateFree(bouudgavg, backend); 
        TemplateFree(bouwdgavg, backend); 
        TemplateFree(bouuhavg, backend); 
    }             
};
using solstruct = solstructT<::dstype, ::Int>;

// Neutral scratch-arena owner (S5 step 3). Owns the big K backing buffer (grow-if-needed).
// The residual struct (res.K + the D/B/F/G/H views) and the solver (sys.v) hold NON-owning
// pointers/reserves into this buffer rather than owning it -- so the scratch memory is owned by
// a dedicated arena concern, not entangled in the residual data. Lives as a CDiscretization
// member (where the size is computed); res/solv/prec reach it only through res.K and the
// reserve* API, so no functional class owns the allocation.
template <class T = ::dstype, class I = ::Int>
struct scratcharenastructT {
    using dstype = T; using Int = I;
    dstype* buffer = nullptr;
    Int sz = 0;
    dstype* allocate(Int n, Int backend) {
        if (buffer == nullptr || sz != n) {   // grow-if-needed (same policy as EnsureTemplateAllocation)
            TemplateFree(buffer, backend);
            TemplateMalloc(&buffer, n, backend);
            sz = n;
        }
        return buffer;
    }
    void freememory(Int backend) { TemplateFree(buffer, backend); buffer = nullptr; sz = 0; }
};
using scratcharenastruct = scratcharenastructT<::dstype, ::Int>;

template <class T = ::dstype, class I = ::Int>
struct resstructT {
    using dstype = T; using Int = I;
    //dstype *R=nullptr;    // shared memory for all residual vectors
    dstype *Rqe=nullptr;  // element residual vector for q
    dstype *Rqf=nullptr;  // face residual vector for q   
    dstype *Rue=nullptr;  // element residual vector for u
    dstype *Ruf=nullptr;  // face residual vector for u
    dstype *Rq=nullptr;   // residual vector for q     
    dstype *Ru=nullptr;   // residual vector for u    
    dstype *Rh=nullptr;   // residual vector for uhat    

    dstype *dRq=nullptr;   // residual vector for q     
    dstype *dRu=nullptr;   // residual vector for u        
    dstype *dRh=nullptr;   // residual vector for uhat
    dstype *dRqe=nullptr;  // element residual vector for q
    dstype *dRqf=nullptr;  // face residual vector for q   
    dstype *dRue=nullptr;  // element residual vector for u
    dstype *dRuf=nullptr;  // face residual vector for u    

    dstype *Mass=nullptr; // store the mass matrix
    dstype *Minv=nullptr; // store the inverse of the mass matrix
    dstype *Mass2=nullptr; // store the mass matrix
    dstype *Minv2=nullptr; // store the inverse of the mass matrix
    // --- HDG/LDG local element-Jacobian blocks (the compact notation the assembly is written in) ---
    // The condensed local system per element is [D F; K H] [du; duh] = [Ru; Rh]; the LDG auxiliary q
    // (mass matrix Minv above) is eliminated by Schur substitution q = Minv*(C*u + E*uh), giving
    //   D += B*Minv*C,  F -= B*Minv*E,  K += G*Minv*C,  H -= G*Minv*E   (see uequation.hpp,
    // qequation.hpp, and docs/theory/block-diagonal-jacobian.md). C/E/B/G carry one slab per spatial
    // dimension (Cx,Cy,Cz ... indexed by e1 + d*ne). Sizes: n=npe*ncu (element-u), m=npf*nfe*ncu (trace).
    dstype *C=nullptr; // dq/du       block, per dim  [npe*npe*ne * nd]      (q from element u)
    dstype *E=nullptr; // dq/duhat    block, per dim  [npe*npf*nfe*ne * nd]  (q from trace uhat)
    dstype *D=nullptr; // dRue/du     block (n x n)   -- element-u vs element-u  (diagonal block)
    dstype *B=nullptr; // dRue/dq     block, per dim  -- element-u vs q  (contracts with Minv*C into D)
    dstype *F=nullptr; // dRue/duhat  block (n x m)   -- element-u vs trace
    dstype *G=nullptr; // dRh/dq      block, per dim  -- trace vs q  (contracts with Minv*C into K)
    dstype *K=nullptr; // dRh/du      block (m x n)   -- trace vs element-u
    dstype *H=nullptr; // dRh/duhat   block (m x m)   -- trace vs trace (the Schur-complemented diagonal)

    dstype *Ri=nullptr; // residual vector for uhat    
    dstype *Gi=nullptr; // store the diffusion matrix
    dstype *Ki=nullptr; // store the diffusion matrix
    dstype *Hi=nullptr; // store the diffusion matrix
    
    Int *ipiv=nullptr;    
    
    Int szRi=0, szHi=0, szKi=0, szGi=0, szP=0, szV=0;
    Int szipiv=0, szH=0, szK=0, szG=0, szF=0, szB=0, szD=0, szE=0, szC=0, szMass=0, szMinv=0, szMass2=0, szMinv2=0;
    Int szRq=0, szRu=0, szRh=0, szRuf=0, szRue=0, szRqf=0, szRqe=0;
    // 1 when F and H alias INTO the K block (the LDG block-Jacobi arena, AllocateLDGBlockJacobianMemory).
    // In that layout K is the only owned allocation; freememory must NOT TemplateFree(F)/(H) (they are
    // interior pointers of K -> freeing them is undefined behavior / a double free). 0 (default) in the
    // HDG layout, where H/F/K are each malloc'd separately and all three are freed.
    Int fhAliasesK = 0;

    int sizeofint() {return szipiv;}
    int sizeoffloat() {
      int sz = szH + szK + szG + szF + szB + szD + szE + szC + szMass + szMinv +
               szMass2 + szMinv2 + szRq + szRu + szRh + szRuf + szRue + szRqf + 
               szRqe + szHi + szKi + szGi + szRi;        
      return sz;
    }

    void printinfo()
    {
      printf("--------------- Residual Struct Information ----------------\n");
      printf("size of ipiv: %d\n", szipiv);
      printf("size of Rq: %d\n", szRq);
      printf("size of Ru: %d\n", szRu);
      printf("size of Rh: %d\n", szRh);
      // printf("size of dRq: %d\n", szdRq);
      // printf("size of dRu: %d\n", szdRu);
      // printf("size of dRh: %d\n", szdRh);
      printf("size of Mass: %d\n", szMass);
      printf("size of Minv: %d\n", szMinv);
      printf("size of Mass2: %d\n", szMass2);
      printf("size of Minv2: %d\n", szMinv2);
      printf("size of C: %d\n", szC);
      printf("size of E: %d\n", szE);
      printf("size of D: %d\n", szD);
      printf("size of B: %d\n", szB);
      printf("size of F: %d\n", szF);
      printf("size of G: %d\n", szG);
      printf("size of K: %d\n", szK);
      printf("size of H: %d\n", szH);
      printf("size of Ri: %d\n", szRi);  
      printf("size of Gi: %d\n", szGi);
      printf("size of Ki: %d\n", szKi);
      printf("size of Hi: %d\n", szHi);
      printf("size of int: %d\n", sizeofint());
      printf("size of float: %d\n", sizeoffloat());
    }

    // --- K-block as a scratch arena: callers RESERVE views instead of hard-coding offsets ---
    // The single owned K allocation doubles as a scratch arena: [0, szP) holds the
    // preconditioner; the [szP, szK) tail is shared scratch -- the assembly views
    // (D/B/F/G/H) live there during assembly, the GMRES Krylov vectors during the solve
    // (the two phases are temporally disjoint, which is why they may share the bytes).
    // Reserving the Krylov view through this method (rather than &res.K[res.szP] at the call
    // site) means no other class needs to know this layout: it decouples CSolver's sys.v from
    // CDiscretization's res, and lets a future change hand back a separate buffer transparently.
    // Non-owning: the returned pointer aliases K and must never be freed (keep sys.szv == 0).
    dstype* reserveKrylovScratch(Int szRequest)
    {
        if (K != nullptr && szP + szRequest > szK)
            printf("WARNING: reserveKrylovScratch overruns the res.K arena (szP=%d + req=%d > szK=%d)\n",
                   (int)szP, (int)szRequest, (int)szK);
        return &K[szP];
    }

    // Cursor-based reservation for the sequential assembly views (D/B/F/G/H) laid out in the K
    // arena. resetKArena(start) sets the cursor, then each reserveView(size) returns the next
    // slice and advances -- centralizing the offset arithmetic that used to be spelled out
    // inline as &K[start + dSize + bSize + ...] at every assignment, and warning on overrun.
    // Non-owning views into K (freed with K). See AllocateLDGBlockJacobianMemory / the HDG branch.
    Int kArenaCursor = 0;
    void resetKArena(Int start) { kArenaCursor = start; }
    dstype* reserveView(Int size)
    {
        dstype* p = &K[kArenaCursor];
        kArenaCursor += size;
        if (K != nullptr && kArenaCursor > szK)
            printf("WARNING: K arena view overruns (cursor=%d > szK=%d)\n", (int)kArenaCursor, (int)szK);
        return p;
    }

    void freememory(Int backend)
    {
        TemplateFree(Rq, backend);    
        TemplateFree(Ru, backend);    
        TemplateFree(Rh, backend);    
      #ifdef HAVE_ENZYME                   
        TemplateFree(dRq, backend);   
        TemplateFree(dRu, backend);   
        TemplateFree(dRh, backend);   
      #endif                                                
        // Size-guarded frees: temporary mass matrices are freed early (massinv/qEquation clear the
        // size marker), so skip here when already released to avoid a double free of a dangling ptr.
        if (szMass > 0)  TemplateFree(Mass, backend);
        if (szMinv > 0)  TemplateFree(Minv, backend);
        if (szMass2 > 0) TemplateFree(Mass2, backend);
        if (szMinv2 > 0) TemplateFree(Minv2, backend);
        TemplateFree(C, backend);
        TemplateFree(E, backend);
        // F and H are owned only in the HDG layout; in the LDG block-Jacobi arena they alias into
        // K (fhAliasesK==1) and must not be freed (freeing K reclaims the whole block).
        if (!fhAliasesK && szF > 0) TemplateFree(F, backend); else F = nullptr;
        K = nullptr;  // non-owning view into the scratch arena (owned/freed by CDiscretization::scratch, S5 step 3)
        if (!fhAliasesK && szH > 0) TemplateFree(H, backend); else H = nullptr;
        TemplateFree(Gi, backend);
        TemplateFree(Ki, backend);
        TemplateFree(Hi, backend);
        TemplateFree(Ri, backend);
        TemplateFree(ipiv, backend);
    }                        
};
using resstruct = resstructT<::dstype, ::Int>;

template <class T = ::dstype, class I = ::Int>
struct tempstructT {
    using dstype = T; using Int = I;
    dstype *tempn=nullptr;
    dstype *tempg=nullptr;
    dstype *buffrecv=nullptr;
    dstype *buffsend=nullptr;
    dstype *bufffacerecv=nullptr;
    dstype *bufffacesend=nullptr;
    
    int sztempn=0, sztempg = 0, szbuffrecv=0, szbuffsend=0, szbufffacerecv=0, szbufffacesend=0;

    int sizeofint() {return 0;}
    int sizeoffloat() 
    {
      int sz = sztempn + sztempg + szbuffrecv + szbuffsend + szbufffacerecv + szbufffacesend;
      return sz;
    }

    void printinfo()
    {
      printf("--------------- Temp Struct Information ----------------\n");
      printf("size of tempn: %d\n", sztempn);
      printf("size of tempg: %d\n", sztempg);
      printf("size of buffrecv: %d\n", szbuffrecv);
      printf("size of buffsend: %d\n", szbuffsend);
      printf("size of bufffacerecv: %d\n", szbufffacerecv);
      printf("size of bufffacesend: %d\n", szbufffacesend);
      printf("size of int: %d\n", sizeofint());
      printf("size of float: %d\n", sizeoffloat());
    }

    void freememory(Int backend)
    {
        TemplateFree(tempn, backend); 
        //TemplateFree(tempg, backend); 
        TemplateFree(buffrecv, backend); 
        TemplateFree(buffsend, backend); 
        TemplateFree(bufffacerecv, backend); 
        TemplateFree(bufffacesend, backend); 
    }            
};
using tempstruct = tempstructT<::dstype, ::Int>;

// Templated on the scalar precision T and index type I (Phase 1 of dstype->template threading, see
// docs/internals/precision-threading.md). The member `using` aliases shadow the global dstype/Int so
// the struct body below is UNCHANGED; with the default args the type is byte-identical to before.
template <class T = ::dstype, class I = ::Int>
struct sysstructT {
    using dstype = T; using Int = I;
    Int backend;
    Int *ipiv=nullptr;

    dstype *x=nullptr; 
    dstype *u=nullptr;
    dstype *r=nullptr;
    dstype *b=nullptr;
    dstype *v=nullptr;
    dstype *randvect=nullptr;
    dstype *q=nullptr;
    dstype *p=nullptr;
    
    // unified memory for GMRES solver
    dstype *tempmem=nullptr;
    dstype *lam=nullptr;
    //dstype *normcu=nullptr;    
    
    // for DIRK schemes
    dstype *utmp=nullptr;
    dstype *wtmp=nullptr;
    
    // previous solutions for time-dependent problems
    dstype *udgprev=nullptr; // 
    dstype *udgprev1=nullptr; // 
    dstype *udgprev2=nullptr; // 
    dstype *udgprev3=nullptr; //         
    dstype *wprev=nullptr; // 
    dstype *wprev1=nullptr; // 
    dstype *wprev2=nullptr; // 
    dstype *wprev3=nullptr; // 
        
    Int szipiv = 0;    
    Int szx=0, szu=0, szr=0, szb=0, szv=0, szq=0, szp=0;
    Int szrandvect=0, sztempmem=0, szlam=0, szPTCmatrix=0, szutmp=0, szwtmp=0;    
    Int szudgprev = 0, szudgprev1 = 0, szudgprev2 = 0, szudgprev3 = 0;
    Int szwprev = 0, szwprev1 = 0, szwprev2 = 0, szwprev3 = 0;

    dstype alpha=1.0; // linesearch alpha
    
    int sizeofint() { return szipiv; }
    int sizeoffloat() {
      int sz = szx + szu + szr + szb + szv + szq + szp +
              szrandvect + sztempmem + szPTCmatrix + szutmp + szwtmp + 
              szudgprev + szudgprev1 + szudgprev2 + szudgprev3 + szwprev + 
              szwprev1 + szwprev2 + szwprev3;
      return sz;
    }

    void printinfo()
    {
      printf("--------------- Sys Struct Information ----------------\n");
      printf("size of ipiv: %d\n", szipiv);
      printf("size of x: %d\n", szx);
      printf("size of u: %d\n", szu);
      printf("size of r: %d\n", szr);
      printf("size of b: %d\n", szb);
      printf("size of v: %d\n", szv);
      printf("size of q: %d\n", szq);
      printf("size of p: %d\n", szp);
      printf("size of randvect: %d\n", szrandvect);
      printf("size of tempmem: %d\n", sztempmem);
      printf("size of lam: %d\n", szlam);
      printf("size of PTCmatrix: %d\n", szPTCmatrix);
      printf("size of utmp: %d\n", szutmp);
      printf("size of wtmp: %d\n", szwtmp);
      printf("size of udgprev: %d\n", szudgprev);
      printf("size of udgprev1: %d\n", szudgprev1);
      printf("size of udgprev2: %d\n", szudgprev2);
      printf("size of udgprev3: %d\n", szudgprev3);
      printf("size of wprev: %d\n", szwprev);
      printf("size of wprev1: %d\n", szwprev1);
      printf("size of wprev2: %d\n", szwprev2);
      printf("size of wprev3: %d\n", szwprev3);
      printf("size of int: %d\n", sizeofint());
      printf("size of float: %d\n", sizeoffloat());
    }

    void freememory(Int backend)
    {
        CPUFREE(lam);  
        CPUFREE(ipiv);  

        TemplateFree(x, backend); 
        TemplateFree(u, backend); 
        TemplateFree(r, backend); 
        TemplateFree(b, backend); 
        if (szv>0) TemplateFree(v, backend); 
        else v = nullptr;
        TemplateFree(q, backend); 
        TemplateFree(p, backend); 
        TemplateFree(randvect, backend);
        if (backend <= 1)
          TemplateFree(tempmem, backend);    
        else if (backend==2) {
#ifdef HAVE_CUDA                
          cudaFreeHost(tempmem);      
#endif                         
        }        
        else if (backend == 3) {
#ifdef HAVE_HIP
            CHECK(hipHostFree(tempmem)); // Free pinned host memory with HIP
#endif
        }        
        TemplateFree(utmp, backend);            
        TemplateFree(wtmp, backend);             
        TemplateFree(udgprev, backend);  
        TemplateFree(udgprev1, backend);  
        TemplateFree(udgprev2, backend);  
        TemplateFree(udgprev3, backend);  
        TemplateFree(wprev, backend);  
        TemplateFree(wprev1, backend);  
        TemplateFree(wprev2, backend);  
        TemplateFree(wprev3, backend);  
    }                
};
// Default instantiation keeps the name `sysstruct` meaning exactly what it did (double/int today).
using sysstruct = sysstructT<::dstype, ::Int>;

template <class T = ::dstype, class I = ::Int>
struct precondstructT {
    using dstype = T; using Int = I;
    Int backend;
    
    dstype *W=nullptr; 
    dstype *U=nullptr; 
    Int *ipiv=nullptr;
    
    Int szipiv = 0, szW = 0, szU = 0;
    int sizeofint() { return szipiv; }
    int sizeoffloat() {
      int sz = szW + szU;
      return sz;
    }

    void printinfo()
    { 
      printf("--------------- Precond Struct Information ----------------\n");
      printf("size of ipiv: %d\n", szipiv);
      printf("size of W: %d\n", szW);
      printf("size of U: %d\n", szU);
      printf("size of int: %d\n", sizeofint());
      printf("size of float: %d\n", sizeoffloat());
    }

    void freememory(Int backend)
    {
        TemplateFree(W, backend); 
        TemplateFree(U, backend); 
        TemplateFree(ipiv, backend); 
    }            
};
using precondstruct = precondstructT<::dstype, ::Int>;

// Grouped mutable transient state of the iterative solver + reduced-basis preconditioner.
// Extracted from commonstruct (C1) and then OWNED BY CSolver (Stage 1 of the internal
// separation): mutable runtime state lives in the solver object, not in the setup/config
// commonstruct. Default-initialized (no setup read needed) so the CSolver member is valid
// on construction on every backend. (The set-once PTC parameter moved to solverparams,
// where the rest of the solver configuration lives.)
struct solverstatestruct {
    Int RBcurrentdim = 0;  // current dimension of the reduced basis space
    Int RBremovedind = 0;  // RB vector to be removed and replaced with a new vector
    Int Wcurrentdim = 0;   // current dimension of W
    Int linearSolverIter = 0;     // current linear-solver iteration
    Int nonlinearSolverIter = 0;  // current nonlinear-solver iteration
    dstype linearSolverTolFactor = 1.0;  // adaptive linear-solver tolerance scaling
    dstype linearSolverRelError = 0.0;   // achieved linear-solver relative residual
};

// One quantity-of-interest instance: a named QoI the backend computes and writes. Several
// instances may coexist in a single program (the "QoI template, many instances" model). Each
// instance owns a contiguous [offset, offset+ncomp) slice of its model QoI-kernel output;
// volume instances integrate over the domain, boundary instances over one boundary id.
struct qoiinstancestruct {
    std::string name = "QoI";  // output-column header prefix
    Int kind = 0;              // 0: volume (domain) QoI; 1: boundary (surface) QoI
    Int boundary = 0;          // boundary id to integrate over (kind==1); unused for volume
    Int offset = 0;            // first component in the model QoI-kernel output vector
    Int ncomp = 0;             // number of components this instance owns
};

// Mutable time-stepping runtime state, advanced in the step/stage loop. Grouped out of
// commonstruct (C3) alongside solverstate so the simulation's mutable state is named and
// isolated from the read-only configuration. Access via common.timestate.<field>.
struct timestatestruct {
    Int currentstep;   // current time step (loop counter)
    Int currentstage;  // current time stage (loop counter)
    dstype dtfactor;   // time-derivative factor for udg (set per stage)
    dstype time;       // current simulation time
};

// Physics / model configuration: stabilization, viscosity/SGS, ALE, artificial viscosity,
// rotating frame, source/model-type flags, plus the AV ramp factor. Grouped out of
// commonstruct (C3). Access via common.physicsparams.<field>.
struct physicsparamsstruct {
    Int appname;          // model identifier (0: Euler; 1: Compressible Navier-Stokes; ...)
    Int source;           // source function flag
    Int convStabMethod;   // convective stabilization (0 const tau, 1 Lax-Friedrichs, 2 Roe)
    Int diffStabMethod;   // diffusive stabilization
    Int rotatingFrame;    // rotating-frame flag
    Int viscosityModel;   // viscosity law
    Int SGSmodel;         // sub-grid-scale model
    Int ALEflag;          // Arbitrary Lagrangian-Eulerian formulation flag
    Int ncAV;             // number of artificial-viscosity components
    Int AVsmoothingIter;  // AV smoothing iterations
    Int frozenAVflag;     // freeze AV per nonlinear solve
    Int AVdistfunction=0; // AV distance-function flag
    dstype rampFactor;    // AV flux ramp factor (advanced over steps)
    dstype tau0=0.0;      // initial stabilization parameter
};

// Time-integration / problem-evolution configuration: temporal scheme + order + stages, time-step
// count, dual-time (DAE) parameters, and the problem-character flags (time-dependent, wave, linear,
// sub-problem, time-derivative function). Grouped out of commonstruct (C3). Access via
// common.timeparams.<field>. (The mutable step/stage counters live in timestate.)
struct timeparamsstruct {
    Int temporalScheme;  // 0: DIRK; 1: BDF; 2: ERK
    Int torder;          // temporal accuracy order
    Int tstages;         // DIRK stages
    Int tsteps;          // number of time steps
    Int dae_steps=0;     // number of dual time steps
    Int tdep;            // 0: steady-state; 1: time-dependent
    Int wave;            // wave problem
    Int tdfunc;          // time-derivative function flag
    Int linearProblem;   // 0: nonlinear; 1: linear
    Int subproblem=0;
    dstype dae_alpha=1.0;
    dstype dae_beta=0.0;
    dstype dae_gamma=0.0;
    dstype dae_epsilon=0.0;
};

// Iterative-solver configuration: linear/nonlinear solver type, iteration caps, tolerances,
// GMRES/matvec/preconditioner settings, and reduced-basis/W max dimensions. Grouped out of
// commonstruct (C3). Access via common.solverparams.<field>. (The mutable per-solve counters
// live in solverstate.)
struct solverparamsstruct {
    Int linearSolver;            // 0: GMRES; 1: CG; ...
    Int nonlinearSolver;
    Int linearSolverMaxIter;
    Int nonlinearSolverMaxIter;
    Int matvecOrder;
    Int gmresRestart;
    Int gmresOrthogMethod;
    Int preconditioner;          // 0: low-rank; 1: reduced basis
    Int precMatrixType;          // 0: identity; 1: inverse mass matrix
    Int ptcMatrixType;           // 0: identity; 1: mass matrix
    Int RBdim;                   // max dimension of the reduced-basis space
    Int Wdim;                    // max dimension of W
    dstype matvecTol;
    dstype linearSolverTol;
    dstype nonlinearSolverTol;
    dstype PTCparam;             // pseudo-transient-continuation parameter (set-once config)
};

// QoI / visualization-output configuration: visualization component counts (scalar/vector/tensor),
// surface/volume QoI counts, the Paraview flag, the boundary-to-save index, the QoI accumulation
// buffers, and the registered QoI-instance list (C2). Grouped out of commonstruct (C3). Access via
// common.qoiparams.<field>.
template <class T = ::dstype, class I = ::Int>
struct qoiparamsstructT {
    using dstype = T; using Int = I;
    Int nsca;             // visualization scalar-field components
    Int nvec;             // visualization vector-field components
    Int nten;             // visualization tensor-field components
    Int nsurf;            // surface QoI / storage components
    Int nvqoi;            // volume QoI components
    Int saveParaview = 0; // enable Paraview output
    Int ibs;              // boundary index to save solution
    dstype* qoivolume=nullptr;  // volume-QoI accumulation buffer
    dstype* qoisurface=nullptr; // surface-QoI accumulation buffer
    std::vector<qoiinstancestruct> qoiinstances;  // registered QoI instances (default: 1 domain + 1 boundary)
};
using qoiparamsstruct = qoiparamsstructT<::dstype, ::Int>;

// Interface-coupling configuration: coupled interface/condition/boundary flags, external
// uhat/fhat/stabilization function flags, the external-force call flag, and interface/coupled-
// element counts. Grouped out of commonstruct (C3). Access via common.couplingparams.<field>.
// (The raw interface arrays vindx/interfacefluxmap/intepartpts are shared with appstruct and
// left in place; the wall-model and synthetic-turbulence fields are separate concerns.)
struct couplingparamsstruct {
    Int ncuext;                   // number of components of uext (external/coupling)
    Int coupledinterface;
    Int coupledcondition;
    Int coupledboundarycondition;
    Int FextCall=0;               // external-force call flag
    Int ncie;                     // number of coupled interface elements
    Int extUhat=0;                // external uhat function flag
    Int extFhat=0;                // external fhat function flag
    Int extStab=0;                // external stabilization function flag
    Int ninterfacefaces=0;        // number of interface faces
    Int ndofuhatinterface=0;
    Int nintfaces;
    Int nvindx;
};

// Output / checkpoint / IO configuration: solution-save frequencies and options, restart offset,
// time-averaged-solution flags, residual-norm logging, file offset, and debug mode. Grouped out of
// commonstruct (C3). Access via common.outputparams.<field>.
struct outputparamsstruct {
    Int saveSolFreq;       // steps between solution saves
    Int saveSolOpt;        // solution-save option
    Int saveRestart=200;   // steps between restart saves
    Int saveSolBouFreq=0;  // steps between boundary-solution saves
    Int timestepOffset=0;  // timestep offset to restart the simulation
    Int compudgavg=1;      // compute time-averaged solution
    Int readudgavg=0;      // read time-averaged solution from file
    Int saveResNorm=0;     // log residual norms
    Int fileoffset;        // file/rank offset for IO
    Int debugMode;         // debug output flag
};

// Wall-model configuration: model-ID / boundary / distance tables and their sizes (the working
// copies; the raw tables are mirrored in appstruct). Grouped out of commonstruct (C3). Access via
// common.wallmodelparams.<field>.
template <class T = ::dstype, class I = ::Int>
struct wallmodelparamsstructT {
    using dstype = T; using Int = I;
    Int nwm=0;               // number of wall-model configurations
    Int szwmModelIDs=0;
    Int szwmBoundaries=0;
    Int szwmDistances=0;
    Int *wmModelIDs=nullptr;
    Int *wmBoundaries=nullptr;
    dstype* wmDistances=nullptr;
};
using wallmodelparamsstruct = wallmodelparamsstructT<::dstype, ::Int>;

// Synthetic-turbulence-generation (STG) configuration: number of modes, inlet-boundary count, and
// the inlet-boundary index table (working copy; raw STG data lives in appstruct). Grouped out of
// commonstruct (C3). Access via common.stgparams.<field>.
struct stgparamsstruct {
    Int stgNmode=0;          // number of synthetic-turbulence modes
    Int nstgib;              // number of STG inlet boundaries
    Int* stgib=nullptr;      // STG inlet-boundary index table
};

// Derived size "view": degree-of-freedom counts that are cross-products of the base dimensions
// (npe/npf x nc-family x ne/nf), computed once in setstructs and read throughout the solver. Grouped
// out of commonstruct (C3) so the ~17 derived ndof* counts have one home distinct from the base
// dimensions (nd, nc-family, npe, ne, ...), which stay top-level as the fundamental shape. Access
// via common.sizes.<field>. NOTE: these are stored, not recomputed on access -- ndof1* switch
// between ne1 and ne by mesh branch (setstructs.cpp), and ndofucg/ndofbou are not simple products,
// so a naive computed view would be incorrect.
struct sizesstruct {
    Int ndof;       // dofs of u   = npe*ncu*ne
    Int ndofq;      // dofs of q   = npe*ncq*ne
    Int ndofw;      // dofs of w   = npe*ncw*ne
    Int ndofuhat;   // dofs of uhat= npf*ncu*nf
    Int ndofudg;    // dofs of udg = npe*nc*ne
    Int ndofsdg;    // dofs of sdg = npe*ncs*ne
    Int ndofodg;    // dofs of odg = npe*nco*ne
    Int ndofedg;    // dofs of edg = npe*nce*ne
    Int ndofbou=0;  // dofs on saved boundary (conditional)
    Int ndofucg;    // dofs of ucg = mesh.nsize[12]-1
    Int ndof1;      // interior(+interface) variant of ndof   (ne1 or ne by mesh branch)
    Int ndofq1;     // interior(+interface) variant of ndofq
    Int ndofw1;     // interior(+interface) variant of ndofw
    Int ndofudg1;   // interior(+interface) variant of ndofudg
    Int ndofsdg1;   // interior(+interface) variant of ndofsdg
    Int ndofodg1;   // interior(+interface) variant of ndofodg
    Int ndofedg1;   // interior(+interface) variant of ndofedg
};

// Component counts of the DG fields: how many scalar components each field (u, q, w, o/odg, uhat,
// xdg, sdg, edg, PTC monitor) carries, plus the stabilization length. The fundamental "field shape"
// the kernels/drivers template their loops on. Grouped out of commonstruct (C3/S3) into an
// intermediate struct so driver/kernel signatures can take just the component counts instead of the
// whole commonstruct. Access via common.components.<field>.
struct componentsstruct {
    Int nc;    // components of (u, q) packed
    Int ncu;   // components of u
    Int ncq;   // components of q
    Int ncw;   // components of w
    Int nco;   // components of o (odg / auxiliary "other DG")
    Int nch;   // components of uhat
    Int ncx;   // components of xdg (coordinates)
    Int ncs;   // components of sdg (source)
    Int nce;   // components of edg (outputs)
    Int ncm;   // components of the PTC monitor function
    Int ntau;  // stabilization length
};

// Model-local solution layout: the per-point field SHAPE the model determines -- the solution-field
// component counts and (scaffolded) per-component names. This is the "data layout component of the
// model": LOCAL/per-point and model-changing, distinct from geometry/mesh sizes (nd / npe / ne / ncx),
// which change with the app and live in gridstruct / meshsizesstruct. Host-only descriptor; the
// kernels still read the raw counts from common.components, so adding this costs zero kernel churn.
// G4 of the model-decoupling plan: the counts are populated now; names are auto-labeled ("u0","u1",
// ...) as a scaffold and will carry real model-supplied names once the wire format emits them (Phase D).
struct solutionlayoutstruct {
    // counts (mirrored from componentsstruct -- the solution's per-point field components)
    Int nc=0;    // (u, q) packed
    Int ncu=0;   // u    (primary unknowns)
    Int ncq=0;   // q    (gradient/auxiliary, = ncu * nd)
    Int ncw=0;   // w    (wave/auxiliary)
    Int nco=0;   // o    (other read-only auxiliary)
    Int nch=0;   // uhat (trace)
    Int nce=0;   // output fields
    // meaning: per-component names/roles (auto-labeled scaffold; Phase D fills real names)
    std::vector<std::string> ufields, qfields, wfields, ofields, efields;
};

// Populate a solution layout from the raw component counts, auto-labeling each field's components.
inline void buildSolutionLayout(solutionlayoutstruct& layout, const componentsstruct& c)
{
    layout.nc = c.nc; layout.ncu = c.ncu; layout.ncq = c.ncq; layout.ncw = c.ncw;
    layout.nco = c.nco; layout.nch = c.nch; layout.nce = c.nce;
    auto autolabel = [](std::vector<std::string>& v, const char* prefix, Int n) {
        v.clear();
        for (Int i = 0; i < n; ++i) v.push_back(std::string(prefix) + std::to_string(i));
    };
    autolabel(layout.ufields, "u", c.ncu);
    autolabel(layout.qfields, "q", c.ncq);
    autolabel(layout.wfields, "w", c.ncw);
    autolabel(layout.ofields, "o", c.nco);
    autolabel(layout.efields, "e", c.nce);
}

// Reference-element / discretization sizes: spatial dimension, element & node types, polynomial
// and quadrature orders, and the per-element/face node & Gauss-point counts (the master-element
// shape the kernels loop over). Grouped out of commonstruct (C3/S3) into an intermediate struct
// alongside componentsstruct. Access via common.grid.<field>.
struct gridstruct {
    Int nd;          // spatial dimension
    Int elemtype;
    Int nodetype;
    Int porder;      // solution polynomial degree
    Int pgauss;      // Gauss-quadrature degree
    Int npe;         // nodes on master element
    Int npf;         // nodes on master face
    Int nge;         // Gauss points on master element
    Int ngf;         // Gauss points on master face
    Int np1d;
    Int ng1d;
    Int curvedMesh;  // curved-mesh flag
};

// Mesh partition sizes: element / face / vertex counts and the block counts the kernels iterate
// over, including the interior / interior+interface / +exterior splits used by the coupling and
// halo paths. Grouped out of commonstruct (C3/S3) into an intermediate struct. Access via
// common.meshsizes.<field>.
struct meshsizesstruct {
    Int maxnbc;  // max number of boundary conditions
    Int ne;      // total elements
    Int nf;      // total faces
    Int nv;      // vertices
    Int nfe;     // faces per element
    Int nbe;     // element blocks
    Int neb;     // max elements per block
    Int nbf;     // face blocks
    Int nfb;     // max faces per block
    Int nbe0;    // element blocks: interior
    Int nbe1;    // element blocks: interior+interface
    Int nbe2;    // element blocks: interior+interface+exterior
    Int nbf0;    // face blocks: interior
    Int nbf1;    // face blocks: interior+interface
    Int ne0;     // interior elements
    Int ne1;     // interior+interface elements
    Int ne2;     // interior+interface+exterior elements
    Int nf0;     // interior faces
};

// CRS index/numbering arrays for the LDG block-Jacobian preconditioner assembly:
// ind_* map element/face block entries into the sparse operator; num_* are per-row
// counts; L*/U* are the lower/upper-triangular block-factorization index+count
// arrays. Allocated in crs_init (discretization.cpp). Grouped out of commonstruct
// (S3-style) into one named concern; access via common.bjindex.<field>.
struct blockjacindexstruct {
    Int* ind_ii=nullptr;
    Int* ind_ji=nullptr;
    Int* ind_il=nullptr;
    Int* ind_jl=nullptr;
    Int* num_ji=nullptr;
    Int* num_jl=nullptr;
    Int* Lnum_ji=nullptr;
    Int* Lind_ji=nullptr;
    Int* Unum_ji=nullptr;
    Int* Uind_ji=nullptr;
};

template <class T = ::dstype, class I = ::Int>
struct commonstructT {
    using dstype = T; using Int = I;
    using qoiparamsstruct       = qoiparamsstructT<T, I>;
    using wallmodelparamsstruct = wallmodelparamsstructT<T, I>;
    meshsizesstruct meshsizes;              // mesh partition element/face/block counts (see above)
    gridstruct grid;                        // reference-element/discretization sizes (see above)
    componentsstruct components;            // DG-field component counts (see above)
    solutionlayoutstruct layout;            // model-local solution layout: field counts + names (G4)
    sizesstruct sizes;                      // derived degree-of-freedom counts (see above)
    wallmodelparamsstruct wallmodelparams;  // wall-model configuration (see above)
    stgparamsstruct stgparams;              // synthetic-turbulence-generation config (see above)
    outputparamsstruct outputparams;      // output/checkpoint/IO configuration (see above)
    couplingparamsstruct couplingparams;  // interface-coupling configuration (see above)
    qoiparamsstruct qoiparams;          // QoI/visualization-output configuration (see above)
    timeparamsstruct timeparams;        // time-integration/problem-evolution config (see above)
    solverparamsstruct solverparams;    // iterative-solver configuration (see above)
    physicsparamsstruct physicsparams;  // physics/model configuration (see above)
    // solverstate (mutable solver/preconditioner runtime state) was lifted out of commonstruct
    // into CSolver (Stage 1 of the internal separation) -- setup/config stays here, runtime state
    // lives in the owning solver object.
    timestatestruct timestate;      // mutable time-stepping runtime state (see above)
    // Runtime model ABI for the unified templated FEM code (M == AbiAdapter path): set by
    // CDiscretization to point at its driver_abi, so the no-driver_abi kernel-driver
    // overloads can reach the ABI without threading it through every call.
    ExasimDriverABI* driver_abi = nullptr;
    std::string exasimpath = "";
    std::string filein;       // Name of binary file with input data
    std::string fileout;      // Name of binary file to write the solution            
    
    Int backend;   // 0: Serial; 1: OpenMP; 2: CUDA  
    
    Int mpiRank;  // MPI rank      
    Int mpiProcs;    // number of MPI ranks
    Int nomodels; // number of models
    Int enzyme=0;
    

    Int ppdegree=0; // polynomial preconditioner degree
    Int isd=0; 
            
    Int nse=0;  // number of superelements
    Int nese=0; // number of elements per superelement
    Int nfse=0; // number of faces per superelement
    Int nnz=0;
    
    
        
    
    Int modelnumber;      // model number
    Int builtinmodelID=0; // model ID    
    Int matrixformat=0;
    
    Int spatialScheme;   /* 0: HDG; 1: EDG; 2: IEDG, HEDG */
                                        //  0: No SGS model. 1: Static Smagorinsky/Yoshizawa/Knight model. 
                                        //  2: Static WALE/Yoshizawa/Knight model. 3: Static Vreman/Yoshizawa/Knight model.
                                        //  4: Dynamic Smagorinsky/Yoshizawa/Knight model.        
                                //   0: AV not frozen, evaluated as part of residual
                                //   1: AV frozen, evluated once per solve (default)   
    Int read_uh = 0;

    Int runmode;
    
            
    Int* eblks=nullptr; // element blocks
    Int* fblks=nullptr; // face blocks   
    Int* ncarray=nullptr;
    Int* nboufaces=nullptr;
    Int* nextfaces=nullptr;

    // ---- partition-aware block accessors ------------------------------------
    //
    // A rank can legitimately own ZERO element or face blocks: the mesh is finite,
    // so past some rank count at least one rank gets none of it. The raw idioms
    //
    //     fblks[3*(meshsizes.nbf-1)+1]     // "one past my last face"
    //     fblks[3*0]                       // "my first face"
    //
    // then evaluate to fblks[-2] / read through a null base. That is exactly how
    // poisson2d died at np=48: AddressSanitizer reported SEGV at
    // 0xfffffffffffffff8, i.e. -8, which is fblks[-2] for a 4-byte Int.
    //
    // Prefer these accessors to indexing eblks/fblks directly. They put the
    // empty-rank check in ONE place instead of at every call site, and cost one
    // predictable integer compare at setup / per-assembly granularity -- never
    // inside a Kokkos kernel or an inner loop.
    //
    // Convention: an empty rank yields first==last==0, so a loop written as
    // `for (i = r.first; i < r.last; ++i)` is naturally a no-op there.

    struct BlockRange {
        Int  first = 0;    // first entity (already converted from 1-based)
        Int  last  = 0;    // one past the last entity
        Int  ib    = 0;    // boundary/interface tag
        bool valid = false;
        Int  count() const { return last - first; }
    };

    bool hasFaceBlocks() const { return fblks != nullptr && meshsizes.nbf  > 0; }
    bool hasElemBlocks() const { return eblks != nullptr && meshsizes.nbe  > 0; }
    bool hasInteriorElemBlocks() const { return eblks != nullptr && meshsizes.nbe1 > 0; }

    BlockRange faceBlock(Int j) const {
        if (!hasFaceBlocks() || j < 0 || j >= meshsizes.nbf) return BlockRange{};
        return BlockRange{ fblks[3*j]-1, fblks[3*j+1], fblks[3*j+2], true };
    }
    BlockRange elemBlock(Int j) const {
        if (!hasElemBlocks() || j < 0 || j >= meshsizes.nbe) return BlockRange{};
        return BlockRange{ eblks[3*j]-1, eblks[3*j+1], eblks[3*j+2], true };
    }

    // Span of everything this rank owns; all zero when it owns nothing.
    Int firstFace() const { return hasFaceBlocks() ? fblks[0]-1 : 0; }
    Int lastFace()  const { return hasFaceBlocks() ? fblks[3*(meshsizes.nbf-1)+1] : 0; }
    Int firstElem() const { return hasElemBlocks() ? eblks[0]-1 : 0; }
    Int lastInteriorElem() const {
        return hasInteriorElemBlocks() ? eblks[3*(meshsizes.nbe1-1)+1] : 0;
    }
    
    Int nnbsd; // number of neighboring subdomains
    Int nelemsend;
    Int nelemrecv;
    Int szinterfacefluxmap;
    Int szcartgridpart;
    Int* nbsd=nullptr; // neighboring subdomains
    Int* elemsend=nullptr;
    Int* elemrecv=nullptr;       
    Int* elemsendpts=nullptr;
    Int* elemrecvpts=nullptr;        
    Int *vindx=nullptr;
    Int *interfacefluxmap=nullptr;
    Int *cartgridpart=nullptr;
    Int *boundaryConditions=nullptr;
    Int *intepartpts=nullptr;
    
    Int nnbintf;
    Int nfacesend;
    Int nfacerecv;
    Int* nbintf=nullptr;
    Int* facesend=nullptr;
    Int* facerecv=nullptr;       
    Int* facesendpts=nullptr;
    Int* facerecvpts=nullptr;        
        
    blockjacindexstruct bjindex;  // LDG block-Jacobian CRS index/numbering arrays (see above)

    dstype  timing[128];
    dstype* dt=nullptr;
    dstype* dae_dt=nullptr;
    dstype* DIRKcoeff_c=nullptr;
    dstype* DIRKcoeff_d=nullptr;
    dstype* DIRKcoeff_t=nullptr;
    dstype* BDFcoeff_c=nullptr;
    dstype* BDFcoeff_t=nullptr;    

    cudaEvent_t eventHandle;
    cublasHandle_t cublasHandle;
    
#ifdef  HAVE_MPI
    MPI_Request * requests;
    MPI_Status * statuses;
#endif
    
    void printinfo()
    {
      printf("--------------- Common Struct Information ----------------\n");
      printf("backend: %d\n", backend);   
      printf("number of MPI ranks: %d\n", mpiProcs);   
      printf("number of models: %d\n", nomodels);               
      printf("number of compoments of (u, q): %d\n", components.nc);   
      printf("number of compoments of u: %d\n", components.ncu);   
      printf("number of compoments of q: %d\n", components.ncq);   
      printf("number of compoments of w: %d\n", components.ncw);   
      printf("number of compoments of v: %d\n", components.nco);   
      printf("number of compoments of uhat: %d\n", components.nch);   
      printf("number of compoments of x: %d\n", components.ncx);   
      printf("number of compoments of s: %d\n", components.ncs);   
      printf("number of compoments of outputs: %d\n", components.nce);    
      printf("spatial dimension: %d\n", grid.nd);   
      printf("spatial scheme: %d\n", spatialScheme);        
      printf("element type: %d\n", grid.elemtype);   
      printf("node type: %d\n", 1);   
      printf("polynomial degree: %d\n", grid.porder);   
      printf("gauss quadrature degree: %d\n", grid.pgauss); 
      printf("number of nodes on master element: %d\n", grid.npe); 
      printf("number of gauss points on master element: %d\n", grid.nge); 
      printf("number of nodes on master face: %d\n", grid.npf); 
      printf("number of gauss points on master face: %d\n", grid.ngf); 
      printf("temporal scheme: %d\n", timeparams.temporalScheme);   
      printf("temporal order: %d\n", timeparams.torder);   
      printf("number of DIRK stages: %d\n", timeparams.tstages);   
      printf("number of time steps: %d\n", timeparams.tsteps);   
      
      printf("total number of elements: %d\n", meshsizes.ne);   
      printf("number of interior elements: %d\n", meshsizes.ne0);   
      printf("number of interior+interface elements: %d\n", meshsizes.ne1);   
      printf("number of interior+interface+exterior elements: %d\n", meshsizes.ne2);   
      printf("total number of faces: %d\n", meshsizes.nf);   
      printf("number of interior faces: %d\n", meshsizes.nf0);   
      
      printf("number of faces per elements: %d\n", meshsizes.nfe);
      printf("number of blocks for elements: %d\n", meshsizes.nbe);
      printf("number of blocks for faces: %d\n", meshsizes.nbf);        
      printf("maximum number of faces per block: %d\n", meshsizes.nfb);
      printf("number of blocks for interior elements: %d\n", meshsizes.nbe0);
      printf("number of blocks for interior+interface elements: %d\n", meshsizes.nbe1);
      printf("number of blocks for interior+interface+exterior elements: %d\n", meshsizes.nbe2);
      printf("number of blocks for interior faces: %d\n", meshsizes.nbf0);
      printf("number of blocks for interior+interface faces: %d\n", meshsizes.nbf1);
      printf("number of interface faces: %d\n", couplingparams.ninterfacefaces);

      printf("number of degrees of freedom of u: %d\n", sizes.ndof);   
      printf("number of degrees of freedom of q: %d\n", sizes.ndofq);   
      printf("number of degrees of freedom of w: %d\n", sizes.ndofw);   
      printf("number of degrees of freedom of uhat: %d\n", sizes.ndofuhat);   
      printf("number of degrees of freedom of udg: %d\n", sizes.ndofudg);   
      printf("number of degrees of freedom of sdg: %d\n", sizes.ndofsdg);   
      printf("number of degrees of freedom of odg: %d\n", sizes.ndofodg);   
      printf("number of degrees of freedom of edg: %d\n", sizes.ndofedg);   
      printf("length of the stabilization: %d\n", components.ntau);   

      printf("maximum dimension of the reduced basis space: %d\n", solverparams.RBdim);
      // (runtime reduced-basis dimensions now live in CSolver::state, not commonstruct)

      printf("external uhat function flag: %d\n", couplingparams.extUhat);
      printf("external fhat function flag: %d\n", couplingparams.extFhat);
      printf("external stabilization function flag: %d\n", couplingparams.extStab);
      printf("curved mesh flag: %d\n", grid.curvedMesh);
      printf("debug mode flag: %d\n", outputparams.debugMode);
      printf("time-dependent problem flag: %d\n", timeparams.tdep);
      printf("wave problem flag: %d\n", timeparams.wave);
      printf("linear problem flag: %d\n", timeparams.linearProblem);
      printf("save solution frequency: %d\n", outputparams.saveSolFreq);
      printf("save solution option: %d\n", outputparams.saveSolOpt);
      printf("timestep offset to restart simulation: %d\n", outputparams.timestepOffset);
      printf("time-derivative function flag: %d\n", timeparams.tdfunc);
      printf("source function flag: %d\n", physicsparams.source);
      printf("model number: %d\n", modelnumber);
      printf("boundary index to save solution: %d\n", qoiparams.ibs);
      printf("save solution boundary frequency: %d\n", outputparams.saveSolBouFreq);
      printf("compute time-averaged solution flag: %d\n", outputparams.compudgavg);
      printf("read time-averaged solution flag: %d\n", outputparams.readudgavg);
    
      printf("number of components of artificial viscosity: %d\n", physicsparams.ncAV);
      printf("number of artificial viscosity smoothing iterations: %d\n", physicsparams.AVsmoothingIter);
      printf("frozen artificial viscosity flag: %d\n", physicsparams.frozenAVflag);
      printf("linear solver type: %d\n", solverparams.linearSolver);
      printf("nonlinear solver type: %d\n", solverparams.nonlinearSolver);
      printf("maximum linear solver iterations: %d\n", solverparams.linearSolverMaxIter);
      printf("maximum nonlinear solver iterations: %d\n", solverparams.nonlinearSolverMaxIter);
      // (current linear/nonlinear iteration counts now live in CSolver::state, not commonstruct)
      printf("matrix-vector multiplication order: %d\n", solverparams.matvecOrder);
      printf("GMRES restart parameter: %d\n", solverparams.gmresRestart);
      printf("GMRES orthogonalization method: %d\n", solverparams.gmresOrthogMethod);
      printf("preconditioner type: %d\n", solverparams.preconditioner);
      printf("preconditioner matrix type: %d\n", solverparams.precMatrixType);
      printf("PTC matrix type: %d\n", solverparams.ptcMatrixType);
      printf("run mode: %d\n", runmode);
      printf("time step factor: %f\n", timestate.dtfactor);
      printf("current simulation time: %f\n", timestate.time);
      printf("matrix-vector multiplication tolerance: %f\n", solverparams.matvecTol);
      printf("linear solver tolerance: %f\n", solverparams.linearSolverTol);
      printf("nonlinear solver tolerance: %f\n", solverparams.nonlinearSolverTol);
      // (linear solver tolerance factor / relative error now live in CSolver::state)
      printf("artificial viscosity ramp factor: %f\n", physicsparams.rampFactor);
      printf("PTC parameter: %f\n", solverparams.PTCparam);
      printf("initial stabilization parameter: %f\n", physicsparams.tau0);
      printf("DAE alpha parameter: %f\n", timeparams.dae_alpha);
      printf("DAE beta parameter: %f\n", timeparams.dae_beta);
      printf("DAE gamma parameter: %f\n", timeparams.dae_gamma);
      printf("DAE epsilon parameter: %f\n", timeparams.dae_epsilon);
      
      printf("number of boundary conditions: %d\n", meshsizes.maxnbc);
      printf("number of wall-model configurations: %d\n", wallmodelparams.nwm);
      printf("number of neighboring subdomains: %d\n", nnbsd);      
      printf("number of elements to send: %d\n", nelemsend);
      printf("number of elements to receive: %d\n", nelemrecv);
      
      printf("eblks array: %d by %d\n", 3, meshsizes.nbe);
      for (int j=0; j<3; j++) {
        for (int i=0; i<meshsizes.nbe; i++)
          printf("%d  ", eblks[j+3*i]);
        printf("\n");  
      }

      printf("fblks array: %d by %d\n", 3, meshsizes.nbf);
      for (int j=0; j<3; j++) {
        for (int i=0; i<meshsizes.nbf; i++)
          printf("%d  ", fblks[j+3*i]);
        printf("\n");  
      }

      if (spatialScheme==1) {
        printf("nboufaces array: %d by %d\n", meshsizes.maxnbc, meshsizes.nbe);
        for (int j=0; j<meshsizes.maxnbc; j++) {
          for (int i=0; i<meshsizes.nbe; i++)
            printf("%d  ", nboufaces[1+j+meshsizes.maxnbc*i]);
          printf("\n");  
        }        
      }
      
      if (nnbsd >= 1) {
        printf("nbsd array: %d by %d\n", 1, nnbsd);
        for (int i=0; i<nnbsd; i++)
          printf("%d  ", nbsd[i]);
        printf("\n");        

        printf("elemsendpts array: %d by %d\n", 1, nnbsd);
        for (int i=0; i<nnbsd; i++)
          printf("%d  ", elemsendpts[i]);
        printf("\n");        

        printf("elemrecvpts array: %d by %d\n", 1, nnbsd);
        for (int i=0; i<nnbsd; i++)
          printf("%d  ", elemrecvpts[i]);
        printf("\n");          

        printf("elemsend array: %d by %d\n", 1, nelemsend);
        for (int i=0; i<nelemsend; i++)
          printf("%d  ", elemsend[i]);
        printf("\n");        

        printf("elemrecv array: %d by %d\n", 1, nelemrecv);
        for (int i=0; i<nelemrecv; i++)
          printf("%d  ", elemrecv[i]);
        printf("\n");        
      }
    }
    
    void freememory()
    {
        CPUFREE(eblks); 
        CPUFREE(fblks);
        CPUFREE(nboufaces);
        CPUFREE(nextfaces);
        CPUFREE(nbsd); 
        CPUFREE(elemsend); 
        CPUFREE(elemrecv); 
        CPUFREE(elemsendpts); 
        CPUFREE(elemrecvpts); 
        if (stgparams.nstgib > 0) CPUFREE(stgparams.stgib); 
        CPUFREE(vindx); 
        CPUFREE(interfacefluxmap); 
        CPUFREE(wallmodelparams.wmModelIDs);
        CPUFREE(wallmodelparams.wmBoundaries);
        CPUFREE(cartgridpart); 
        CPUFREE(boundaryConditions); 
        CPUFREE(intepartpts);         
        CPUFREE(dt); 
        CPUFREE(dae_dt); 
        CPUFREE(wallmodelparams.wmDistances);
        CPUFREE(DIRKcoeff_c); 
        CPUFREE(DIRKcoeff_d); 
        CPUFREE(DIRKcoeff_t); 
        CPUFREE(BDFcoeff_c); 
        CPUFREE(BDFcoeff_t);  
        if (qoiparams.nvqoi > 0) CPUFREE(qoiparams.qoivolume);
        if (qoiparams.nsurf > 0) CPUFREE(qoiparams.qoisurface);
    }
};
using commonstruct = commonstructT<::dstype, ::Int>;

// --- QoI output helpers (instance-driven) ------------------------------------------------
// Write the QoI column headers / one QoI value row by iterating the registered QoI instances.
// With the default single-domain + single-boundary instances this reproduces the historical
// "Domain_QoI<i>" / "Boundary_QoI<i>" columns and values byte-for-byte.
inline void writeQoIHeader(std::ostream& outqoi, const qoiparamsstruct& qoiparams)
{
    for (const auto& q : qoiparams.qoiinstances)
        for (Int j = 0; j < q.ncomp; ++j)
            outqoi << std::setw(16) << std::left << (q.name + std::to_string(j + 1));
}
inline void writeQoIRow(std::ostream& outqoi, const qoiparamsstruct& qoiparams)
{
    for (const auto& q : qoiparams.qoiinstances) {
        const dstype* buf = (q.kind == 0) ? qoiparams.qoivolume : qoiparams.qoisurface;
        for (Int j = 0; j < q.ncomp; ++j)
            outqoi << std::setw(16) << std::scientific << std::setprecision(6) << buf[q.offset + j];
    }
}
// Write the "Time" + QoI-instance header line exactly once, on the first row (while the
// stream is still empty). Writing the header lazily (rather than at file-open) lets QoI
// instances registered after model init (ExasimSolver::AddQoI) be reflected in the columns.
inline void writeQoIHeaderOnce(std::ostream& outqoi, const qoiparamsstruct& qoiparams)
{
    if (outqoi.tellp() == std::streampos(0)) {
        outqoi << std::setw(16) << std::left << "Time";
        writeQoIHeader(outqoi, qoiparams);
        outqoi << "\n";
    }
}

#endif
