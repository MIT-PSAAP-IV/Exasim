// SPDX-License-Identifier: see LICENSE
//
// <exasim/operators.hpp> — the templated FEM backend, aggregated for an
// out-of-tree consumer that wants Exasim's DISCRETIZATION + OPERATORS
// (residual R(u), Jacobian-apply J·v, preconditioner) WITHOUT Exasim's
// own time loop / nonlinear solver. This is the surface a PETSc (or any
// external) driver builds against: it constructs `CDiscretization` +
// `CResidual<M>` + `CAssembler<M>` + `CPreconditioner<M>` in memory and
// calls back into them matrix-free.
//
// Relationship to <exasim/run.hpp>: run.hpp is the *full* legacy entry
// point — it includes THIS header for the backend aggregation and then
// adds the MPI/Kokkos bootstrap + `exasim::run<M>()` (the time/Newton
// loop). A consumer that wants to own the solve drives it itself includes
// only operators.hpp; a consumer that wants Exasim's built-in loop
// includes run.hpp. Both share one copy of the aggregation, so there is
// no ODR hazard and the include set cannot drift between them.
//
// What this header adds beyond the aggregation run.hpp also pulls:
//   * <backend/Preprocessing/preprocessing.hpp> — CPreprocessing /
//     meshFromArrays / take(), so a consumer can build an in-memory
//     exasim::Preprocessed without writing datain binaries.
//   * <backend/Discretization/discretization_inmemory.hpp> — the
//     CDiscretization(exasim::Preprocessed&&, ...) in-memory constructor.
//
// CPU only for now (the in-memory ctor errors on a GPU backend); the
// initial PETSc operator-export path is serial CPU.

#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <filesystem>

#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstring>

#ifdef _OPENMP
#define HAVE_OPENMP
#else
#define HAVE_ONETHREAD
#endif

#ifdef _CUDA
#define HAVE_GPU
#define HAVE_CUDA
#endif

#ifdef _HIP
#define HAVE_GPU
#define HAVE_HIP
#endif

#ifdef _TEXT2CODE
#define HAVE_TEXT2CODE
#endif

#ifdef _KOKKOSKERNEL
#define HAVE_KOKKOSKERNEL
#endif

#if defined(HAVE_TEXT2CODE) || defined(HAVE_BUILTINMODEL)
#define HAVE_SHARED_MODEL_LIB
#endif

#ifdef _ENZYME
#define HAVE_ENZYME
#endif

#ifdef _MUTATIONPP
#define HAVE_MPP
#endif

#ifdef _MPI
#define HAVE_MPI
#endif

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include "cublas_v2.h"
#endif

#ifdef HAVE_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>
#endif

#ifdef HAVE_MPP
#include <mutation++.h>
#endif

#ifdef HAVE_SHARED_MODEL_LIB
#include <array>
#include <regex>
#include <unordered_map>
#include <unordered_set>
#endif

#ifdef HAVE_MPI
#include <mpi.h>
// Exasim communicators. `inline` (C++17) so an out-of-tree consumer TU (which aggregates
// the backend headers via this file) DEFINES them itself instead of leaving an undefined
// reference that drags the whole unity ExasimSolver.cpp.o out of the preprocessing library
// -- which then multiply-defines every other header function the consumer also compiled.
// The library still has its own (strong) definitions in postprocess.cpp; a parallel
// consumer sets these to its MPI communicator (the halo exchange in hdgMatVec uses
// EXASIM_COMM_LOCAL). A serial build never touches them (HAVE_MPI undefined).
inline MPI_Comm EXASIM_COMM_WORLD = MPI_COMM_NULL;
inline MPI_Comm EXASIM_COMM_LOCAL = MPI_COMM_NULL;
#endif

#include <Kokkos_Core.hpp>

#include <backend/Common/common.h>
#include <backend/Common/cpuimpl.h>
#include <backend/Common/kokkosimpl.h>
#include <backend/Common/pblas.h>

// The aggregated FEM implementation .cpp use unqualified std names (string, cout,
// endl, ...) -- they were authored for the unity build (ExasimSolver.cpp), which
// opens `using namespace std;` before including them. Mirror that here so the same
// .cpp compile unchanged in the templated consumer TU.
using namespace std;

// Aggregate the FEM implementation the same way the in-tree unity build
// (backend/Main/ExasimSolver.cpp) does: include the templated class .cpp
// directly, in dependency order, rather than the per-class .hpp twins (those
// were deleted as duplicates of the .cpp). model_drivers_abi.cpp supplies the
// common-based global ::Name drivers used by the AbiAdapter dispatch branch;
// it is codegen-free (no per-model KokkosFlux.cpp), so a concrete Model M
// compiles with NO text2code kernels -- the templated exasim::Name<M> path.
#include <backend/Model/ModelDispatch/model_drivers_abi.cpp>
#include <backend/Discretization/discretization.cpp>
#include <backend/Discretization/assembler.cpp>
#include <backend/Discretization/residualeval.cpp>
#include <backend/Preconditioning/preconditioner.cpp>
#include <backend/Solver/solver.cpp>
#include <backend/Visualization/visualization.cpp>
#include <backend/PointLocator/pointlocator.cpp>
#include <backend/Solution/solution.cpp>
#include <backend/Solution/solutionwriter.cpp>
#include <backend/Solution/nonlinearsolver.cpp>

// Preprocessing entry points (CPreprocessing, meshFromArrays, take, PDE/
// InputParams/ParsedSpec, parseInputFile/initializePDE). run.hpp only pulls
// these under HAVE_SHARED_MODEL_LIB; the operator-export consumer always needs
// them to assemble an in-memory exasim::Preprocessed. The symbols are defined
// in the preprocessing library (Exasim::pre), so the consumer links it.
#include <backend/Preprocessing/preprocessing.hpp>

// The in-memory CDiscretization ctor: must come AFTER the FEM aggregation (so
// CDiscretization / cpuInitSetup / buildConn are defined) and after
// buildstructs.hpp / preprocessing.hpp (so exasim::Preprocessed is complete).
#include <backend/Discretization/discretization_inmemory.hpp>

#include "detail/abi_adapter.hpp"

// Interface coupling helper: buffer-owning geometry + flux exchange for one coupled
// boundary. Included last because it needs CSolution<M> (solution.cpp, above) and is
// instantiated at AbiAdapter by ExasimSolver as well as at a concrete model by
// no-ABI consumers.
#include "interface_coupling.hpp"
#include "model_arrays.hpp"
