# P1b — PETSc drives Exasim's exported HDG operators (steady Poisson).
#
# Build (CI wires this via tests/petsc/run-petsc-test.sh):
#   cmake -S tests/petsc/robustness -B /tmp/robustness_build \
#         -DCMAKE_PREFIX_PATH="<install>;<kokkos-buildserial>" \
#         -DPETSC_PREFIX="$CONDA_PREFIX"
#   cmake --build /tmp/robustness_build
#   EXASIM_DATA_DIR=<install>/share/exasim /tmp/robustness_build/robustness
#
# NOTE on PETSc + MPI: use ONE consistent stack. Exasim here is built with the conda
# (summit-dev) toolchain, whose env also ships PETSc + open-mpi -- so we default
# PETSC_PREFIX to $CONDA_PREFIX. (A brew PETSc pulls a *different* open-mpi than
# conda's libmpi -> mismatched mpi.h / ABI clash with the conda-built Exasim.) The
# serial consumer needs only header-inline preprocessing (CPreprocessing::take,
# meshFromArrays) + the templated FEM aggregation, so Exasim::headers suffices.

cmake_minimum_required(VERSION 3.16)
project(exasim_robustness CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# EXASIM_GPU=ON builds against the CUDA Exasim (Exasim::gpu) + a CUDA PETSc, and the
# binary must be compiled by Kokkos' nvcc_wrapper (pass -DCMAKE_CXX_COMPILER=.../nvcc_wrapper).
# The same main.cpp runs CPU (EXASIM_BACKEND=0) or CUDA (EXASIM_BACKEND=2) at runtime.
option(EXASIM_GPU "Build against the CUDA Exasim + CUDA PETSc" OFF)
if(EXASIM_GPU)
  set(CMAKE_CXX_STANDARD 17)   # Kokkos CUDA tree here is C++17
  find_package(Exasim REQUIRED COMPONENTS gpu petsc)
else()
  find_package(Exasim REQUIRED COMPONENTS cpu petsc)
endif()
find_package(Kokkos REQUIRED)

# PETSc via pkg-config (ships PETSc.pc, no CMake config). CPU default: the conda env
# (the toolchain Exasim was built with). GPU: pass -DPETSC_PREFIX=<cuda petsc prefix>.
if(DEFINED ENV{CONDA_PREFIX})
  set(_petsc_default "$ENV{CONDA_PREFIX}")
else()
  set(_petsc_default "/opt/homebrew/opt/petsc")
endif()
set(PETSC_PREFIX "${_petsc_default}" CACHE PATH "PETSc install prefix")
find_package(PkgConfig REQUIRED)
set(ENV{PKG_CONFIG_PATH} "${PETSC_PREFIX}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
# petsc.pc references the PETSC_DIR variable; define it so pkg-config resolves cleanly.
set(ENV{PKG_CONFIG} "pkg-config --define-variable=PETSC_DIR=${PETSC_PREFIX}")
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET petsc)

add_executable(robustness main.cpp)
target_include_directories(robustness PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../../models")
target_link_libraries(robustness PRIVATE
    Exasim::headers Exasim::pre Exasim::petsc Kokkos::kokkos PkgConfig::PETSC)

if(NOT EXASIM_GPU)
  # CPU: a CONDA PETSc built with open-mpi -> PETSC_COMM_SELF references ompi_mpi_comm_self,
  # so link the same open-mpi explicitly. (A serial CUDA PETSc has no MPI, so the GPU path
  # skips this.) BLAS/LAPACK for the FEM pblas calls.
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  # conda/summit: mpi.h shares PETSC_PREFIX; system PETSc (Debian petsc-real): find it.
  if(EXISTS "${PETSC_PREFIX}/include/mpi.h")
    set(OPENMPI_PREFIX "${PETSC_PREFIX}" CACHE PATH "open-mpi prefix (the one PETSc uses)")
    target_include_directories(robustness PRIVATE "${OPENMPI_PREFIX}/include")
    target_link_directories(robustness PRIVATE "${OPENMPI_PREFIX}/lib")
    target_link_libraries(robustness PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} mpi)
  else()
    find_package(MPI REQUIRED COMPONENTS CXX)
    target_link_libraries(robustness PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} MPI::MPI_CXX)
  endif()
endif()
