# P2 — PETSc TS owns the time loop for the heat equation, driving Exasim's exported HDG operators
#      (MatShell res.H + PCShell res.K + assemble_matrix + Vanka), same as steady petsc_poisson.
#
# Build (CI wires this via tests/petsc/run-petsc-test.sh):
#   cmake -S tests/petsc/petsc_heat -B /tmp/petsc_heat_build \
#         -DCMAKE_PREFIX_PATH="<install>;<kokkos-buildserial>" \
#         -DPETSC_PREFIX="$CONDA_PREFIX"
#   cmake --build /tmp/petsc_heat_build
#   EXASIM_DATA_DIR=<install>/share/exasim /tmp/petsc_heat_build/petsc_heat
#
# NOTE on PETSc + MPI: use ONE consistent stack. Exasim here is built with the conda
# (summit-dev) toolchain, whose env also ships PETSc + open-mpi -- so we default
# PETSC_PREFIX to $CONDA_PREFIX. (A brew PETSc pulls a *different* open-mpi than
# conda's libmpi -> mismatched mpi.h / ABI clash with the conda-built Exasim.) The
# serial consumer needs only header-inline preprocessing (CPreprocessing::take,
# meshFromArrays) + the templated FEM aggregation, so Exasim::headers suffices.

cmake_minimum_required(VERSION 3.16)
project(exasim_petsc_heat CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Exasim REQUIRED COMPONENTS cpu petsc)
find_package(Kokkos REQUIRED)
find_package(BLAS   REQUIRED)
find_package(LAPACK REQUIRED)

# PETSc via pkg-config (ships PETSc.pc, no CMake config). Default to the conda env
# (the toolchain Exasim was built with); override -DPETSC_PREFIX=... for another stack.
if(DEFINED ENV{CONDA_PREFIX})
  set(_petsc_default "$ENV{CONDA_PREFIX}")
else()
  set(_petsc_default "/opt/homebrew/opt/petsc")
endif()
set(PETSC_PREFIX "${_petsc_default}" CACHE PATH "PETSc install prefix")
find_package(PkgConfig REQUIRED)
set(ENV{PKG_CONFIG_PATH} "${PETSC_PREFIX}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET petsc)

# PETSC_COMM_SELF expands to &ompi_mpi_comm_self, so the binary references open-mpi
# symbols directly -> link the SAME open-mpi PETSc was built against (conda's libmpi).
# We do NOT use find_package(MPI) (could resolve a different MPI); point at the prefix.
set(OPENMPI_PREFIX "${PETSC_PREFIX}" CACHE PATH "open-mpi prefix (the one PETSc uses)")

add_executable(petsc_heat main.cpp)
target_include_directories(petsc_heat PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../../models" "${OPENMPI_PREFIX}/include")
target_link_directories(petsc_heat PRIVATE "${OPENMPI_PREFIX}/lib")
target_link_libraries(petsc_heat PRIVATE
    Exasim::headers Exasim::petsc Kokkos::kokkos
    ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
    PkgConfig::PETSC mpi)
