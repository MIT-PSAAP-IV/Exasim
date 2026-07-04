# PETSc-driven isoq2d CHT solid transient: PETSc TS owns the time loop, driving
# Exasim's exported HDG operators (MatShell res.H + PCShell res.K + assemble_matrix).
# Same consumer pattern as tests/petsc/petsc_heat; auto-picked-up by
# tests/petsc/run-petsc-test.sh.
#
# Build (mirrors the harness):
#   cmake -S tests/petsc/petsc_isoq2d_cht -B /tmp/petsc_isoq2d_cht_build \
#         -DCMAKE_PREFIX_PATH="<install>;<kokkos-buildserial>" \
#         -DPETSC_PREFIX="$CONDA_PREFIX"
#   cmake --build /tmp/petsc_isoq2d_cht_build
#   EXASIM_DATA_DIR=<install>/share/exasim /tmp/petsc_isoq2d_cht_build/petsc_isoq2d_cht

cmake_minimum_required(VERSION 3.16)
project(exasim_petsc_isoq2d_cht CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Exasim REQUIRED COMPONENTS cpu petsc)
find_package(Kokkos REQUIRED)
find_package(BLAS   REQUIRED)
find_package(LAPACK REQUIRED)

# PETSc via pkg-config (ships PETSc.pc, no CMake config). Default to the conda env
# (the summit-dev toolchain Exasim was built with); override -DPETSC_PREFIX=... else.
if(DEFINED ENV{CONDA_PREFIX})
  set(_petsc_default "$ENV{CONDA_PREFIX}")
else()
  set(_petsc_default "/opt/homebrew/opt/petsc")
endif()
set(PETSC_PREFIX "${_petsc_default}" CACHE PATH "PETSc install prefix")
find_package(PkgConfig REQUIRED)
set(ENV{PKG_CONFIG_PATH} "${PETSC_PREFIX}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET petsc)

add_executable(petsc_isoq2d_cht main.cpp)
target_include_directories(petsc_isoq2d_cht PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../../models")

# Link the SAME open-mpi PETSc was built against. conda/summit: mpi.h shares
# PETSC_PREFIX -> point at the prefix. System PETSc keeps mpi.h in a separate MPI
# package outside PETSC_PREFIX -> locate it via find_package(MPI).
if(EXISTS "${PETSC_PREFIX}/include/mpi.h")
  set(OPENMPI_PREFIX "${PETSC_PREFIX}" CACHE PATH "open-mpi prefix (the one PETSc uses)")
  target_include_directories(petsc_isoq2d_cht PRIVATE "${OPENMPI_PREFIX}/include")
  target_link_directories(petsc_isoq2d_cht PRIVATE "${OPENMPI_PREFIX}/lib")
  target_link_libraries(petsc_isoq2d_cht PRIVATE
      Exasim::headers Exasim::petsc Kokkos::kokkos
      ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} PkgConfig::PETSC mpi)
else()
  find_package(MPI REQUIRED COMPONENTS CXX)
  target_link_libraries(petsc_isoq2d_cht PRIVATE
      Exasim::headers Exasim::petsc Kokkos::kokkos
      ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} PkgConfig::PETSC MPI::MPI_CXX)
endif()
