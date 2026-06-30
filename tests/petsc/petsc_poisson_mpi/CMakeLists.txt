# PETSc drives Exasim's exported HDG operators IN PARALLEL (MPI), steady Poisson.
#
# Needs the CPU+MPI Exasim install (build-slurm.sh: -DEXASIM_MPI=ON -DWITH_PARMETIS=ON)
# and an MPI-enabled PETSc. The Exasim::cpumpi component carries the _MPI + HAVE_PARMETIS
# compile definitions (operators.hpp bridges _MPI -> HAVE_MPI), the ParMETIS/METIS/GKlib
# libs, MPI, and BLAS -- so the only extra deps here are Kokkos + the MPI PETSc.
#
# Build (on slurm-b, after build-slurm.sh + an MPI PETSc):
#   cmake -S tests/petsc/petsc_poisson_mpi -B <build> \
#         -DCMAKE_PREFIX_PATH="<cpu-mpi-install>;<kokkos>" \
#         -DPETSC_PREFIX=/data/scratch/teoc/petsc/arch-mpi
#   cmake --build <build>
#   mpiexec -n 4 <build>/petsc_poisson_mpi

cmake_minimum_required(VERSION 3.16)
project(exasim_petsc_poisson_mpi CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Exasim REQUIRED COMPONENTS cpumpi)
find_package(Kokkos REQUIRED)

# MPI PETSc via pkg-config (ships petsc.pc; arch-mpi). petsc.pc references PETSC_DIR.
set(PETSC_PREFIX "/data/scratch/teoc/petsc/arch-mpi" CACHE PATH "MPI PETSc install prefix")
find_package(PkgConfig REQUIRED)
set(ENV{PKG_CONFIG_PATH} "${PETSC_PREFIX}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET petsc)

add_executable(petsc_poisson_mpi main.cpp)
target_include_directories(petsc_poisson_mpi PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../../models")
target_link_libraries(petsc_poisson_mpi PRIVATE
    Exasim::headers Exasim::pre Kokkos::kokkos PkgConfig::PETSC)
