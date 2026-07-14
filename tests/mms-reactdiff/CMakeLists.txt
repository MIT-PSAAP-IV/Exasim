# MMS reaction-diffusion consumer (analytical-solution smoke for the header-only
# emit-app / exasim::petsc::solve_steady path).
#
# Builds a solve_steady verify harness for a small 2-species coupled reaction-diffusion
# model whose exact (manufactured) solution is u0 = sin(pi x) sin(pi y),
# u1 = 0.5 sin(pi x) sin(pi y). The harness solves the HDG system through the exported
# PETSc operator and checks the field against the exact solution pointwise.
#
# PETSc is required (this is the emit-app driver path); see run.sh, which SKIPs (77) when
# text2code or PETSc is unavailable.
cmake_minimum_required(VERSION 3.16)
project(consumer_mms_reactdiff CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(EXASIM_MPI "MPI variant" ON)
if(EXASIM_MPI)
  set(EXASIM_VARIANT cpumpi)
else()
  set(EXASIM_VARIANT cpu)
endif()

find_package(Exasim REQUIRED COMPONENTS ${EXASIM_VARIANT})
find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_executable(consumer_mms main.cpp)
target_compile_definitions(consumer_mms PRIVATE EXASIM_HAVE_PETSC HAVE_BACKEND_PREPROCESSING)
if(EXASIM_MPI)
  target_compile_definitions(consumer_mms PRIVATE _MPI)
endif()
target_link_libraries(consumer_mms PRIVATE
    Exasim::headers Kokkos::kokkos MPI::MPI_CXX PkgConfig::PETSC
    LAPACK::LAPACK BLAS::BLAS)
target_include_directories(consumer_mms PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
