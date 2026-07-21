cmake_minimum_required(VERSION 3.16)
project(poisson2d C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Vendored find-modules (FindPETSc etc.), if shared alongside the app.
if(NOT CHEFSI_FIND_MODULES)
  foreach(_cand "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
                "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules")
    if(IS_DIRECTORY "${_cand}")
      get_filename_component(CHEFSI_FIND_MODULES "${_cand}" ABSOLUTE)
      break()
    endif()
  endforeach()
endif()
if(CHEFSI_FIND_MODULES)
  list(PREPEND CMAKE_MODULE_PATH "${CHEFSI_FIND_MODULES}")
endif()

option(EXASIM_MPI "Use the MPI-enabled Exasim variant" ON)
option(EXASIM_GPU "Use the GPU-enabled Exasim variant" OFF)
if(EXASIM_GPU AND EXASIM_MPI)
  set(EXASIM_VARIANT gpumpi)
elseif(EXASIM_GPU)
  set(EXASIM_VARIANT gpu)
elseif(EXASIM_MPI)
  set(EXASIM_VARIANT cpumpi)
else()
  set(EXASIM_VARIANT cpu)
endif()

find_package(Exasim REQUIRED COMPONENTS ${EXASIM_VARIANT})
find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED)

# PETSc — the app drives the solve via <exasim/petsc.hpp>, so link PETSc directly.
find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)
message(STATUS "PETSc: ${PETSC_VERSION}")

# The unity-compiled backend needs BLAS/LAPACK (normally transitive via the prelib).
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Backend is unity-compiled INTO this TU (no Exasim::pre prelib); concrete No-ABI model.
set(_UNITY_DEFS EXASIM_HAVE_PETSC HAVE_BACKEND_PREPROCESSING)
if(EXASIM_MPI)
  list(APPEND _UNITY_DEFS _MPI)
endif()

add_executable(${PROJECT_NAME} poisson2d.cc)
target_compile_definitions(${PROJECT_NAME} PRIVATE ${_UNITY_DEFS})
target_link_libraries(${PROJECT_NAME} PRIVATE
    Exasim::headers          # include dirs + parmetis/metis deps (NO prelib)
    Kokkos::kokkos
    MPI::MPI_CXX
    PkgConfig::PETSC
    LAPACK::LAPACK
    BLAS::BLAS)
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
