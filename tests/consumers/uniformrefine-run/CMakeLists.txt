cmake_minimum_required(VERSION 3.16)
project(exasim_consumer_uniformrefine_run CXX)
set(CMAKE_CXX_STANDARD 20)

# Solves the builtin Poisson problem on a coarse grid refined by
# `uniformrefinementlevel = 1` (serial preprocessing at NP=1, per-rank
# refinement in ParallelPreprocessing at NP>1); the runner gates the QoI.
option(EXASIM_MPI "MPI variant" ON)
if(EXASIM_MPI)
  set(EXASIM_VARIANT cpumpi)
else()
  set(EXASIM_VARIANT cpu)
endif()
find_package(Exasim REQUIRED COMPONENTS ${EXASIM_VARIANT})
find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED)

add_executable(consumer_uniformrefine_run main.cpp)
target_compile_definitions(consumer_uniformrefine_run PRIVATE _BUILTINLIBRARY)
if(EXASIM_MPI)
  target_compile_definitions(consumer_uniformrefine_run PRIVATE _MPI)
endif()
target_link_libraries(consumer_uniformrefine_run PRIVATE
    Exasim::headers Exasim::pre Exasim::builtinmodel Kokkos::kokkos MPI::MPI_CXX)
