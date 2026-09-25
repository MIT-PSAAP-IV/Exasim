cmake_minimum_required(VERSION 3.16)
project(exasim_consumer_surfacequantities_run CXX)
set(CMAKE_CXX_STANDARD 20)

# Solves the builtin Poisson problem (model 1, which defines SurfaceQuantities) with
# saveSolBouFreq = 1 on ibs = [1, 2]; check.sh then verifies outbousurf against the
# other boundary files, the Gauss-point mode against Boundary_QoI, and the ibs-list
# union against a single-boundary run.
option(EXASIM_MPI "MPI variant" ON)
if(EXASIM_MPI)
  set(EXASIM_VARIANT cpumpi)
else()
  set(EXASIM_VARIANT cpu)
endif()
find_package(Exasim REQUIRED COMPONENTS ${EXASIM_VARIANT})
find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED)

add_executable(consumer_surfacequantities_run main.cpp)
target_compile_definitions(consumer_surfacequantities_run PRIVATE _BUILTINLIBRARY)
if(EXASIM_MPI)
  target_compile_definitions(consumer_surfacequantities_run PRIVATE _MPI)
endif()
target_link_libraries(consumer_surfacequantities_run PRIVATE
    Exasim::headers Exasim::pre Exasim::builtinmodel Kokkos::kokkos MPI::MPI_CXX)
