cmake_minimum_required(VERSION 3.16)
project(exasim_consumer_builtin CXX)
set(CMAKE_CXX_STANDARD 20)

# Pure out-of-tree consumer: find_package only, no reach into the Exasim source tree.
option(EXASIM_MPI "MPI variant" ON)
if(EXASIM_MPI)
  set(EXASIM_VARIANT cpumpi)
else()
  set(EXASIM_VARIANT cpu)
endif()
find_package(Exasim REQUIRED COMPONENTS ${EXASIM_VARIANT})
find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED)

add_executable(consumer_builtin main.cpp)
target_compile_definitions(consumer_builtin PRIVATE _BUILTINLIBRARY)
if(EXASIM_MPI)
  target_compile_definitions(consumer_builtin PRIVATE _MPI)
endif()
target_link_libraries(consumer_builtin PRIVATE
    Exasim::headers Exasim::pre Exasim::builtinmodel Kokkos::kokkos MPI::MPI_CXX)
