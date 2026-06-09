cmake_minimum_required(VERSION 3.16)
project(exasim_consumer_builtin_gpumpi CXX)
set(CMAKE_CXX_STANDARD 17)

# Pure out-of-tree GPU+MPI consumer: find_package only, no reach into the Exasim source tree.
# Build with nvcc_wrapper as CXX: cmake -DCMAKE_CXX_COMPILER=/path/to/nvcc_wrapper ...
find_package(Exasim REQUIRED COMPONENTS gpumpi)
find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED)

add_executable(consumer_builtin_gpumpi main.cpp)
target_compile_definitions(consumer_builtin_gpumpi PRIVATE _BUILTINLIBRARY _MPI)
target_link_libraries(consumer_builtin_gpumpi PRIVATE
    Exasim::headers Exasim::pre Exasim::builtinmodel Kokkos::kokkos MPI::MPI_CXX)
