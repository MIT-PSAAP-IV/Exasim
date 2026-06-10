cmake_minimum_required(VERSION 3.16)
project(exasim_consumer_builtin_gpu CXX)
set(CMAKE_CXX_STANDARD 17)

# Pure out-of-tree GPU consumer: find_package only, no reach into the Exasim source tree.
# Build with nvcc_wrapper as CXX: cmake -DCMAKE_CXX_COMPILER=/path/to/nvcc_wrapper ...
find_package(Exasim REQUIRED COMPONENTS gpu)
find_package(Kokkos REQUIRED)

add_executable(consumer_builtin_gpu main.cpp)
target_compile_definitions(consumer_builtin_gpu PRIVATE _BUILTINLIBRARY)
target_link_libraries(consumer_builtin_gpu PRIVATE
    Exasim::headers Exasim::pre Exasim::builtinmodel Kokkos::kokkos)
