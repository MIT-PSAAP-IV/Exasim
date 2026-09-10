# uniformrefine — uniformrefinementlevel mesh refinement + field prolongation consumer.
cmake_minimum_required(VERSION 3.16)
project(exasim_consumer_uniformrefine CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Exasim REQUIRED COMPONENTS cpu)
find_package(Kokkos REQUIRED)
find_package(BLAS   REQUIRED)
find_package(LAPACK REQUIRED)

add_executable(consumer_uniformrefine main.cpp)
target_link_libraries(consumer_uniformrefine PRIVATE
    Exasim::headers Exasim::pre Kokkos::kokkos
    ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
