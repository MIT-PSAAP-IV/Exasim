# meshload — read_mesh + as_mesh_spec bridge consumer (file mesh == array mesh).
cmake_minimum_required(VERSION 3.16)
project(exasim_consumer_meshload CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Exasim REQUIRED COMPONENTS cpu)
find_package(Kokkos REQUIRED)
find_package(BLAS   REQUIRED)
find_package(LAPACK REQUIRED)

add_executable(consumer_meshload main.cpp)
target_include_directories(consumer_meshload PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../models")
target_link_libraries(consumer_meshload PRIVATE
    Exasim::headers Exasim::pre Kokkos::kokkos
    ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
