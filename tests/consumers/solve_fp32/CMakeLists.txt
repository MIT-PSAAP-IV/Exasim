# solve_fp32 — full float32 Poisson SOLVE consumer (double + float in one binary), no rebuild.
cmake_minimum_required(VERSION 3.16)
project(exasim_consumer_solve_fp32 CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Exasim REQUIRED COMPONENTS cpu)
find_package(Kokkos REQUIRED)
find_package(BLAS   REQUIRED)
find_package(LAPACK REQUIRED)

add_executable(consumer_solve_fp32 main.cpp)
target_include_directories(consumer_solve_fp32 PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../models")
target_link_libraries(consumer_solve_fp32 PRIVATE
    Exasim::headers Exasim::pre Kokkos::kokkos
    ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
