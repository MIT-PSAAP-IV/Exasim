# model_fp32 — consumer-side float32 test of the templated model-kernel path.
# Builds against an installed (double-default) Exasim; instantiates the Poisson2D
# kernels at float AND double. No Exasim rebuild, no PETSc.
#
#   cmake -S tests/consumers/model_fp32 -B /tmp/model_fp32 \
#         -DCMAKE_PREFIX_PATH="<install>;<kokkos-buildserial>"
#   cmake --build /tmp/model_fp32
#   /tmp/model_fp32/consumer_model_fp32
#
# GPU: build against the CUDA Exasim install with -DCMAKE_CXX_COMPILER=<nvcc_wrapper>
# and the CUDA Kokkos tree; the same source runs on the device.

cmake_minimum_required(VERSION 3.16)
project(exasim_model_fp32 CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Exasim REQUIRED COMPONENTS cpu)
find_package(Kokkos REQUIRED)
find_package(BLAS   REQUIRED)
find_package(LAPACK REQUIRED)

add_executable(consumer_model_fp32 main.cpp)
target_include_directories(consumer_model_fp32 PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../models")   # shared poisson2d.hpp
target_link_libraries(consumer_model_fp32 PRIVATE
    Exasim::headers Exasim::pre Kokkos::kokkos
    ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
