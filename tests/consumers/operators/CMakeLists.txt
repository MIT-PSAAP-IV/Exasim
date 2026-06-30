# Out-of-tree operator-export consumer (P1a).
#
# Builds the in-memory discretization + operators for a concrete hand-written
# Model (Poisson2D) via find_package(Exasim) only — the recipe a PETSc app will
# reuse. Unlike tests/consumers/facade (install-layout check only, no backend
# compile), this consumer COMPILES the full templated FEM aggregation
# (<exasim/operators.hpp>) and links the preprocessing library for the in-memory
# Preprocessed assembly (CPreprocessing::take / meshFromArrays).
#
# Reviewer-runnable:
#   cmake -S tests/consumers/operators -B /tmp/operators_build \
#         -DCMAKE_PREFIX_PATH="<install>;<kokkos-buildserial>"
#   cmake --build /tmp/operators_build
#   ./ /tmp/operators_build/consumer_operators
#
# Serial CPU only (the in-memory CDiscretization ctor errors on a GPU backend);
# no _MPI / _BUILTINLIBRARY / _TEXT2CODE, so HAVE_SHARED_MODEL_LIB is off and the
# pure templated exasim::Name<M> path is exercised (no per-model codegen kernels).

cmake_minimum_required(VERSION 3.16)
project(exasim_consumer_operators CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Exasim REQUIRED COMPONENTS cpu)
find_package(Kokkos REQUIRED)
find_package(BLAS   REQUIRED)
find_package(LAPACK REQUIRED)

add_executable(consumer_operators main.cpp)
target_include_directories(consumer_operators PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}")            # for #include "my_model.hpp"
target_link_libraries(consumer_operators PRIVATE
    Exasim::headers Exasim::pre Kokkos::kokkos
    ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
