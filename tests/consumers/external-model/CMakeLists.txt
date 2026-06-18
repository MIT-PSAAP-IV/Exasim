cmake_minimum_required(VERSION 3.16)
project(exasim_consumer_external_model CXX)
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

# Register TWO external models (ids 100 and 101, same Poisson physics) as ONE
# combined provider via the macro's multi-id form. This guards the multi-id
# codegen + single-provider build/link path on CI (the run below exercises id
# 100; combined multi-id runtime dispatch is covered end-to-end by the CHEFSI
# coupled_heat2d cases). Exasim_TEXT2CODE/Exasim_BUILTIN_DIR are set by
# find_package(Exasim).
exasim_add_external_builtin_model(TARGET ext_model_100
  IDS 100 101
  PDEMODEL ${CMAKE_CURRENT_SOURCE_DIR}/pdeapp100.txt
           ${CMAKE_CURRENT_SOURCE_DIR}/pdeapp101.txt)

add_executable(consumer_external_model main.cpp)
target_compile_definitions(consumer_external_model PRIVATE _BUILTINLIBRARY)
if(EXASIM_MPI)
  target_compile_definitions(consumer_external_model PRIVATE _MPI)
endif()
# Link ext_model_100 (NOT Exasim::builtinmodel) — the external target provides
# getBuiltInLibraryExasimDriverABI() and links Exasim::builtinmodel transitively.
target_link_libraries(consumer_external_model PRIVATE
    Exasim::headers Exasim::pre ext_model_100 Kokkos::kokkos MPI::MPI_CXX)
