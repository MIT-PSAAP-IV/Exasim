cmake_minimum_required(VERSION 3.16)
project(exasim_consumer_external_data CXX)
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

# Register external model 200 (linear elasticity with custom QoI) via text2code.
# The runtime pdeapp.txt doubles as the text2code config — the macro rewrites
# exasimpath internally and derives the pdemodel name from the file name.
exasim_add_external_builtin_model(TARGET ext_model_fext
  ID 200
  PDEMODEL ${CMAKE_CURRENT_SOURCE_DIR}/pdeapp.txt)

add_executable(consumer_external_data main.cpp)
target_compile_definitions(consumer_external_data PRIVATE _BUILTINLIBRARY)
if(EXASIM_MPI)
  target_compile_definitions(consumer_external_data PRIVATE HAVE_MPI)
endif()
# Link ext_model_fext (NOT Exasim::builtinmodel) — the external target provides
# getBuiltInLibraryExasimDriverABI() and links Exasim::builtinmodel transitively.
target_link_libraries(consumer_external_data PRIVATE
    Exasim::headers Exasim::pre ext_model_fext Kokkos::kokkos MPI::MPI_CXX)
