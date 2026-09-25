cmake_minimum_required(VERSION 3.16)
project(exasim_consumer_surface_vis CXX)
set(CMAKE_CXX_STANDARD 20)

# Surface-vis consumer: a text2code external model (id 100, Poisson 2D) that
# declares a VisSurfScalars output, run with saveParaview + ibvis so the solve
# emits an outsurf surface-visualization alongside the volume outvis files.
# Built ONLY via find_package(Exasim) -- no source tree references.
option(EXASIM_MPI "MPI variant" ON)
if(EXASIM_MPI)
  set(EXASIM_VARIANT cpumpi)
else()
  set(EXASIM_VARIANT cpu)
endif()
find_package(Exasim REQUIRED COMPONENTS ${EXASIM_VARIANT})
find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED)

exasim_add_external_builtin_model(TARGET ext_model_surfvis
  ID 100
  PDEMODEL ${CMAKE_CURRENT_SOURCE_DIR}/pdeapp100.txt)

add_executable(consumer_surface_vis main.cpp)
target_compile_definitions(consumer_surface_vis PRIVATE _BUILTINLIBRARY)
if(EXASIM_MPI)
  target_compile_definitions(consumer_surface_vis PRIVATE _MPI)
endif()
target_link_libraries(consumer_surface_vis PRIVATE
    Exasim::headers Exasim::pre ext_model_surfvis Kokkos::kokkos MPI::MPI_CXX)