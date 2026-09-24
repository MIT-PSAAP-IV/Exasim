cmake_minimum_required(VERSION 3.16)
project(exasim_app_isoq3d_surfvis CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# isoq3d wall heat-flux surface vis (model ID 108): the 3D Navier-Stokes
# kernels (including VisSurfScalars) are generated at build time from
# pdeapp108.txt via text2code and registered as an external built-in model.
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

exasim_add_external_builtin_model(TARGET ext_model_isoq3d
  ID 108
  PDEMODEL ${CMAKE_CURRENT_SOURCE_DIR}/pdeapp108.txt)

add_executable(exasimapp main.cpp)
target_compile_definitions(exasimapp PRIVATE _BUILTINLIBRARY)
if(EXASIM_MPI)
  target_compile_definitions(exasimapp PRIVATE _MPI)
endif()
target_link_libraries(exasimapp PRIVATE
    Exasim::headers Exasim::pre ext_model_isoq3d Kokkos::kokkos MPI::MPI_CXX)
