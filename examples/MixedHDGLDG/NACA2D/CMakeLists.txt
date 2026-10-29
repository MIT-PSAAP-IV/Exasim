cmake_minimum_required(VERSION 3.20)
project(exasim_mixed_naca LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
find_package(Exasim REQUIRED COMPONENTS cpumpi)
find_package(MPI REQUIRED COMPONENTS CXX)
exasim_add_external_builtin_model(TARGET naca_model ID 100 SHARED
  PDEMODEL "${CMAKE_CURRENT_SOURCE_DIR}/pdeapp.txt")
add_executable(exasimapp main.cpp)
target_compile_definitions(exasimapp PRIVATE _MPI _BUILTINLIBRARY)
set_target_properties(exasimapp PROPERTIES
  BUILD_RPATH "@loader_path;$ORIGIN" ENABLE_EXPORTS ON)
target_link_libraries(exasimapp PRIVATE Exasim::headers Exasim::cpumpiprelib
  naca_model MPI::MPI_CXX Kokkos::kokkos)
