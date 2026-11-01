cmake_minimum_required(VERSION 3.20)
project(exasim_mixed_couette LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
find_package(Exasim REQUIRED COMPONENTS cpumpi)
find_package(MPI REQUIRED COMPONENTS CXX)

# Independent Text2Code generation, using the normal installed package.
exasim_add_external_builtin_model(TARGET couette_model ID 100
  PDEMODEL "${CMAKE_CURRENT_SOURCE_DIR}/pdeapp.txt" SHARED)
add_executable(exasimapp main.cpp)
target_compile_definitions(exasimapp PRIVATE _MPI _BUILTINLIBRARY)
target_link_libraries(exasimapp PRIVATE Exasim::headers Exasim::cpumpiprelib
  couette_model MPI::MPI_CXX Kokkos::kokkos)
set_target_properties(exasimapp PROPERTIES ENABLE_EXPORTS ON
  BUILD_RPATH "@loader_path;$ORIGIN")
