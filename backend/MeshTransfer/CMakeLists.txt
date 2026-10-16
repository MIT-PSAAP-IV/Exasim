cmake_minimum_required(VERSION 3.20)
project(ExasimMeshTransfer LANGUAGES CXX)

option(MESHTRANSFER_ENABLE_MPI "Enable distributed mesh transfer" ON)
option(MESHTRANSFER_BUILD_TESTS "Build MeshTransfer tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Kokkos QUIET)
if(NOT Kokkos_FOUND)
  set(_exasim_root "${CMAKE_CURRENT_SOURCE_DIR}/../..")
  if(NOT EXISTS "${_exasim_root}/deps/kokkos/CMakeLists.txt")
    message(FATAL_ERROR "Kokkos was not found and Exasim's vendored Kokkos is unavailable")
  endif()
  set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
  add_subdirectory(
    "${_exasim_root}/deps/kokkos" "${CMAKE_BINARY_DIR}/kokkos"
    EXCLUDE_FROM_ALL)
endif()

if(MESHTRANSFER_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
endif()

add_library(exasim_mesh_transfer INTERFACE)
add_library(Exasim::MeshTransfer ALIAS exasim_mesh_transfer)
target_include_directories(exasim_mesh_transfer INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_link_libraries(exasim_mesh_transfer INTERFACE Kokkos::kokkos)
if(MESHTRANSFER_ENABLE_MPI)
  target_compile_definitions(exasim_mesh_transfer INTERFACE HAVE_MPI=1)
  target_link_libraries(exasim_mesh_transfer INTERFACE MPI::MPI_CXX)
endif()

if(MESHTRANSFER_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
