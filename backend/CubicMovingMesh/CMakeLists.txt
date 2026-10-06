cmake_minimum_required(VERSION 3.20)
project(ExasimCubicMovingMesh LANGUAGES CXX)

option(CUBIC_MOVING_MESH_BUILD_TESTS "Build CubicMovingMesh tests" ON)

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

add_library(exasim_cubic_moving_mesh INTERFACE)
add_library(Exasim::CubicMovingMesh ALIAS exasim_cubic_moving_mesh)
target_include_directories(exasim_cubic_moving_mesh INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_link_libraries(exasim_cubic_moving_mesh INTERFACE Kokkos::kokkos)

if(CUBIC_MOVING_MESH_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
