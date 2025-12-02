cmake_minimum_required(VERSION 3.16)

project(test_builder
  VERSION 0.1
  LANGUAGES CXX
)

# ---- General configuration ---------------------------------------------------

# Default to Release if nothing is specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Export compile_commands.json for tooling (clangd, etc.)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Options & paths ---------------------------------------------------------

# Path to the Exasim root (so we install/run into <EXASIM_ROOT>/build)
set(EXASIM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../"
    CACHE PATH "Path to Exasim root")

# Put executables directly under Exasim/build at build time
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${EXASIM_ROOT}/backend/Preprocessing")

# Optionally auto-run the test executable after build
option(TEST_RUN_AFTER_BUILD "Run 'test' automatically after build" ON)

# ParMETIS / METIS / GKlib roots (override if your layout differs)
set(PARMETIS_ROOT "${EXASIM_ROOT}/text2code/ParMETIS"
    CACHE PATH "Path to ParMETIS root")
set(METIS_ROOT    "${EXASIM_ROOT}/text2code/METIS"
    CACHE PATH "Path to METIS root")
set(GKLIB_ROOT    "${EXASIM_ROOT}/text2code/GKlib"
    CACHE PATH "Path to GKlib root")

# Derived include/lib dirs (override if needed)
set(PARMETIS_INCLUDE_DIR "${PARMETIS_ROOT}/include"
    CACHE PATH "PARMETIS include dir")
set(PARMETIS_LIBRARY_DIR "${PARMETIS_ROOT}/lib"
    CACHE PATH "PARMETIS library dir")

# Your METIS layout: public headers under build/xinclude, library under build/libmetis
set(METIS_INCLUDE_DIR "${METIS_ROOT}/build/xinclude"
    CACHE PATH "METIS public include dir")
set(METIS_LIBRARY_DIR "${METIS_ROOT}/build/libmetis"
    CACHE PATH "METIS library dir")

set(GKLIB_INCLUDE_DIR "${GKLIB_ROOT}/include"
    CACHE PATH "GKlib include dir")
set(GKLIB_LIBRARY_DIR "${GKLIB_ROOT}/lib"
    CACHE PATH "GKlib library dir")

# Model file used as input to the test executable
set(MODEL_FILE "${CMAKE_CURRENT_SOURCE_DIR}/pdeapp.txt"
    CACHE FILEPATH "Input file for test")

# ---- BLAS / LAPACK -----------------------------------------------------------

# Try to find proper BLAS/LAPACK; fall back to -lblas -llapack if not found.
find_package(BLAS)
find_package(LAPACK)

if (NOT BLAS_FOUND)
  message(WARNING "CMake could not find BLAS; will link '-lblas' explicitly.")
endif()
if (NOT LAPACK_FOUND)
  message(WARNING "CMake could not find LAPACK; will link '-llapack' explicitly.")
endif()

# ---- MPI ---------------------------------------------------------------------

# Keep MPI optional, but prefer the C++ component and imported targets if available.
find_package(MPI COMPONENTS CXX)
if (MPI_FOUND)
  message(STATUS "MPI found: includes at ${MPI_CXX_INCLUDE_PATH}")
endif()

# ---- Helper functions --------------------------------------------------------

# Helper to add -O2 (or /O2 on MSVC), but only for non-Debug builds.
function(add_o2 target)
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    return()
  endif()

  if (MSVC)
    target_compile_options(${target} PRIVATE /O2)
  else()
    target_compile_options(${target} PRIVATE -O2)
  endif()
endfunction()

# Helper to link BLAS/LAPACK (handles imported targets and older Find modules)
function(link_blas_lapack target)
  # BLAS
  if (BLAS_FOUND)
    if (TARGET BLAS::BLAS)
      target_link_libraries(${target} PRIVATE BLAS::BLAS)
    elseif (BLAS_LIBRARIES)
      target_link_libraries(${target} PRIVATE ${BLAS_LIBRARIES})
    else()
      target_link_libraries(${target} PRIVATE blas)
    endif()
  else()
    target_link_libraries(${target} PRIVATE blas)
  endif()

  # LAPACK
  if (LAPACK_FOUND)
    if (TARGET LAPACK::LAPACK)
      target_link_libraries(${target} PRIVATE LAPACK::LAPACK)
    elseif (LAPACK_LIBRARIES)
      target_link_libraries(${target} PRIVATE ${LAPACK_LIBRARIES})
    else()
      target_link_libraries(${target} PRIVATE lapack)
    endif()
  else()
    target_link_libraries(${target} PRIVATE lapack)
  endif()
endfunction()

# Helper to link MPI (uses imported target when available)
function(link_mpi target)
  if (MPI_FOUND)
    if (TARGET MPI::MPI_CXX)
      target_link_libraries(${target} PRIVATE MPI::MPI_CXX)
    else()
      # Fallback to old-style variables
      target_include_directories(${target} SYSTEM PRIVATE ${MPI_CXX_INCLUDE_PATH} ${MPI_INCLUDE_PATH})
      target_link_libraries(${target} PRIVATE ${MPI_CXX_LIBRARIES} ${MPI_LIBRARIES})
    endif()
  endif()
endfunction()

# ---- Executable --------------------------------------------------------------

add_executable(test test.cpp)

# Per-target C++ standard (instead of global variables)
set_target_properties(test PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)

add_o2(test)

# For METIS usage
target_compile_definitions(test PRIVATE HAVE_METIS)

# Includes: ParMETIS, METIS, GKlib
target_include_directories(test PRIVATE
  "${PARMETIS_INCLUDE_DIR}"
  "${METIS_INCLUDE_DIR}"
  "${GKLIB_INCLUDE_DIR}"
)

# Link directories (only if they exist, to avoid noisy warnings)
if (EXISTS "${PARMETIS_LIBRARY_DIR}")
  target_link_directories(test PRIVATE "${PARMETIS_LIBRARY_DIR}")
endif()
if (EXISTS "${METIS_LIBRARY_DIR}")
  target_link_directories(test PRIVATE "${METIS_LIBRARY_DIR}")
endif()
if (EXISTS "${GKLIB_LIBRARY_DIR}")
  target_link_directories(test PRIVATE "${GKLIB_LIBRARY_DIR}")
endif()

# Link order: ParMETIS, METIS, GKlib, then BLAS/LAPACK, MPI
target_link_libraries(test PRIVATE
  parmetis
  metis
  GKlib
)

link_blas_lapack(test)
link_mpi(test)

# ---- Post-build auto-run -----------------------------------------------------

if (TEST_RUN_AFTER_BUILD AND EXISTS "${MODEL_FILE}")
  add_custom_command(
    TARGET test POST_BUILD
    COMMAND $<TARGET_FILE:test> "${MODEL_FILE}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"   # adjust if needed
    COMMENT "Running test on ${MODEL_FILE}"
  )
endif()

# ---- Summary messages --------------------------------------------------------

message(STATUS "EXASIM_ROOT          = ${EXASIM_ROOT}")
message(STATUS "PARMETIS_ROOT        = ${PARMETIS_ROOT}")
message(STATUS "METIS_ROOT           = ${METIS_ROOT}")
message(STATUS "GKLIB_ROOT           = ${GKLIB_ROOT}")
message(STATUS "MODEL_FILE           = ${MODEL_FILE}")

# ----------------------
# Usage examples:
#   cmake -S . -B build
#   cmake --build build -j
#
#
# If CMake cannot find BLAS/LAPACK but you know they’re on link path, it will fall back
# to linking '-lblas -llapack' as in your original commands.
