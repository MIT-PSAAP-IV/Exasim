cmake_minimum_required(VERSION 3.16)
project(text2code_builder LANGUAGES CXX)

# ---- C++ standard ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Options & paths ----
option(WITH_METIS "Build text2code with METIS/GKlib" ON)

# Path to the Exasim root (so we install into <EXASIM_ROOT>/build)
# Default guesses two levels up from this CMakeLists.txt; override if needed.
set(EXASIM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../" CACHE PATH "Path to Exasim root")

# Exasim prefix location forwarded by the superbuild. Baked into the binary
# so buildDynamicLibraries() compiles the runtime model .so against the right
# metis, kokkos, and symengine libaries
set(EXASIM_INSTALL_PREFIX "" CACHE PATH "Exasim install prefix used by installed text2code")

# Where the superbuild installed the vendored SymEngine (out of the source tree),
# and whether the resolved SymEngine is that vendored one. Both forwarded by the
# superbuild; empty/OFF for a standalone build (falls back to the in-source dep).
set(EXASIM_SYMENGINE_DIR "" CACHE PATH "Vendored SymEngine install dir (from superbuild)")
option(EXASIM_SYMENGINE_VENDORED "Resolved SymEngine is the vendored one" OFF)
option(EXASIM_SYMENGINE_WITH_GMP "Resolved SymEngine was built with GMP" OFF)

# Kokkos location + backend, forwarded by the superbuild (it builds Kokkos out of
# the source tree). Baked into the binary so buildDynamicLibraries() compiles the
# runtime model .so against the right Kokkos instead of a stale exasimpath/kokkos.
set(EXASIM_KOKKOS_DIR "" CACHE PATH "Kokkos install dir (from superbuild)")
set(EXASIM_KOKKOS_BACKEND "serial" CACHE STRING "Active Kokkos backend: serial|cuda|hip")

# Put executables directly under Exasim/build at build time
# Binary lands in the build tree (the superbuild passes its own bin/ here);
# never write into the source tree.
if(NOT DEFINED EXASIM_T2C_OUTPUT_DIR)
  set(EXASIM_T2C_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin")
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${EXASIM_T2C_OUTPUT_DIR}")

# Adjust these if your layout differs; they default to your examples.
#set(METIS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../METIS" CACHE PATH "Path to METIS root")
#set(GKLIB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../GKlib" CACHE PATH "Path to GKlib root")
set(METIS_ROOT    "${EXASIM_ROOT}/deps/metis/METIS" CACHE PATH "Path to METIS root")
set(GKLIB_ROOT    "${EXASIM_ROOT}/deps/metis/GKlib" CACHE PATH "Path to GKlib root")

function(t2c_first_existing_dir outvar)
  foreach(_dir ${ARGN})
    if(EXISTS "${_dir}" AND IS_DIRECTORY "${_dir}")
      set(${outvar} "${_dir}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  list(GET ARGN 0 _fallback)
  set(${outvar} "${_fallback}" PARENT_SCOPE)
endfunction()

function(t2c_first_existing_file_dir outvar filename)
  foreach(_dir ${ARGN})
    if(EXISTS "${_dir}/${filename}")
      set(${outvar} "${_dir}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  list(GET ARGN 0 _fallback)
  set(${outvar} "${_fallback}" PARENT_SCOPE)
endfunction()

function(t2c_refresh_cache_dir_if_needed var default_dir required_file doc)
  set(${var} "${default_dir}" CACHE PATH "${doc}")
  if(NOT EXISTS "${${var}}/${required_file}" AND EXISTS "${default_dir}/${required_file}")
    set(${var} "${default_dir}" CACHE PATH "${doc}" FORCE)
  endif()
endfunction()

# Derived include/lib dirs (override if needed)
t2c_first_existing_dir(_metis_default_include
  "${METIS_ROOT}/build/xinclude"
  "${METIS_ROOT}/include")
t2c_first_existing_file_dir(_metis_default_libdir libmetis.a
  "${METIS_ROOT}/lib"
  "${METIS_ROOT}/lib64"
  "${METIS_ROOT}/build/libmetis")
t2c_first_existing_dir(_gklib_default_include
  "${GKLIB_ROOT}/include")
t2c_first_existing_file_dir(_gklib_default_libdir libGKlib.a
  "${GKLIB_ROOT}/lib"
  "${GKLIB_ROOT}/lib64"
  "${GKLIB_ROOT}/build/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(METIS_INCLUDE_DIR "${_metis_default_include}" CACHE PATH "METIS public include dir")
t2c_refresh_cache_dir_if_needed(METIS_LIBRARY_DIR
  "${_metis_default_libdir}" libmetis.a "METIS library dir")
set(GKLIB_INCLUDE_DIR "${_gklib_default_include}" CACHE PATH "GKlib include dir")
t2c_refresh_cache_dir_if_needed(GKLIB_LIBRARY_DIR
  "${_gklib_default_libdir}" libGKlib.a "GKlib library dir")

# Resolve METIS/GKlib robustly: find them on the system (conda/apt) OR fall back
# to the vendored build dirs via HINTS — exactly like the main solver build.
# Without this, text2code silently fails to build when METIS is external (e.g.
# conda on macOS), and the built-in model regen then falls back to the checked-in
# kernels instead of regenerating from text2code.
find_path(T2C_METIS_INCLUDE    metis.h   HINTS "${METIS_INCLUDE_DIR}")
find_library(T2C_METIS_LIBRARY metis     HINTS "${METIS_LIBRARY_DIR}")
find_library(T2C_GKLIB_LIBRARY GKlib     HINTS "${GKLIB_LIBRARY_DIR}")  # optional; conda metis bundles it

# ---- BLAS / LAPACK ----
# Try to find proper BLAS/LAPACK; fall back to -lblas -llapack if not found.
find_package(BLAS)
find_package(LAPACK)
if (NOT BLAS_FOUND)
  message(WARNING "CMake could not find BLAS; will link '-lblas' explicitly.")
endif()
if (NOT LAPACK_FOUND)
  message(WARNING "CMake could not find LAPACK; will link '-llapack' explicitly.")
endif()

# Helper to add -O2 (or /O2 on MSVC)
function(add_o2 target)
  if (MSVC)
    target_compile_options(${target} PRIVATE /O2)
  else()
    target_compile_options(${target} PRIVATE -O2)
  endif()
endfunction()

# Helper to link BLAS/LAPACK (found targets or raw names)
function(link_blas_lapack target)
  if (BLAS_FOUND)
    target_link_libraries(${target} PRIVATE BLAS::BLAS)
  else()
    target_link_libraries(${target} PRIVATE blas)
  endif()
  if (LAPACK_FOUND)
    target_link_libraries(${target} PRIVATE LAPACK::LAPACK)
  else()
    target_link_libraries(${target} PRIVATE lapack)
  endif()
endfunction()

# ---- SymEngine (used at runtime to compile generated Code2Cpp.cpp) ----
# Prefer an installed SymEngine (apt libsymengine-dev / conda / brew all ship
# SymEngineConfig.cmake); fall back to the vendored build under
# text2code/symengine. The resolved include/lib flags are baked into the binary
# (symengine_config.h) so CodeCompiler uses them instead of a hardcoded path.
find_package(SymEngine CONFIG QUIET
  HINTS "${EXASIM_SYMENGINE_DIR}/lib/cmake/symengine" "${EXASIM_SYMENGINE_DIR}"
        "${EXASIM_ROOT}/deps/symengine/lib/cmake/symengine"
        "${EXASIM_ROOT}/deps/symengine")
set(_se_header "${CMAKE_CURRENT_BINARY_DIR}/symengine_config.h")
if(SymEngine_FOUND)
  set(_se_inc "")
  foreach(_d ${SYMENGINE_INCLUDE_DIRS})
    string(APPEND _se_inc "-I${_d} ")
  endforeach()
  # The vendored SymEngine is built with INTEGER_CLASS=boostmp against a bundled
  # boost that lives in the SOURCE tree (deps/symengine/boost) — even though the
  # library now installs out-of-tree — and its installed config does not expose
  # that boost. When the resolved SymEngine is the vendored one (flagged by the
  # superbuild), add the source boost dir so Code2Cpp.cpp resolves
  # <boost/multiprecision/...>. An external/system SymEngine (apt, conda, brew)
  # resolves boost from system dirs; adding the vendored boost there could shadow
  # it with a different version, so this is gated on EXASIM_SYMENGINE_VENDORED.
  if(EXASIM_SYMENGINE_VENDORED AND EXISTS "${EXASIM_ROOT}/deps/symengine/boost")
    string(APPEND _se_inc "-I${EXASIM_ROOT}/deps/symengine ")
  endif()
  if(TARGET symengine)
    set(_se_lib "$<TARGET_FILE:symengine>")
  else()
    string(REPLACE ";" " " _se_lib "${SYMENGINE_LIBRARIES}")
  endif()
  
  # Add GMP dependencies required by GMP-based static SymEngine. The superbuild
  # installs GMP into EXASIM_INSTALL_PREFIX, while SymEngine itself is installed
  # in EXASIM_SYMENGINE_DIR, so check both lib and lib64 layouts.
  set(_se_extra_libs "")
  if(EXASIM_SYMENGINE_WITH_GMP)
    foreach(_se_gmp_lib
        "${EXASIM_SYMENGINE_DIR}/lib/libgmpxx.a"
        "${EXASIM_SYMENGINE_DIR}/lib64/libgmpxx.a"
        "${EXASIM_INSTALL_PREFIX}/lib/libgmpxx.a"
        "${EXASIM_INSTALL_PREFIX}/lib64/libgmpxx.a"
        "${EXASIM_SYMENGINE_DIR}/lib/libgmp.a"
        "${EXASIM_SYMENGINE_DIR}/lib64/libgmp.a"
        "${EXASIM_INSTALL_PREFIX}/lib/libgmp.a"
        "${EXASIM_INSTALL_PREFIX}/lib64/libgmp.a")
      if(EXISTS "${_se_gmp_lib}")
        string(APPEND _se_extra_libs " ${_se_gmp_lib}")
      endif()
    endforeach()
  endif()
  
  file(GENERATE OUTPUT "${_se_header}" CONTENT
  "#pragma once
  #define EXASIM_INSTALL_PREFIX \"${EXASIM_INSTALL_PREFIX}\"
  #define EXASIM_SYMENGINE_FOUND 1
  #define EXASIM_SYMENGINE_INCFLAGS \"${_se_inc}\"
  #define EXASIM_SYMENGINE_LIB \"${_se_lib}${_se_extra_libs}\"
  #define EXASIM_KOKKOS_DIR \"${EXASIM_KOKKOS_DIR}\"
  #define EXASIM_KOKKOS_BACKEND \"${EXASIM_KOKKOS_BACKEND}\"
  ")

  message(STATUS "SymEngine                = found (${SYMENGINE_INCLUDE_DIRS})")
  message(STATUS "EXASIM_SYMENGINE_FOUND   = 1")
  message(STATUS "EXASIM_SYMENGINE_INCFLAGS = ${_se_inc}")
  message(STATUS "EXASIM_SYMENGINE_LIB     = ${_se_lib}${_se_extra_libs}")
else()
  file(GENERATE OUTPUT "${_se_header}" CONTENT
"#pragma once
#define EXASIM_SYMENGINE_FOUND 0
#define EXASIM_KOKKOS_DIR \"${EXASIM_KOKKOS_DIR}\"
#define EXASIM_KOKKOS_BACKEND \"${EXASIM_KOKKOS_BACKEND}\"
")
  message(STATUS "SymEngine                = not found; text2code uses the vendored path at runtime")
  message(STATUS "EXASIM_SYMENGINE_FOUND   = 0")
  message(STATUS "EXASIM_SYMENGINE_INCFLAGS = ")
  message(STATUS "EXASIM_SYMENGINE_LIB     = ")
endif()
message(STATUS "EXASIM_KOKKOS_DIR        = ${EXASIM_KOKKOS_DIR}")
message(STATUS "EXASIM_KOKKOS_BACKEND    = ${EXASIM_KOKKOS_BACKEND}")
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

set(MODEL_FILE "${CMAKE_CURRENT_SOURCE_DIR}/pdeapp.txt" CACHE FILEPATH "Input for text2code")

# Auto-run the freshly built tool on a sample model after linking. OFF by
# default: this is a dev smoke-test that needs a valid exasimpath/cwd, which is
# not portable across machines and has no place in the tool build. Enable
# explicitly with -DTEST_RUN_AFTER_BUILD=ON.
option(TEST_RUN_AFTER_BUILD "Run the codegen tool on MODEL_FILE after build" OFF)

# ============================================================
# Target: text2code (with METIS)  (equivalent to your METIS line)
# Only if WITH_METIS=ON and the expected dirs exist
# ============================================================
if (WITH_METIS)
  if (T2C_METIS_INCLUDE AND T2C_METIS_LIBRARY)

    # GKlib: link the resolved lib if found; the vendored build also has headers.
    set(_t2c_metis_libs "${T2C_METIS_LIBRARY}")
    if (T2C_GKLIB_LIBRARY)
      list(APPEND _t2c_metis_libs "${T2C_GKLIB_LIBRARY}")
    endif()
    set(_t2c_metis_incs "${T2C_METIS_INCLUDE}")
    if (EXISTS "${GKLIB_INCLUDE_DIR}")
      list(APPEND _t2c_metis_incs "${GKLIB_INCLUDE_DIR}")
    endif()

    add_executable(text2code text2code.cpp)
    add_o2(text2code)
    target_compile_definitions(text2code PRIVATE HAVE_METIS)
    target_include_directories(text2code PRIVATE ${_t2c_metis_incs})
    target_link_libraries(text2code PRIVATE ${_t2c_metis_libs})
    link_blas_lapack(text2code)

    # Auto-run after the executable links (opt-in dev smoke-test only)
    if (TEST_RUN_AFTER_BUILD AND EXISTS "${MODEL_FILE}")
      add_custom_command(
        TARGET text2code POST_BUILD
        COMMAND $<TARGET_FILE:text2code> "${MODEL_FILE}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"   # or wherever the program expects to run
        COMMENT "Running text2code on ${MODEL_FILE}"
      )
    endif()

    # -------- t2cusecmake (HAVE_METIS + USE_CMAKE) --------
    add_executable(t2cusecmake text2code.cpp)
    add_o2(t2cusecmake)
    target_compile_definitions(t2cusecmake PRIVATE HAVE_METIS USE_CMAKE)
    target_include_directories(t2cusecmake PRIVATE ${_t2c_metis_incs})
    target_link_libraries(t2cusecmake PRIVATE ${_t2c_metis_libs})
    link_blas_lapack(t2cusecmake)

  else()
    message(FATAL_ERROR
      "WITH_METIS=ON but METIS was not found (system or vendored):\n"
      "  metis.h        -> '${T2C_METIS_INCLUDE}'\n"
      "  libmetis       -> '${T2C_METIS_LIBRARY}'\n"
      "Install libmetis-dev / conda metis, or build the vendored deps/metis. "
      "Without text2code the built-in models cannot be regenerated.")
  endif()
else()
  # ============================================================
  # Target: t2cnometis  (equivalent to: g++ -O2 -std=c++20 text2code.cpp -o t2cnometis -lblas -llapack)
  # ============================================================
  add_executable(t2cnometis text2code.cpp)
  add_o2(t2cnometis)
  link_blas_lapack(t2cnometis)

  # Auto-run after the executable links (opt-in dev smoke-test only)
  if (TEST_RUN_AFTER_BUILD AND EXISTS "${MODEL_FILE}")
    add_custom_command(
      TARGET t2cnometis POST_BUILD
      COMMAND $<TARGET_FILE:t2cnometis> "${MODEL_FILE}"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      COMMENT "Running t2cnometis on ${MODEL_FILE}"
    )
  endif()

  # -------- t2cnometisusecmake (USE_CMAKE) --------
  add_executable(t2cnometisusecmake text2code.cpp)
  add_o2(t2cnometisusecmake)
  target_compile_definitions(t2cnometisusecmake PRIVATE USE_CMAKE)
  link_blas_lapack(t2cnometisusecmake)
endif()

# ======================================================================
#  Summary of text2code configuration
# ======================================================================
message(STATUS "==================== text2code CONFIGURATION ==================")

message(STATUS "EXASIM_ROOT              = ${EXASIM_ROOT}")
message(STATUS "EXASIM_INSTALL_PREFIX    = ${EXASIM_INSTALL_PREFIX}")
message(STATUS "WITH_METIS               = ${WITH_METIS}")

message(STATUS "METIS_ROOT               = ${METIS_ROOT}")
message(STATUS "GKLIB_ROOT               = ${GKLIB_ROOT}")

message(STATUS "METIS_INCLUDE_DIR        = ${METIS_INCLUDE_DIR}")
message(STATUS "METIS_LIBRARY_DIR        = ${METIS_LIBRARY_DIR}")
message(STATUS "GKLIB_INCLUDE_DIR        = ${GKLIB_INCLUDE_DIR}")
message(STATUS "GKLIB_LIBRARY_DIR        = ${GKLIB_LIBRARY_DIR}")

message(STATUS "MODEL_FILE               = ${MODEL_FILE}")

# BLAS / LAPACK
message(STATUS "BLAS_FOUND               = ${BLAS_FOUND}")
message(STATUS "LAPACK_FOUND             = ${LAPACK_FOUND}")
message(STATUS "BLAS_LIBRARIES           = ${BLAS_LIBRARIES}")
message(STATUS "LAPACK_LIBRARIES         = ${LAPACK_LIBRARIES}")

# C++ compiler
message(STATUS "C++ COMPILER             = ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ COMPILER ID          = ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "C++ COMPILER VERSION     = ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_BUILD_TYPE         = ${CMAKE_BUILD_TYPE}")

# Targets summary
if (WITH_METIS)
  if (TARGET text2code)
    message(STATUS "Target 'text2code'       = ENABLED (HAVE_METIS)")
  else()
    message(STATUS "Target 'text2code'       = SKIPPED (missing METIS/GKlib dirs)")
  endif()

  if (TARGET t2cusecmake)
    message(STATUS "Target 't2cusecmake'     = ENABLED (HAVE_METIS + USE_CMAKE)")
  else()
    message(STATUS "Target 't2cusecmake'     = NOT CREATED")
  endif()
else()
  if (TARGET t2cnometis)
    message(STATUS "Target 't2cnometis'      = ENABLED (no METIS)")
  else()
    message(STATUS "Target 't2cnometis'      = NOT CREATED")
  endif()

  if (TARGET t2cnometisusecmake)
    message(STATUS "Target 't2cnometisusecmake' = ENABLED (no METIS + USE_CMAKE)")
  else()
    message(STATUS "Target 't2cnometisusecmake' = NOT CREATED")
  endif()
endif()

message(STATUS "===============================================================")

# ----------------------
# Usage examples:
#   cmake -S . -B build
#   cmake --build build -j
#
# To point at custom installs:
#   cmake -S . -B build -DWITH_METIS=ON \
#         -DMETIS_ROOT=/path/to/METIS \
#         -DGKLIB_ROOT=/path/to/GKlib
#
# If CMake cannot find BLAS/LAPACK but you know they’re on link path, it will fall back
# to linking '-lblas -llapack' as in your original commands.
