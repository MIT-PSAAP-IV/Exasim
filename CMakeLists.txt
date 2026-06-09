# =============================================================================
#  Exasim superbuild — one-command build.
#
#      cmake -B build [-D<option>=...]
#      cmake --build build
#
#  Orchestrates the dependency chain in order: each heavy dependency is used from
#  an installed copy when available (find-or-build), otherwise the vendored source
#  is built; then text2code, then Exasim itself. Advanced users can still build the
#  solver layer directly with `cmake -S install -B build`.
#
#  Dependencies (prefer installed, e.g. apt install libmetis-dev libparmetis-dev
#  libsymengine-dev; vendored fallback otherwise):
#    - Kokkos      (find_package; vendored deps/kokkos/buildserial)
#    - METIS       (find_library; vendored deps/metis/)
#    - SymEngine   (find_package; vendored deps/symengine/)  [text2code only]
# =============================================================================
cmake_minimum_required(VERSION 3.16)
project(ExasimSuperbuild C CXX)
include(ExternalProject)

set(EXASIM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

# ---- Discourage building inside the source tree ----
# A true in-source build (binary dir == source dir) is rejected outright. A build
# directory *inside* the repo is allowed but warned about: the vendored deps and
# generated outputs would otherwise clutter the checkout. Prefer a sibling dir:
#     cmake -B ../Exasim-build && cmake --build ../Exasim-build
get_filename_component(_exasim_src "${CMAKE_SOURCE_DIR}" REALPATH)
get_filename_component(_exasim_bin "${CMAKE_BINARY_DIR}" REALPATH)
if(_exasim_bin STREQUAL _exasim_src)
  message(FATAL_ERROR
    "In-source build detected. Configure in a separate build directory, e.g.:\n"
    "    cmake -B ../Exasim-build && cmake --build ../Exasim-build\n"
    "(delete the CMakeCache.txt and CMakeFiles/ this attempt just created).")
endif()
if(_exasim_bin MATCHES "^${_exasim_src}/")
  message(WARNING
    "Build directory is inside the Exasim source tree:\n  ${CMAKE_BINARY_DIR}\n"
    "This works, but build outputs clutter the checkout. Prefer a sibling "
    "directory: cmake -B ../Exasim-build")
endif()

# ---- Build options (forwarded to the Exasim solver build) ----
option(EXASIM_MPI         "Build MPI variants"                 ON)
option(EXASIM_NOMPI       "Build non-MPI variants"             ON)
option(EXASIM_CUDA        "CUDA backend"                       OFF)
option(EXASIM_HIP         "HIP backend"                        OFF)
option(EXASIM_LIB         "Build static libraries"             ON)
option(WITH_PARMETIS      "Link METIS/ParMETIS"                ON)
option(WITH_TEXT2CODE     "Build text2code-linked executables" OFF)
option(WITH_BUILTINMODEL  "Build the built-in model library"   ON)
option(EXASIM_BUILD_TESTS "Register the ctest regression suite" OFF)

# Forward the toolchain to every sub-build.
set(_fwd
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
if(CMAKE_BUILD_TYPE)
  list(APPEND _fwd -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
endif()
# Run Makefile-based vendored builds with the same compiler.
set(_envcxx ${CMAKE_COMMAND} -E env "CXX=${CMAKE_CXX_COMPILER}")

set(_exasim_deps "")   # ExternalProject targets Exasim must wait for

# ----------------------------------------------------------------------------
# Kokkos: installed -> already-built vendored -> build vendored.
# The vendored backend (serial/cuda/hip) is chosen from the requested EXASIM
# backend so the inner Exasim build (which picks deps/kokkos/build{serial,cuda,hip}
# by the same flags) finds a matching Kokkos. CUDA/HIP use nvcc_wrapper/hipcc.
# ----------------------------------------------------------------------------
if(EXASIM_CUDA)
  set(_kk_target cuda)
  set(_kk_dir    "${EXASIM_ROOT}/deps/kokkos/buildcuda")
elseif(EXASIM_HIP)
  set(_kk_target hip)
  set(_kk_dir    "${EXASIM_ROOT}/deps/kokkos/buildhip")
else()
  set(_kk_target serial)
  set(_kk_dir    "${EXASIM_ROOT}/deps/kokkos/buildserial")
endif()

find_package(Kokkos QUIET)
if(Kokkos_FOUND)
  message(STATUS "Kokkos: using installed (${Kokkos_DIR})")
elseif(EXISTS "${_kk_dir}/KokkosConfig.cmake")
  message(STATUS "Kokkos: using vendored ${_kk_dir} (already built)")
else()
  message(STATUS "Kokkos: building vendored ${_kk_target}")
  ExternalProject_Add(kokkos_vendored
    SOURCE_DIR        "${EXASIM_ROOT}/deps/kokkos"
    DOWNLOAD_COMMAND  ""
    CONFIGURE_COMMAND ""
    BUILD_IN_SOURCE   TRUE
    BUILD_COMMAND     ${_envcxx} make -f Makefile.builds ${_kk_target}
    INSTALL_COMMAND   "")
  list(APPEND _exasim_deps kokkos_vendored)
endif()

# ----------------------------------------------------------------------------
# METIS / ParMETIS: installed -> already-built vendored -> build vendored
# ----------------------------------------------------------------------------
if(WITH_PARMETIS)
  find_library(_sb_metis    metis)
  find_library(_sb_parmetis parmetis)
  if(_sb_metis AND _sb_parmetis)
    message(STATUS "METIS: using installed (${_sb_metis})")
  elseif(EXISTS "${EXASIM_ROOT}/deps/metis/METIS/build/libmetis/libmetis.a")
    message(STATUS "METIS: using vendored deps/metis/ (already built)")
  else()
    message(STATUS "METIS: building vendored")
    ExternalProject_Add(metis_vendored
      SOURCE_DIR        "${EXASIM_ROOT}/deps/metis"
      DOWNLOAD_COMMAND  ""
      CONFIGURE_COMMAND ""
      BUILD_IN_SOURCE   TRUE
      BUILD_COMMAND     ${_envcxx} make metis
      INSTALL_COMMAND   "")
    list(APPEND _exasim_deps metis_vendored)
  endif()
endif()

# ----------------------------------------------------------------------------
# SymEngine (text2code only): installed -> already-built vendored -> build vendored
# ----------------------------------------------------------------------------
set(_symengine_dep "")
find_package(SymEngine CONFIG QUIET)
if(SymEngine_FOUND)
  message(STATUS "SymEngine: using installed")
elseif(EXISTS "${EXASIM_ROOT}/deps/symengine/lib/cmake/symengine/SymEngineConfig.cmake")
  message(STATUS "SymEngine: using vendored deps/symengine (already built)")
else()
  message(STATUS "SymEngine: building vendored")
  ExternalProject_Add(symengine_vendored
    SOURCE_DIR  "${EXASIM_ROOT}/deps/symengine"
    BINARY_DIR  "${EXASIM_ROOT}/deps/symengine/build"
    DOWNLOAD_COMMAND ""
    CMAKE_ARGS  ${_fwd}
      -DCMAKE_INSTALL_PREFIX=${EXASIM_ROOT}/deps/symengine
      -DBUILD_SHARED_LIBS=OFF -DWITH_GMP=OFF -DWITH_MPFR=OFF -DWITH_LLVM=OFF
      -DWITH_SYMENGINE_THREAD_SAFE=ON -DINTEGER_CLASS=boostmp
      "-DCMAKE_CXX_FLAGS=-std=gnu++17 -fPIC")
  set(_symengine_dep symengine_vendored)
endif()

# ----------------------------------------------------------------------------
# text2code (Exasim's own codegen tool; depends on SymEngine + METIS)
# Installs its binary to ${EXASIM_ROOT}/build/text2code.
# ----------------------------------------------------------------------------
ExternalProject_Add(text2code_build
  SOURCE_DIR  "${EXASIM_ROOT}/text2code/text2code"
  DOWNLOAD_COMMAND ""
  CMAKE_ARGS  ${_fwd} -DWITH_METIS=${WITH_PARMETIS} -DEXASIM_ROOT=${EXASIM_ROOT}
  INSTALL_COMMAND ""
  DEPENDS ${_symengine_dep} ${_exasim_deps})
list(APPEND _exasim_deps text2code_build)

# ----------------------------------------------------------------------------
# Exasim solver layer (install/). Its own CMake find-or-builds Kokkos/METIS, so it
# locates whatever the steps above produced. Depends on everything.
#
# Unlike deps/symengine (host-only), the solver compiles Kokkos device code,
# so under CUDA/HIP it must build with the Kokkos compiler wrapper as CXX.
# ----------------------------------------------------------------------------
if(EXASIM_CUDA)
  set(_exasim_cxx "${EXASIM_ROOT}/deps/kokkos/bin/nvcc_wrapper")
elseif(EXASIM_HIP)
  set(_exasim_cxx "hipcc")
else()
  set(_exasim_cxx "${CMAKE_CXX_COMPILER}")
endif()
set(_exasim_fwd -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${_exasim_cxx})
if(CMAKE_BUILD_TYPE)
  list(APPEND _exasim_fwd -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
endif()

ExternalProject_Add(exasim_build
  SOURCE_DIR  "${EXASIM_ROOT}/install"
  DOWNLOAD_COMMAND ""
  CMAKE_ARGS  ${_exasim_fwd}
    -DEXASIM_MPI=${EXASIM_MPI} -DEXASIM_NOMPI=${EXASIM_NOMPI}
    -DEXASIM_CUDA=${EXASIM_CUDA} -DEXASIM_HIP=${EXASIM_HIP}
    -DEXASIM_LIB=${EXASIM_LIB} -DWITH_PARMETIS=${WITH_PARMETIS}
    -DWITH_TEXT2CODE=${WITH_TEXT2CODE} -DWITH_BUILTINMODEL=${WITH_BUILTINMODEL}
    -DEXASIM_BUILD_TESTS=${EXASIM_BUILD_TESTS}
    "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}"
  BUILD_ALWAYS TRUE
  INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR>
  DEPENDS ${_exasim_deps})

message(STATUS "Exasim superbuild configured. Build with: cmake --build ${CMAKE_BINARY_DIR}")
