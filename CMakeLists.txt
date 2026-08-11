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
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
include(ExternalProject)
include(GNUInstallDirs)

set(EXASIM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

# ---- Refuse building inside the source tree ----
# The repo stays pristine: all build/install outputs live in the (out-of-tree)
# binary dir. Both a true in-source build and a build dir nested anywhere
# under the repo are rejected.
get_filename_component(_exasim_src "${CMAKE_SOURCE_DIR}" REALPATH)
get_filename_component(_exasim_bin "${CMAKE_BINARY_DIR}" REALPATH)
if(_exasim_bin STREQUAL _exasim_src OR _exasim_bin MATCHES "^${_exasim_src}/")
  message(FATAL_ERROR
    "Build directories inside the Exasim source tree are not supported. "
    "Configure out of tree, e.g.:\n"
    "    cmake -S ${_exasim_src} -B ${_exasim_src}-build\n"
    "(delete any CMakeCache.txt / CMakeFiles/ this attempt just created).")
endif()

# ---- Build options (forwarded to the Exasim solver build) ----
option(EXASIM_MPI         "Build MPI variants"                 ON)
option(EXASIM_NOMPI       "Build non-MPI variants"             ON)
option(EXASIM_CUDA        "CUDA backend"                       OFF)
option(EXASIM_HIP         "HIP backend"                        OFF)
option(EXASIM_LIB         "Build static libraries"             ON)
option(WITH_PARMETIS      "Link METIS/ParMETIS"                ON)
option(EXASIM_FLOAT       "Single-precision dstype (float); default double"  OFF)
option(EXASIM_INT64       "64-bit Int index type (long); default int"        OFF)
option(EXASIM_BOUNDS_CHECK "Rank-aware bounds checks on partition-derived indices (debug/CI)" OFF)
option(EXASIM_ASAN "Build with AddressSanitizer (debug/CI; needs the WHOLE binary instrumented)" OFF)

# AddressSanitizer. Applied as CMAKE_*_FLAGS rather than to one target because a
# partially-instrumented binary reports little: ASan must see the allocation AND the
# bad access. -fno-omit-frame-pointer keeps the report readable; -g gives it line
# numbers. Off by default; a default build is unaffected.
if(EXASIM_ASAN)
  set(EXASIM_ASAN_FLAGS "-fsanitize=address -fno-omit-frame-pointer -g")
  set(CMAKE_C_FLAGS            "${CMAKE_C_FLAGS} ${EXASIM_ASAN_FLAGS}")
  set(CMAKE_CXX_FLAGS          "${CMAKE_CXX_FLAGS} ${EXASIM_ASAN_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS   "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address")
  message(STATUS "EXASIM_ASAN=ON: building with AddressSanitizer")
endif()
option(WITH_TEXT2CODE     "Build text2code-linked executables" OFF)
option(WITH_BUILTINMODEL  "Build the built-in model library"   ON)
option(WITH_GMP           "Download/build GMP and use it for vendored SymEngine" OFF)
option(EXASIM_BUILD_SHARED_APP "Build and install the shared-library exasimapp executable" OFF)
option(EXASIM_BUILD_TESTS "Register the ctest regression suite" OFF)
option(EXASIM_FRONTENDS   "Install the Python/Julia/MATLAB frontends" ON)
option(EXASIM_PIP_INSTALL "pip-install the exasim Python package at install time" OFF)
option(EXASIM_JULIA_DEVELOP "Pkg.develop the installed Exasim.jl at install time" OFF)
set(EXASIM_GPU_ARCH "" CACHE STRING
  "GPU architecture for vendored Kokkos builds. Examples: gfx90a, gfx942, 70, sm_80, VOLTA70, AMPERE80, HOPPER90")

set(_exasim_kokkos_cuda_arch "")
set(_exasim_kokkos_hip_arch "")
set(_exasim_cuda_architectures "")
if(EXASIM_GPU_ARCH)
  string(STRIP "${EXASIM_GPU_ARCH}" _exasim_gpu_arch)
  string(TOUPPER "${_exasim_gpu_arch}" _exasim_gpu_arch_upper)

  if(_exasim_gpu_arch_upper MATCHES "^AMD_GFX[0-9A-Z_]+$")
    set(_exasim_kokkos_hip_arch "${_exasim_gpu_arch_upper}")
  elseif(_exasim_gpu_arch_upper MATCHES "^GFX[0-9A-Z_]+$")
    set(_exasim_kokkos_hip_arch "AMD_${_exasim_gpu_arch_upper}")
  elseif(_exasim_gpu_arch_upper MATCHES "^SM_?([0-9]+)$")
    set(_exasim_cuda_architectures "${CMAKE_MATCH_1}")
  elseif(_exasim_gpu_arch_upper MATCHES "^[0-9]+$")
    set(_exasim_cuda_architectures "${_exasim_gpu_arch_upper}")
  elseif(_exasim_gpu_arch_upper STREQUAL "VOLTA70")
    set(_exasim_kokkos_cuda_arch "VOLTA70")
    set(_exasim_cuda_architectures "70")
  elseif(_exasim_gpu_arch_upper STREQUAL "VOLTA72")
    set(_exasim_kokkos_cuda_arch "VOLTA72")
    set(_exasim_cuda_architectures "72")
  elseif(_exasim_gpu_arch_upper STREQUAL "TURING75")
    set(_exasim_kokkos_cuda_arch "TURING75")
    set(_exasim_cuda_architectures "75")
  elseif(_exasim_gpu_arch_upper STREQUAL "AMPERE80")
    set(_exasim_kokkos_cuda_arch "AMPERE80")
    set(_exasim_cuda_architectures "80")
  elseif(_exasim_gpu_arch_upper STREQUAL "AMPERE86")
    set(_exasim_kokkos_cuda_arch "AMPERE86")
    set(_exasim_cuda_architectures "86")
  elseif(_exasim_gpu_arch_upper STREQUAL "AMPERE87")
    set(_exasim_kokkos_cuda_arch "AMPERE87")
    set(_exasim_cuda_architectures "87")
  elseif(_exasim_gpu_arch_upper STREQUAL "ADA89")
    set(_exasim_kokkos_cuda_arch "ADA89")
    set(_exasim_cuda_architectures "89")
  elseif(_exasim_gpu_arch_upper STREQUAL "HOPPER90")
    set(_exasim_kokkos_cuda_arch "HOPPER90")
    set(_exasim_cuda_architectures "90")
  elseif(_exasim_gpu_arch_upper STREQUAL "BLACKWELL100")
    set(_exasim_kokkos_cuda_arch "BLACKWELL100")
    set(_exasim_cuda_architectures "100")
  elseif(_exasim_gpu_arch_upper STREQUAL "BLACKWELL120")
    set(_exasim_kokkos_cuda_arch "BLACKWELL120")
    set(_exasim_cuda_architectures "120")
  else()
    message(FATAL_ERROR
      "Unsupported EXASIM_GPU_ARCH='${EXASIM_GPU_ARCH}'. Use a HIP arch "
      "(gfx90a, gfx942, AMD_GFX90A, AMD_GFX942) or a CUDA arch "
      "(70, sm_80, VOLTA70, AMPERE80, HOPPER90, ...).")
  endif()

  if(_exasim_cuda_architectures AND NOT _exasim_kokkos_cuda_arch)
    if(_exasim_cuda_architectures STREQUAL "70")
      set(_exasim_kokkos_cuda_arch "VOLTA70")
    elseif(_exasim_cuda_architectures STREQUAL "72")
      set(_exasim_kokkos_cuda_arch "VOLTA72")
    elseif(_exasim_cuda_architectures STREQUAL "75")
      set(_exasim_kokkos_cuda_arch "TURING75")
    elseif(_exasim_cuda_architectures STREQUAL "80")
      set(_exasim_kokkos_cuda_arch "AMPERE80")
    elseif(_exasim_cuda_architectures STREQUAL "86")
      set(_exasim_kokkos_cuda_arch "AMPERE86")
    elseif(_exasim_cuda_architectures STREQUAL "87")
      set(_exasim_kokkos_cuda_arch "AMPERE87")
    elseif(_exasim_cuda_architectures STREQUAL "89")
      set(_exasim_kokkos_cuda_arch "ADA89")
    elseif(_exasim_cuda_architectures STREQUAL "90")
      set(_exasim_kokkos_cuda_arch "HOPPER90")
    elseif(_exasim_cuda_architectures STREQUAL "100")
      set(_exasim_kokkos_cuda_arch "BLACKWELL100")
    elseif(_exasim_cuda_architectures STREQUAL "120")
      set(_exasim_kokkos_cuda_arch "BLACKWELL120")
    else()
      message(FATAL_ERROR
        "EXASIM_GPU_ARCH='${EXASIM_GPU_ARCH}' gives CUDA architecture "
        "'${_exasim_cuda_architectures}', but no Kokkos architecture mapping is known.")
    endif()
  endif()

  if(EXASIM_CUDA AND _exasim_cuda_architectures)
    set(CMAKE_CUDA_ARCHITECTURES "${_exasim_cuda_architectures}" CACHE STRING
      "CUDA architectures requested through EXASIM_GPU_ARCH" FORCE)
  endif()
endif()

# Forward the toolchain to every sub-build.
set(_fwd
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
if(CMAKE_BUILD_TYPE)
  list(APPEND _fwd -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
endif()
if(DEFINED CMAKE_CUDA_ARCHITECTURES)
  list(APPEND _fwd -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES})
endif()
# Run Makefile-based vendored builds with the same compiler.
set(_envcxx ${CMAKE_COMMAND} -E env "CXX=${CMAKE_CXX_COMPILER}")

set(_exasim_deps "")   # ExternalProject targets Exasim must wait for

# Vendored deps build OUT of the source tree, into the build directory, so a
# checkout stays clean. Their source stays under deps/; only the build/install
# artifacts go here. (The deps' own Makefiles otherwise build in place.)
set(_dep_build "${CMAKE_BINARY_DIR}/deps")

function(exasim_first_existing_dir outvar)
  foreach(_dir ${ARGN})
    if(EXISTS "${_dir}" AND IS_DIRECTORY "${_dir}")
      set(${outvar} "${_dir}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  list(GET ARGN 0 _fallback)
  set(${outvar} "${_fallback}" PARENT_SCOPE)
endfunction()

function(exasim_first_existing_file outvar)
  foreach(_file ${ARGN})
    if(EXISTS "${_file}")
      set(${outvar} "${_file}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  set(${outvar} "" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------
# GMP: optional downloaded build, installed directly into the final prefix.
# Disabled by default to preserve the existing BoostMP SymEngine workflow.
# ----------------------------------------------------------------------------
set(EXASIM_GMP_VERSION "6.3.0" CACHE STRING "GMP version to download when WITH_GMP=ON")
set(EXASIM_GMP_URLS
  "https://gmplib.org/download/gmp/gmp-${EXASIM_GMP_VERSION}.tar.xz;https://ftp.gnu.org/gnu/gmp/gmp-${EXASIM_GMP_VERSION}.tar.xz"
  CACHE STRING "Semicolon-separated official GMP source archive URLs")
set(EXASIM_GMP_URL_HASH
  "SHA256=a3c2b80201b89e68616f4ad30bc66aee4927c3ce50e33929ca819d5c43538898"
  CACHE STRING "Hash for the GMP source archive")
set(_gmp_install "${CMAKE_INSTALL_PREFIX}")
set(_gmp_include_dir "${CMAKE_INSTALL_INCLUDEDIR}")
if(NOT IS_ABSOLUTE "${_gmp_include_dir}")
  set(_gmp_include_dir "${CMAKE_INSTALL_PREFIX}/${_gmp_include_dir}")
endif()
set(_gmp_lib_dir "${CMAKE_INSTALL_LIBDIR}")
if(NOT IS_ABSOLUTE "${_gmp_lib_dir}")
  set(_gmp_lib_dir "${CMAKE_INSTALL_PREFIX}/${_gmp_lib_dir}")
endif()
set(_gmp_dep "")
if(WITH_GMP)
  if(EXISTS "${_gmp_include_dir}/gmp.h"
      AND EXISTS "${_gmp_include_dir}/gmpxx.h"
      AND EXISTS "${_gmp_lib_dir}/libgmp.a"
      AND EXISTS "${_gmp_lib_dir}/libgmpxx.a")
    message(STATUS "GMP: using ${_gmp_install} (already built)")
  else()
    message(STATUS "GMP: downloading from ${EXASIM_GMP_URLS}")
    message(STATUS "GMP: building -> ${_gmp_install}")
    ExternalProject_Add(gmp_downloaded
      URL         ${EXASIM_GMP_URLS}
      URL_HASH    "${EXASIM_GMP_URL_HASH}"
      BINARY_DIR  "${_dep_build}/gmp-build"
      CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env
        "CC=${CMAKE_C_COMPILER}"
        "CXX=${CMAKE_CXX_COMPILER}"
        <SOURCE_DIR>/configure
        "--prefix=${_gmp_install}"
        "--includedir=${_gmp_include_dir}"
        "--libdir=${_gmp_lib_dir}"
        --enable-cxx
        --with-pic
        --disable-shared
        --enable-static
      BUILD_COMMAND ${CMAKE_MAKE_PROGRAM}
      INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install)
    set(_gmp_dep gmp_downloaded)
    list(APPEND _exasim_deps ${_gmp_dep})
  endif()
endif()

# ----------------------------------------------------------------------------
# Kokkos: installed -> already-built vendored -> build vendored.
# The vendored backend (serial/cuda/hip) is chosen from the requested EXASIM
# backend. Its location is forwarded to the solver build as -DKokkos_DIR.
# CUDA/HIP use nvcc_wrapper/hipcc (set inside Makefile.builds).
# ----------------------------------------------------------------------------
if(EXASIM_CUDA)
  set(_kk_target cuda)
  set(_kk_var    CUDA_DIR)
  set(_kk_dir    "${_dep_build}/kokkos/buildcuda")
  if(_exasim_kokkos_cuda_arch)
    set(_kk_arch_arg "KOKKOS_CUDA_ARCH=${_exasim_kokkos_cuda_arch}")
  else()
    set(_kk_arch_arg "")
  endif()
elseif(EXASIM_HIP)
  set(_kk_target hip)
  set(_kk_var    HIP_DIR)
  set(_kk_dir    "${_dep_build}/kokkos/buildhip")
  if(_exasim_kokkos_hip_arch)
    set(_kk_arch_arg "KOKKOS_HIP_ARCH=${_exasim_kokkos_hip_arch}")
  else()
    set(_kk_arch_arg "")
  endif()
else()
  set(_kk_target serial)
  set(_kk_var    SERIAL_DIR)
  set(_kk_dir    "${_dep_build}/kokkos/buildserial")
  set(_kk_arch_arg "")
endif()

if(EXASIM_GPU_ARCH AND EXASIM_CUDA AND NOT _exasim_kokkos_cuda_arch)
  message(FATAL_ERROR
    "EXASIM_GPU_ARCH='${EXASIM_GPU_ARCH}' is not a CUDA architecture. "
    "Use a CUDA value such as 70, sm_80, VOLTA70, AMPERE80, or HOPPER90.")
endif()
if(EXASIM_GPU_ARCH AND EXASIM_HIP AND NOT _exasim_kokkos_hip_arch)
  message(FATAL_ERROR
    "EXASIM_GPU_ARCH='${EXASIM_GPU_ARCH}' is not a HIP architecture. "
    "Use a HIP value such as gfx90a, gfx942, AMD_GFX90A, or AMD_GFX942.")
endif()

message(STATUS "EXASIM_GPU_ARCH          = ${EXASIM_GPU_ARCH}")
if(EXASIM_CUDA)
  message(STATUS "KOKKOS_CUDA_ARCH         = ${_exasim_kokkos_cuda_arch}")
  message(STATUS "CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")
elseif(EXASIM_HIP)
  message(STATUS "KOKKOS_HIP_ARCH          = ${_exasim_kokkos_hip_arch}")
endif()

set(_kokkos_for_exasim "")   # -DKokkos_DIR forwarded to the solver build
find_package(Kokkos QUIET)

# A find_package "success" is not enough: a faulty/half-installed Kokkos (e.g. a
# system package whose relocatable config collapsed _IMPORT_PREFIX to "") exports
# a Kokkos::kokkos target with a non-existent include path such as "/include".
# CMake only errors on that much later ("Imported target ... includes non-existent
# path /include"), after it has already been forwarded to every consumer. Verify
# the advertised include dirs actually exist; if not, reject this Kokkos and fall
# through to the vendored tiers below.
set(_kokkos_usable FALSE)
if(Kokkos_FOUND AND TARGET Kokkos::kokkos)
  set(_kokkos_usable TRUE)
  set(_kk_incs "")
  foreach(_kk_tgt Kokkos::kokkos Kokkos::kokkoscore)
    if(TARGET ${_kk_tgt})
      get_target_property(_kk_inc ${_kk_tgt} INTERFACE_INCLUDE_DIRECTORIES)
      if(_kk_inc)
        list(APPEND _kk_incs ${_kk_inc})
      endif()
    endif()
  endforeach()
  foreach(_kk_inc IN LISTS _kk_incs)
    # Skip generator expressions (e.g. $<BUILD_INTERFACE:...>) we cannot resolve here.
    if(NOT _kk_inc MATCHES "\\$<" AND NOT EXISTS "${_kk_inc}")
      message(STATUS "Kokkos: ignoring installed ${Kokkos_DIR} "
                     "(advertises non-existent include path '${_kk_inc}')")
      set(_kokkos_usable FALSE)
      break()
    endif()
  endforeach()
endif()

if(_kokkos_usable)
  message(STATUS "Kokkos: using installed (${Kokkos_DIR})")
  set(_kokkos_for_exasim "${Kokkos_DIR}")
elseif(EXISTS "${_kk_dir}/KokkosConfig.cmake")
  message(STATUS "Kokkos: using vendored ${_kk_dir} (already built)")
  set(_kokkos_for_exasim "${_kk_dir}")
else()
  message(STATUS "Kokkos: building vendored ${_kk_target} -> ${_kk_dir}")
  ExternalProject_Add(kokkos_vendored
    SOURCE_DIR        "${EXASIM_ROOT}/deps/kokkos"
    DOWNLOAD_COMMAND  ""
    CONFIGURE_COMMAND ""
    BUILD_IN_SOURCE   TRUE
    BUILD_COMMAND     ${_envcxx} make -f Makefile.builds ${_kk_target} ${_kk_var}=${_kk_dir} ${_kk_arch_arg}
    INSTALL_COMMAND   "")
  list(APPEND _exasim_deps kokkos_vendored)
  set(_kokkos_for_exasim "${_kk_dir}")
endif()

install(CODE "
  foreach(_exasim_kokkos_backend buildserial buildcuda buildhip)
    set(_exasim_kokkos_src \"${_dep_build}/kokkos/\${_exasim_kokkos_backend}\")
    set(_exasim_kokkos_dst \"\${CMAKE_INSTALL_PREFIX}/deps/kokkos/\${_exasim_kokkos_backend}\")
    foreach(_exasim_kokkos_subdir include lib)
      if(EXISTS \"\${_exasim_kokkos_src}/\${_exasim_kokkos_subdir}\")
        file(INSTALL
          DESTINATION \"\${_exasim_kokkos_dst}\"
          TYPE DIRECTORY
          FILES \"\${_exasim_kokkos_src}/\${_exasim_kokkos_subdir}\")
      endif()
    endforeach()
  endforeach()
")

# ----------------------------------------------------------------------------
# METIS / ParMETIS: installed -> already-built vendored -> build vendored
# ----------------------------------------------------------------------------
if(WITH_PARMETIS)
  find_library(_sb_metis    metis)
  find_library(_sb_parmetis parmetis)
  set(EXASIM_METIS_C_COMPILER "" CACHE STRING
    "C compiler used to build vendored GKlib/METIS")
  set(EXASIM_PARMETIS_C_COMPILER "" CACHE STRING
    "MPI C compiler used to build vendored ParMETIS")
  set(_exasim_metis_c_compiler "${EXASIM_METIS_C_COMPILER}")
  if(NOT _exasim_metis_c_compiler)
    if(EXASIM_HIP)
      find_program(_exasim_amdclang amdclang)
    endif()
    if(_exasim_amdclang)
      set(_exasim_metis_c_compiler "${_exasim_amdclang}")
    else()
      set(_exasim_metis_c_compiler "${CMAKE_C_COMPILER}")
    endif()
  endif()
  set(_exasim_parmetis_c_compiler "${EXASIM_PARMETIS_C_COMPILER}")
  if(NOT _exasim_parmetis_c_compiler)
    find_package(MPI QUIET COMPONENTS C)
    if(MPI_C_FOUND AND MPI_C_COMPILER)
      set(_exasim_parmetis_c_compiler "${MPI_C_COMPILER}")
    else()
      set(_exasim_parmetis_c_compiler "${_exasim_metis_c_compiler}")
    endif()
  endif()
  if(_sb_metis AND _sb_parmetis)
    message(STATUS "METIS: using installed (${_sb_metis})")
  else()
    exasim_first_existing_file(_exasim_vendored_gklib_archive
      "${EXASIM_ROOT}/deps/metis/GKlib/lib/libGKlib.a"
      "${EXASIM_ROOT}/deps/metis/GKlib/lib64/libGKlib.a"
      "${EXASIM_ROOT}/deps/metis/GKlib/build/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/libGKlib.a"
      "${EXASIM_ROOT}/deps/metis/GKlib/build/libGKlib.a")
    exasim_first_existing_file(_exasim_vendored_metis_archive
      "${EXASIM_ROOT}/deps/metis/METIS/lib/libmetis.a"
      "${EXASIM_ROOT}/deps/metis/METIS/lib64/libmetis.a"
      "${EXASIM_ROOT}/deps/metis/METIS/build/libmetis/libmetis.a")
    exasim_first_existing_file(_exasim_vendored_parmetis_archive
      "${EXASIM_ROOT}/deps/metis/ParMETIS/lib/libparmetis.a"
      "${EXASIM_ROOT}/deps/metis/ParMETIS/lib64/libparmetis.a"
      "${EXASIM_ROOT}/deps/metis/ParMETIS/build/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/libparmetis.a"
      "${EXASIM_ROOT}/deps/metis/ParMETIS/build/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/libparmetis/libparmetis.a")
  endif()
  if(_sb_metis AND _sb_parmetis)
    # Already handled above.
  elseif(_exasim_vendored_gklib_archive
      AND _exasim_vendored_metis_archive
      AND _exasim_vendored_parmetis_archive)
    message(STATUS "METIS: using vendored deps/metis/ (already built)")
  else()
    message(STATUS
      "METIS: building vendored with ${_exasim_metis_c_compiler}; "
      "ParMETIS with ${_exasim_parmetis_c_compiler}")
    ExternalProject_Add(metis_vendored
      SOURCE_DIR        "${EXASIM_ROOT}/deps/metis"
      DOWNLOAD_COMMAND  ""
      CONFIGURE_COMMAND ""
      BUILD_IN_SOURCE   TRUE
      BUILD_COMMAND     ${CMAKE_COMMAND} -E env
        "CC=${_exasim_metis_c_compiler}"
        "CXX=${CMAKE_CXX_COMPILER}"
        make metis
          "cc=${_exasim_metis_c_compiler}"
          "parmetis_cc=${_exasim_parmetis_c_compiler}"
      INSTALL_COMMAND   "")
    list(APPEND _exasim_deps metis_vendored)
  endif()

  # METIS: derived include/lib dirs (override if needed)
  set(PARMETIS_ROOT "${EXASIM_ROOT}/deps/metis/ParMETIS" CACHE PATH "Path to ParMETIS root")
  exasim_first_existing_dir(_parmetis_default_include
    "${PARMETIS_ROOT}/include")
  exasim_first_existing_dir(_parmetis_default_libdir
    "${PARMETIS_ROOT}/lib"
    "${PARMETIS_ROOT}/lib64"
    "${PARMETIS_ROOT}/build/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/libparmetis")
  set(PARMETIS_INCLUDE_DIR "${_parmetis_default_include}" CACHE PATH "ParMETIS include dir")
  set(PARMETIS_LIBRARY_DIR "${_parmetis_default_libdir}" CACHE PATH "ParMETIS library dir")

  # METIS layout: public headers under build/xinclude, library under build/libmetis
  set(METIS_ROOT    "${EXASIM_ROOT}/deps/metis/METIS" CACHE PATH "Path to METIS root")
  exasim_first_existing_dir(_metis_default_include
    "${METIS_ROOT}/build/xinclude"
    "${METIS_ROOT}/include")
  exasim_first_existing_dir(_metis_default_libdir
    "${METIS_ROOT}/lib"
    "${METIS_ROOT}/lib64"
    "${METIS_ROOT}/build/libmetis")
  set(METIS_INCLUDE_DIR "${_metis_default_include}" CACHE PATH "METIS public include dir")
  set(METIS_LIBRARY_DIR "${_metis_default_libdir}" CACHE PATH "METIS library dir")

  set(GKLIB_ROOT    "${EXASIM_ROOT}/deps/metis/GKlib" CACHE PATH "Path to GKlib root")
  exasim_first_existing_dir(_gklib_default_include
    "${GKLIB_ROOT}/include")
  exasim_first_existing_dir(_gklib_default_libdir
    "${GKLIB_ROOT}/lib"
    "${GKLIB_ROOT}/lib64"
    "${GKLIB_ROOT}/build/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
  set(GKLIB_INCLUDE_DIR "${_gklib_default_include}" CACHE PATH "GKlib include dir")
  set(GKLIB_LIBRARY_DIR "${_gklib_default_libdir}" CACHE PATH "GKlib library dir")

  install(CODE "
    set(_exasim_install_includedir \"${CMAKE_INSTALL_INCLUDEDIR}\")
    if(NOT IS_ABSOLUTE \"\${_exasim_install_includedir}\")
      set(_exasim_install_includedir \"\${CMAKE_INSTALL_PREFIX}/\${_exasim_install_includedir}\")
    endif()

    set(_exasim_install_libdir \"${CMAKE_INSTALL_LIBDIR}\")
    if(NOT IS_ABSOLUTE \"\${_exasim_install_libdir}\")
      set(_exasim_install_libdir \"\${CMAKE_INSTALL_PREFIX}/\${_exasim_install_libdir}\")
    endif()

    set(_exasim_metis_headers
      \"${METIS_INCLUDE_DIR}/metis.h\"
      \"${METIS_ROOT}/build/xinclude/metis.h\"
      \"${METIS_ROOT}/include/metis.h\"
      \"${PARMETIS_INCLUDE_DIR}/parmetis.h\")
    set(_exasim_metis_installed_headers)
    foreach(_exasim_metis_header IN LISTS _exasim_metis_headers)
      if(EXISTS \"\${_exasim_metis_header}\")
        get_filename_component(_exasim_metis_header_name \"\${_exasim_metis_header}\" NAME)
        list(FIND _exasim_metis_installed_headers \"\${_exasim_metis_header_name}\" _exasim_metis_header_index)
        if(_exasim_metis_header_index EQUAL -1)
          file(INSTALL
            DESTINATION \"\${_exasim_install_includedir}\"
            TYPE FILE
            FILES \"\${_exasim_metis_header}\")
          list(APPEND _exasim_metis_installed_headers \"\${_exasim_metis_header_name}\")
        endif()
      endif()
    endforeach()

    set(_exasim_metis_archives
      \"${PARMETIS_LIBRARY_DIR}/libparmetis.a\"
      \"${PARMETIS_ROOT}/lib/libparmetis.a\"
      \"${PARMETIS_ROOT}/lib64/libparmetis.a\"
      \"${PARMETIS_ROOT}/build/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/libparmetis/libparmetis.a\"
      \"${METIS_LIBRARY_DIR}/libmetis.a\"
      \"${METIS_ROOT}/lib/libmetis.a\"
      \"${METIS_ROOT}/lib64/libmetis.a\"
      \"${METIS_ROOT}/build/libmetis/libmetis.a\"
      \"${GKLIB_LIBRARY_DIR}/libGKlib.a\"
      \"${GKLIB_ROOT}/lib/libGKlib.a\"
      \"${GKLIB_ROOT}/lib64/libGKlib.a\"
      \"${GKLIB_ROOT}/build/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/libGKlib.a\")
    set(_exasim_metis_installed_archives)
    foreach(_exasim_metis_archive IN LISTS _exasim_metis_archives)
      if(EXISTS \"\${_exasim_metis_archive}\")
        get_filename_component(_exasim_metis_archive_name \"\${_exasim_metis_archive}\" NAME)
        list(FIND _exasim_metis_installed_archives \"\${_exasim_metis_archive_name}\" _exasim_metis_archive_index)
        if(_exasim_metis_archive_index EQUAL -1)
          file(INSTALL
            DESTINATION \"\${_exasim_install_libdir}\"
            TYPE FILE
            FILES \"\${_exasim_metis_archive}\")
          list(APPEND _exasim_metis_installed_archives \"\${_exasim_metis_archive_name}\")
        endif()
      endif()
    endforeach()
  ")
endif()

# ----------------------------------------------------------------------------
# SymEngine (text2code only): installed -> already-built vendored -> build vendored
# ----------------------------------------------------------------------------
if(WITH_GMP)
  set(_se_install "${_dep_build}/symengine-gmp")
else()
  set(_se_install "${_dep_build}/symengine")
endif()
set(_symengine_dep "")
set(_symengine_for_t2c "")     # -DEXASIM_SYMENGINE_DIR forwarded to text2code
set(_symengine_vendored OFF)
if(WITH_GMP)
  set(_symengine_integer_class gmp)
  set(_symengine_with_gmp ON)
  set(_symengine_gmp_args
    -DGMP_INCLUDE_DIR=${_gmp_include_dir}
    -DGMP_LIBRARY=${_gmp_lib_dir}/libgmp.a
    -DGMPXX_INCLUDE_DIR=${_gmp_include_dir}
    -DGMPXX_LIBRARY=${_gmp_lib_dir}/libgmpxx.a)
else()
  set(_symengine_integer_class boostmp)
  set(_symengine_with_gmp OFF)
  set(_symengine_gmp_args "")
endif()
if(NOT WITH_GMP)
  find_package(SymEngine CONFIG QUIET)
endif()
if(SymEngine_FOUND)
  message(STATUS "SymEngine: using installed")
elseif(EXISTS "${_se_install}/lib/cmake/symengine/SymEngineConfig.cmake")
  message(STATUS "SymEngine: using vendored ${_se_install} (already built)")
  set(_symengine_for_t2c "${_se_install}")
  set(_symengine_vendored ON)
else()
  message(STATUS "SymEngine: building vendored -> ${_se_install}")
  ExternalProject_Add(symengine_vendored
    SOURCE_DIR  "${EXASIM_ROOT}/deps/symengine"
    BINARY_DIR  "${_dep_build}/symengine-build"
    DOWNLOAD_COMMAND ""
    CMAKE_ARGS  ${_fwd}
      -DCMAKE_INSTALL_PREFIX=${_se_install}
      -DBUILD_SHARED_LIBS=OFF -DWITH_GMP=${_symengine_with_gmp}
      -DWITH_MPFR=OFF -DWITH_LLVM=OFF
      -DWITH_SYMENGINE_THREAD_SAFE=ON -DINTEGER_CLASS=${_symengine_integer_class}
      ${_symengine_gmp_args}
      "-DCMAKE_CXX_FLAGS=-std=gnu++17 -fPIC")
  set(_symengine_dep symengine_vendored)
  if(_gmp_dep)
    add_dependencies(symengine_vendored ${_gmp_dep})
  endif()
  set(_symengine_for_t2c "${_se_install}")
  set(_symengine_vendored ON)
endif()

install(CODE "
  if(EXISTS \"${_se_install}\")
    file(INSTALL
      DESTINATION \"\${CMAKE_INSTALL_PREFIX}/deps\"
      TYPE DIRECTORY
      FILES \"${_se_install}\")
  endif()
")

# ----------------------------------------------------------------------------
# text2code (Exasim's own codegen tool; depends on SymEngine + METIS)
# Its binary lands in ${CMAKE_BINARY_DIR}/bin/text2code.
# ----------------------------------------------------------------------------
set(EXASIM_TEXT2CODE "${CMAKE_BINARY_DIR}/bin/text2code")
ExternalProject_Add(text2code_build
  SOURCE_DIR  "${EXASIM_ROOT}/text2code/text2code"
  DOWNLOAD_COMMAND ""
  CMAKE_ARGS  ${_fwd} -DWITH_METIS=${WITH_PARMETIS} -DEXASIM_ROOT=${EXASIM_ROOT}
    -DEXASIM_T2C_OUTPUT_DIR=${CMAKE_BINARY_DIR}/bin
    -DEXASIM_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
    -DEXASIM_SYMENGINE_DIR=${_symengine_for_t2c}
    -DEXASIM_SYMENGINE_VENDORED=${_symengine_vendored}
    -DEXASIM_SYMENGINE_WITH_GMP=${WITH_GMP}
    -DEXASIM_KOKKOS_DIR=${_kokkos_for_exasim}
    -DEXASIM_KOKKOS_BACKEND=${_kk_target}
  BUILD_ALWAYS TRUE
  INSTALL_COMMAND ""
  DEPENDS ${_symengine_dep} ${_exasim_deps})
list(APPEND _exasim_deps text2code_build)

if(EXASIM_BUILD_SHARED_APP)
  # Generate the shared-library model used by the installed exasimapp smoke case.
  # This must complete before the solver layer is configured, because the app target
  # links against the text2code-produced libt2cmodel{serial,cuda,hip}.
  set(EXASIM_SHARED_APP_TEXT2CODE_CASE
    "${EXASIM_ROOT}/apps/navierstokes/naca0012steady"
    CACHE PATH "Text2Code case used to generate the shared-library exasimapp model")
  ExternalProject_Add(exasim_shared_app_model
    SOURCE_DIR  "${EXASIM_SHARED_APP_TEXT2CODE_CASE}"
    DOWNLOAD_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND
      ${CMAKE_COMMAND} -E make_directory "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_INSTALL_PREFIX}/include"
      COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${EXASIM_ROOT}/include" "${CMAKE_INSTALL_PREFIX}/include"
      COMMAND ${CMAKE_COMMAND} -E make_directory
        "${CMAKE_INSTALL_PREFIX}/backend/Model/Text2codeGenerated"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${EXASIM_ROOT}/backend/Model/Text2codeGenerated/SymEngineFunctionWrappers.hpp"
        "${CMAKE_INSTALL_PREFIX}/backend/Model/Text2codeGenerated/SymEngineFunctionWrappers.hpp"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${EXASIM_ROOT}/backend/Model/Text2codeGenerated/libt2cmodel.cpp"
        "${CMAKE_INSTALL_PREFIX}/backend/Model/Text2codeGenerated/libt2cmodel.cpp"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${EXASIM_ROOT}/backend/Model/Text2codeGenerated/libt2cmodel.hpp"
        "${CMAKE_INSTALL_PREFIX}/backend/Model/Text2codeGenerated/libt2cmodel.hpp"
      COMMAND ${CMAKE_COMMAND} -E chdir "${EXASIM_SHARED_APP_TEXT2CODE_CASE}"
        ${CMAKE_COMMAND} -E env "EXASIM_PREFIX=${CMAKE_INSTALL_PREFIX}"
        "${EXASIM_TEXT2CODE}" pdeapp.txt
    BUILD_ALWAYS TRUE
    INSTALL_COMMAND ""
    DEPENDS text2code_build)
  list(APPEND _exasim_deps exasim_shared_app_model)
endif()

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
if(DEFINED BLAS_LIBRARIES)
  list(APPEND _exasim_fwd -DBLAS_LIBRARIES=${BLAS_LIBRARIES})
endif()
if(DEFINED LAPACK_LIBRARIES)
  list(APPEND _exasim_fwd -DLAPACK_LIBRARIES=${LAPACK_LIBRARIES})
endif()
if(EXASIM_MPI)
  if(DEFINED MPI_C_COMPILER)
    list(APPEND _exasim_fwd -DMPI_C_COMPILER=${MPI_C_COMPILER})
  endif()
  if(DEFINED MPI_CXX_COMPILER)
    list(APPEND _exasim_fwd -DMPI_CXX_COMPILER=${MPI_CXX_COMPILER})
  endif()
  if(DEFINED MPI_HOME)
    list(APPEND _exasim_fwd -DMPI_HOME=${MPI_HOME})
  endif()
endif()
if(DEFINED CMAKE_INSTALL_LIBDIR)
  list(APPEND _exasim_fwd -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR})
endif()
if(CMAKE_CXX_FLAGS)
  string(REPLACE ";" "\\;" _escaped_cxx_flags "${CMAKE_CXX_FLAGS}")
  list(APPEND _exasim_fwd "-DCMAKE_CXX_FLAGS=${_escaped_cxx_flags}")
endif()

ExternalProject_Add(exasim_build
  SOURCE_DIR  "${EXASIM_ROOT}/install"
  DOWNLOAD_COMMAND ""
  CMAKE_ARGS  ${_exasim_fwd}
    -DKokkos_DIR=${_kokkos_for_exasim}
    -DEXASIM_MPI=${EXASIM_MPI} -DEXASIM_NOMPI=${EXASIM_NOMPI}
    -DEXASIM_CUDA=${EXASIM_CUDA} -DEXASIM_HIP=${EXASIM_HIP}
    -DEXASIM_LIB=${EXASIM_LIB} -DWITH_PARMETIS=${WITH_PARMETIS}
    -DEXASIM_FLOAT=${EXASIM_FLOAT} -DEXASIM_INT64=${EXASIM_INT64}
    -DEXASIM_BOUNDS_CHECK=${EXASIM_BOUNDS_CHECK}
    -DEXASIM_ASAN=${EXASIM_ASAN}
    -DWITH_TEXT2CODE=${WITH_TEXT2CODE} -DWITH_BUILTINMODEL=${WITH_BUILTINMODEL}
    -DEXASIM_BUILD_TESTS=${EXASIM_BUILD_TESTS}
    -DEXASIM_TEXT2CODE=${EXASIM_TEXT2CODE}
    -DEXASIM_BUILD_SHARED_APP=${EXASIM_BUILD_SHARED_APP}
    -DEXASIM_SHARED_APP_PREFIX=${CMAKE_INSTALL_PREFIX}
    -DEXASIM_FRONTENDS=${EXASIM_FRONTENDS}
    -DEXASIM_PIP_INSTALL=${EXASIM_PIP_INSTALL}
    -DEXASIM_JULIA_DEVELOP=${EXASIM_JULIA_DEVELOP}
    "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}"
  BUILD_ALWAYS TRUE
  INSTALL_COMMAND ""
  DEPENDS ${_exasim_deps})

ExternalProject_Get_Property(exasim_build BINARY_DIR)
install(CODE
  "execute_process(
     COMMAND \"${CMAKE_COMMAND}\" --install \"${BINARY_DIR}\" --prefix \"\${CMAKE_INSTALL_PREFIX}\"
     RESULT_VARIABLE _rv
   )
   if(NOT _rv EQUAL 0)
     message(FATAL_ERROR \"Installing exasim_build failed with exit code \${_rv}\")
   endif()")

message(STATUS "Exasim superbuild configured. Build with: cmake --build ${CMAKE_BINARY_DIR}")
