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
option(WITH_TEXT2CODE     "Build text2code-linked executables" OFF)
option(WITH_BUILTINMODEL  "Build the built-in model library"   ON)
option(WITH_GMP           "Build vendored GMP and use it for vendored SymEngine" OFF)
option(EXASIM_BUILD_TESTS "Register the ctest regression suite" OFF)
option(EXASIM_FRONTENDS   "Install the Python/Julia/MATLAB frontends" ON)
option(EXASIM_PIP_INSTALL "pip-install the exasim Python package at install time" OFF)
option(EXASIM_JULIA_DEVELOP "Pkg.develop the installed Exasim.jl at install time" OFF)

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

# ----------------------------------------------------------------------------
# GMP: optional vendored build, installed directly into the final prefix.
# Disabled by default to preserve the existing BoostMP SymEngine workflow.
# ----------------------------------------------------------------------------
set(_gmp_source "${EXASIM_ROOT}/deps/gmp-6.3.0")
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
if(WITH_GMP AND EXISTS "${_gmp_source}/configure")
  if(EXISTS "${_gmp_include_dir}/gmp.h"
      AND EXISTS "${_gmp_include_dir}/gmpxx.h"
      AND EXISTS "${_gmp_lib_dir}/libgmp.a"
      AND EXISTS "${_gmp_lib_dir}/libgmpxx.a")
    message(STATUS "GMP: using vendored ${_gmp_install} (already built)")
  else()
    message(STATUS "GMP: building vendored -> ${_gmp_install}")
    ExternalProject_Add(gmp_vendored
      SOURCE_DIR  "${_gmp_source}"
      BINARY_DIR  "${_dep_build}/gmp-build"
      DOWNLOAD_COMMAND ""
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
    set(_gmp_dep gmp_vendored)
    list(APPEND _exasim_deps ${_gmp_dep})
  endif()
elseif(WITH_GMP)
  message(FATAL_ERROR "WITH_GMP=ON but ${_gmp_source}/configure was not found.")
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
elseif(EXASIM_HIP)
  set(_kk_target hip)
  set(_kk_var    HIP_DIR)
  set(_kk_dir    "${_dep_build}/kokkos/buildhip")
else()
  set(_kk_target serial)
  set(_kk_var    SERIAL_DIR)
  set(_kk_dir    "${_dep_build}/kokkos/buildserial")
endif()

set(_kokkos_for_exasim "")   # -DKokkos_DIR forwarded to the solver build
find_package(Kokkos QUIET)
if(Kokkos_FOUND)
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
    BUILD_COMMAND     ${_envcxx} make -f Makefile.builds ${_kk_target} ${_kk_var}=${_kk_dir}
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
    "C compiler used to build vendored GKlib/METIS/ParMETIS")
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
  if(_sb_metis AND _sb_parmetis)
    message(STATUS "METIS: using installed (${_sb_metis})")
  elseif(EXISTS "${EXASIM_ROOT}/deps/metis/METIS/build/libmetis/libmetis.a")
    message(STATUS "METIS: using vendored deps/metis/ (already built)")
  else()
    message(STATUS "METIS: building vendored with ${_exasim_metis_c_compiler}")
    ExternalProject_Add(metis_vendored
      SOURCE_DIR        "${EXASIM_ROOT}/deps/metis"
      DOWNLOAD_COMMAND  ""
      CONFIGURE_COMMAND ""
      BUILD_IN_SOURCE   TRUE
      BUILD_COMMAND     ${CMAKE_COMMAND} -E env
        "CC=${_exasim_metis_c_compiler}"
        "CXX=${CMAKE_CXX_COMPILER}"
        make metis "cc=${_exasim_metis_c_compiler}"
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
    -DKokkos_DIR=${_kokkos_for_exasim}
    -DEXASIM_MPI=${EXASIM_MPI} -DEXASIM_NOMPI=${EXASIM_NOMPI}
    -DEXASIM_CUDA=${EXASIM_CUDA} -DEXASIM_HIP=${EXASIM_HIP}
    -DEXASIM_LIB=${EXASIM_LIB} -DWITH_PARMETIS=${WITH_PARMETIS}
    -DWITH_TEXT2CODE=${WITH_TEXT2CODE} -DWITH_BUILTINMODEL=${WITH_BUILTINMODEL}
    -DEXASIM_BUILD_TESTS=${EXASIM_BUILD_TESTS}
    -DEXASIM_TEXT2CODE=${EXASIM_TEXT2CODE}
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
