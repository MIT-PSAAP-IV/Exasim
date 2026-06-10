cmake_minimum_required(VERSION 3.16)
project(exasim_app CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(EXASIM_MPI "Link the MPI-enabled Exasim preprocessing library" ON)
option(EXASIM_CUDA "Link the CUDA-enabled Exasim preprocessing library" OFF)
option(EXASIM_HIP "Link the HIP-enabled Exasim preprocessing library" OFF)

if(EXASIM_CUDA AND EXASIM_HIP)
  message(FATAL_ERROR
    "EXASIM_CUDA and EXASIM_HIP cannot both be ON. Select exactly one GPU backend.")
endif()

if(Exasim_DIR)
  if(EXISTS "${Exasim_DIR}/ExasimConfig.cmake")
    if(EXISTS "${Exasim_DIR}/../install/CMakeLists.txt")
      get_filename_component(EXASIM_APP_PREFIX "${Exasim_DIR}/.." ABSOLUTE)
    else()
      get_filename_component(EXASIM_APP_PREFIX "${Exasim_DIR}/../../.." ABSOLUTE)
    endif()
  elseif(EXISTS "${Exasim_DIR}/build/ExasimConfig.cmake")
    get_filename_component(EXASIM_APP_PREFIX "${Exasim_DIR}" ABSOLUTE)
    set(Exasim_DIR "${EXASIM_APP_PREFIX}/build" CACHE PATH "Path to ExasimConfig.cmake" FORCE)
  elseif(EXISTS "${Exasim_DIR}/lib/cmake/Exasim/ExasimConfig.cmake")
    get_filename_component(EXASIM_APP_PREFIX "${Exasim_DIR}" ABSOLUTE)
    set(Exasim_DIR "${EXASIM_APP_PREFIX}/lib/cmake/Exasim" CACHE PATH "Path to ExasimConfig.cmake" FORCE)
  else()
    message(FATAL_ERROR
      "Exasim_DIR was set to ${Exasim_DIR}, but no ExasimConfig.cmake was found "
      "in ${Exasim_DIR}, ${Exasim_DIR}/build, or ${Exasim_DIR}/lib/cmake/Exasim.")
  endif()
endif()

find_package(Exasim REQUIRED)

set(EXASIM_APP_LIBRARY_TARGET Exasim::cpuprelib)
set(EXASIM_APP_BUILTINMODEL_LIB builtinmodelserial)
set(EXASIM_APP_BUILTINMODEL_TARGET Exasim::builtinmodelserial)
if(EXASIM_MPI)
  set(EXASIM_APP_LIBRARY_TARGET Exasim::cpumpiprelib)
endif()
if(EXASIM_CUDA OR EXASIM_HIP)
  set(EXASIM_APP_BUILTINMODEL_LIB builtinmodelcuda)
  set(EXASIM_APP_BUILTINMODEL_TARGET Exasim::builtinmodelcuda)
  if(EXASIM_HIP)
    set(EXASIM_APP_BUILTINMODEL_LIB builtinmodelhip)
    set(EXASIM_APP_BUILTINMODEL_TARGET Exasim::builtinmodelhip)
  endif()
  if(EXASIM_MPI)
    set(EXASIM_APP_LIBRARY_TARGET Exasim::gpumpiprelib)
  else()
    set(EXASIM_APP_LIBRARY_TARGET Exasim::gpuprelib)
  endif()
endif()

if(NOT TARGET ${EXASIM_APP_LIBRARY_TARGET})
  message(FATAL_ERROR
    "Requested Exasim target ${EXASIM_APP_LIBRARY_TARGET} is not available in the "
    "current Exasim package. Reconfigure Exasim to build that library variant.")
endif()

set(EXASIM_APP_BUILTINMODEL_LINK ${EXASIM_APP_BUILTINMODEL_TARGET})
if(NOT TARGET ${EXASIM_APP_BUILTINMODEL_TARGET})
  if(NOT EXASIM_APP_PREFIX)
    message(FATAL_ERROR
      "Could not determine the BuiltIn model library location from the Exasim package. "
      "Set Exasim_DIR to the Exasim root, build directory, or installed package directory.")
  endif()

  find_library(EXASIM_APP_BUILTINMODEL_LIBRARY
    NAMES ${EXASIM_APP_BUILTINMODEL_LIB}
    HINTS "${EXASIM_APP_PREFIX}/lib"
    NO_DEFAULT_PATH)
  if(NOT EXASIM_APP_BUILTINMODEL_LIBRARY)
    message(FATAL_ERROR
      "Could not find ${EXASIM_APP_BUILTINMODEL_LIB} in ${EXASIM_APP_PREFIX}/lib. "
      "Build and install the BuiltIn model library before configuring this app.")
  endif()
  set(EXASIM_APP_BUILTINMODEL_LINK ${EXASIM_APP_BUILTINMODEL_LIBRARY})
endif()

add_executable(exasimapp exasimapp.cpp)
target_include_directories(exasimapp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(exasimapp PRIVATE _BUILTINLIBRARY)
if(EXASIM_MPI)
  target_compile_definitions(exasimapp PRIVATE _MPI)
endif()
target_link_libraries(exasimapp PRIVATE
  Exasim::headers
  ${EXASIM_APP_LIBRARY_TARGET}
  ${EXASIM_APP_BUILTINMODEL_LINK})
if(DEFINED EXASIM_APP_BUILTINMODEL_LIBRARY)
  get_filename_component(EXASIM_APP_BUILTINMODEL_DIR
    "${EXASIM_APP_BUILTINMODEL_LIBRARY}" DIRECTORY)
  set_target_properties(exasimapp PROPERTIES
    BUILD_RPATH "${EXASIM_APP_BUILTINMODEL_DIR}")
endif()
