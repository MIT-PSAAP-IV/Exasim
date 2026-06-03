cmake_minimum_required(VERSION 3.16)
project(exasim_app CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(EXASIM_MPI "Link the MPI-enabled Exasim preprocessing library" ON)
option(EXASIM_GPU "Link the GPU-enabled Exasim preprocessing library" OFF)

if(Exasim_DIR AND NOT EXISTS "${Exasim_DIR}/ExasimConfig.cmake")
  if(EXISTS "${Exasim_DIR}/build/ExasimConfig.cmake")
    set(Exasim_DIR "${Exasim_DIR}/build" CACHE PATH "Path to ExasimConfig.cmake" FORCE)
  elseif(EXISTS "${Exasim_DIR}/lib/cmake/Exasim/ExasimConfig.cmake")
    set(Exasim_DIR "${Exasim_DIR}/lib/cmake/Exasim" CACHE PATH "Path to ExasimConfig.cmake" FORCE)
  else()
    message(FATAL_ERROR
      "Exasim_DIR was set to ${Exasim_DIR}, but no ExasimConfig.cmake was found "
      "in ${Exasim_DIR}, ${Exasim_DIR}/build, or ${Exasim_DIR}/lib/cmake/Exasim.")
  endif()
endif()

find_package(Exasim REQUIRED)

set(EXASIM_APP_LIBRARY_TARGET Exasim::cpuprelib)
set(EXASIM_APP_BUILTINMODEL_LIB builtinmodelserial)
if(EXASIM_MPI)
  set(EXASIM_APP_LIBRARY_TARGET Exasim::cpumpiprelib)
endif()
if(EXASIM_GPU)
  set(EXASIM_APP_BUILTINMODEL_LIB builtinmodelcuda)
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

get_filename_component(EXASIM_APP_PREFIX "${Exasim_DIR}/.." ABSOLUTE)
find_library(EXASIM_APP_BUILTINMODEL_LIBRARY
  NAMES ${EXASIM_APP_BUILTINMODEL_LIB}
  HINTS "${EXASIM_APP_PREFIX}/lib"
  NO_DEFAULT_PATH)
if(NOT EXASIM_APP_BUILTINMODEL_LIBRARY)
  message(FATAL_ERROR
    "Could not find ${EXASIM_APP_BUILTINMODEL_LIB} in ${EXASIM_APP_PREFIX}/lib. "
    "Build and install the BuiltIn model library before configuring this app.")
endif()

add_executable(exasimapp main.cpp)
target_include_directories(exasimapp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(exasimapp PRIVATE _BUILTINLIBRARY)
if(EXASIM_MPI)
  target_compile_definitions(exasimapp PRIVATE _MPI)
endif()
target_link_libraries(exasimapp PRIVATE
  Exasim::headers
  ${EXASIM_APP_LIBRARY_TARGET}
  ${EXASIM_APP_BUILTINMODEL_LIBRARY})
get_filename_component(EXASIM_APP_BUILTINMODEL_DIR
  "${EXASIM_APP_BUILTINMODEL_LIBRARY}" DIRECTORY)
set_target_properties(exasimapp PROPERTIES
  BUILD_RPATH "${EXASIM_APP_BUILTINMODEL_DIR}")
