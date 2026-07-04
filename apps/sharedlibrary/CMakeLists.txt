cmake_minimum_required(VERSION 3.16)
project(exasim_app CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(EXASIM_MPI  "Link the MPI-enabled Exasim preprocessing library" ON)
option(EXASIM_CUDA "Link the CUDA-enabled Exasim preprocessing library" OFF)
option(EXASIM_HIP  "Link the HIP-enabled Exasim preprocessing library" OFF)

if(EXASIM_CUDA AND EXASIM_HIP)
  message(FATAL_ERROR
    "EXASIM_CUDA and EXASIM_HIP cannot both be ON. Select exactly one GPU backend.")
endif()

if(Exasim_DIR)
  if(EXISTS "${Exasim_DIR}/ExasimConfig.cmake")
    set(Exasim_DIR "${Exasim_DIR}" CACHE PATH "Path to ExasimConfig.cmake" FORCE)
  elseif(EXISTS "${Exasim_DIR}/build/ExasimConfig.cmake")
    set(Exasim_DIR "${Exasim_DIR}/build" CACHE PATH "Path to ExasimConfig.cmake" FORCE)
  elseif(EXISTS "${Exasim_DIR}/lib/cmake/Exasim/ExasimConfig.cmake")
    set(Exasim_DIR "${Exasim_DIR}/lib/cmake/Exasim" CACHE PATH "Path to ExasimConfig.cmake" FORCE)
  elseif(EXISTS "${Exasim_DIR}/lib64/cmake/Exasim/ExasimConfig.cmake")
    set(Exasim_DIR "${Exasim_DIR}/lib64/cmake/Exasim" CACHE PATH "Path to ExasimConfig.cmake" FORCE)
  else()
    message(FATAL_ERROR
      "Exasim_DIR was set to ${Exasim_DIR}, but no ExasimConfig.cmake was found.")
  endif()
endif()

set(EXASIM_VARIANT cpu)
if(EXASIM_MPI)
  set(EXASIM_VARIANT cpumpi)
endif()
if(EXASIM_CUDA OR EXASIM_HIP)
  if(EXASIM_MPI)
    set(EXASIM_VARIANT gpumpi)
  else()
    set(EXASIM_VARIANT gpu)
  endif()
endif()

find_package(Exasim REQUIRED COMPONENTS ${EXASIM_VARIANT})

# ----------------------------------------------------------------------
# Select generated shared model library.
# ----------------------------------------------------------------------
set(EXASIM_APP_MODEL_LIB t2cmodelserial)

if(EXASIM_CUDA)
  set(EXASIM_APP_MODEL_LIB t2cmodelcuda)
endif()

if(EXASIM_HIP)
  set(EXASIM_APP_MODEL_LIB t2cmodelhip)
endif()

set(EXASIM_APP_PREFIX "${Exasim_ROOT}" CACHE PATH "Exasim install prefix")
if(NOT EXASIM_APP_PREFIX)
  message(FATAL_ERROR
    "Could not determine EXASIM_APP_PREFIX. "
    "Set Exasim_DIR to the Exasim install prefix or set EXASIM_APP_PREFIX explicitly.")
endif()

message(STATUS "Exasim_DIR        = ${Exasim_DIR}")
message(STATUS "EXASIM_VARIANT    = ${EXASIM_VARIANT}")
message(STATUS "EXASIM_APP_PREFIX = ${EXASIM_APP_PREFIX}")

find_library(EXASIM_APP_MODEL_LIBRARY
  NAMES ${EXASIM_APP_MODEL_LIB}
  HINTS "${EXASIM_APP_PREFIX}/lib" "${EXASIM_APP_PREFIX}/lib64"
  NO_DEFAULT_PATH)

if(NOT EXASIM_APP_MODEL_LIBRARY)
  message(FATAL_ERROR
    "Could not find shared model library ${EXASIM_APP_MODEL_LIB} in "
    "${EXASIM_APP_PREFIX}/lib or ${EXASIM_APP_PREFIX}/lib64. "
    "Run text2code first to generate and build it.")
endif()

message(STATUS "EXASIM_APP_MODEL_LIBRARY = ${EXASIM_APP_MODEL_LIBRARY}")

# ----------------------------------------------------------------------
# Build app.
# ----------------------------------------------------------------------
add_executable(exasimapp exasimsharedapp.cpp)

target_include_directories(exasimapp BEFORE PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../include")

target_compile_definitions(exasimapp PRIVATE _SHAREDLIBRARY)

if(EXASIM_MPI)
  target_compile_definitions(exasimapp PRIVATE _MPI)
endif()

if(EXASIM_CUDA)
  target_compile_definitions(exasimapp PRIVATE _CUDA)
endif()

if(EXASIM_HIP)
  target_compile_definitions(exasimapp PRIVATE _HIP)
endif()

target_link_libraries(exasimapp PRIVATE
  Exasim::headers
  Exasim::pre
  "${EXASIM_APP_MODEL_LIBRARY}")

get_filename_component(EXASIM_APP_MODEL_LIBRARY_DIR
  "${EXASIM_APP_MODEL_LIBRARY}" DIRECTORY)

set_target_properties(exasimapp PROPERTIES
  BUILD_RPATH "${EXASIM_APP_MODEL_LIBRARY_DIR}"
  INSTALL_RPATH "${EXASIM_APP_MODEL_LIBRARY_DIR}")

install(TARGETS exasimapp
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
