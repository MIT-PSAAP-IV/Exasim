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
    set(Exasim_DIR "${Exasim_DIR}" CACHE PATH "Path to ExasimConfig.cmake" FORCE)
  elseif(EXISTS "${Exasim_DIR}/build/ExasimConfig.cmake")
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
message(STATUS "Exasim_DIR = ${Exasim_DIR}")

set(EXASIM_APP_LIBRARY_TARGET Exasim::cpulib)
if(EXASIM_CUDA OR EXASIM_HIP)
  set(EXASIM_APP_LIBRARY_TARGET Exasim::gpulib)
  if(EXASIM_MPI)
    set(EXASIM_APP_LIBRARY_TARGET Exasim::gpumpilib)
  endif()
elseif(EXASIM_MPI)
  set(EXASIM_APP_LIBRARY_TARGET Exasim::cpumpilib)
endif()

if(NOT TARGET ${EXASIM_APP_LIBRARY_TARGET})
  message(FATAL_ERROR
    "Requested Exasim target ${EXASIM_APP_LIBRARY_TARGET} is not available in the "
    "current Exasim package. Reconfigure Exasim to build that library variant.")
endif()

add_executable(exasimapp exasimfeapp.cpp)
if(EXASIM_MPI)
  target_compile_definitions(exasimapp PRIVATE _MPI)
endif()
target_link_libraries(exasimapp PRIVATE
  Exasim::headers
  ${EXASIM_APP_LIBRARY_TARGET})
