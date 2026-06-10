cmake_minimum_required(VERSION 3.16)
project(exasim_app CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(EXASIM_MPI "Link the MPI-enabled Exasim preprocessing library" ON)
option(EXASIM_CUDA "Link the CUDA-enabled Exasim preprocessing library" OFF)
option(EXASIM_HIP "Link the HIP-enabled Exasim preprocessing library" OFF)

if(EXASIM_CUDA AND EXASIM_HIP)
  message(FATAL_ERROR
    "EXASIM_CUDA and EXASIM_HIP cannot both be ON. Select exactly one GPU backend.")
endif()

find_package(Exasim REQUIRED)

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

add_executable(exasimfe exasimfeapp.cpp)
if(EXASIM_MPI)
  target_compile_definitions(exasimfe PRIVATE _MPI)
endif()
target_link_libraries(exasimfe PRIVATE
  Exasim::headers
  ${EXASIM_APP_LIBRARY_TARGET})
