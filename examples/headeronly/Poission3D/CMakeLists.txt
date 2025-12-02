cmake_minimum_required(VERSION 3.16)
project(Poission3D LANGUAGES CXX)

# If you installed to a non-standard prefix:
set(CMAKE_PREFIX_PATH "/Users/teoc/projects/exasim/Exasim/export" ${CMAKE_PREFIX_PATH})
set(Kokkos_DIR "/Users/teoc/projects/exasim/Exasim/kokkos/buildserial")

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
if(LAPACK_FOUND AND BLAS_FOUND)
  set(lapackblas_libraries ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
else()
  message(FATAL_ERROR "Lapack and Blas libraries are not found")
endif()


find_package(Kokkos REQUIRED)

find_package(exasim CONFIG REQUIRED)

add_executable(myexasim_tool main.cpp)

# Link the header/source-only variant:
target_link_libraries(myexasim_tool PRIVATE exasim::cpuheader
    Kokkos::kokkos
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES})
