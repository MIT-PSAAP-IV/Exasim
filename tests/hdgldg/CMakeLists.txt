cmake_minimum_required(VERSION 3.20)
project(exasim_hdgldg_checks LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
get_filename_component(EXASIM_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
find_package(Exasim REQUIRED COMPONENTS cpu cpumpi)
find_package(Kokkos REQUIRED)
enable_testing()
find_path(HDGLDG_PATHS_INCLUDE exasim_paths.h
  HINTS "${Exasim_DIR}/exasim_gen" ${Exasim_INCLUDE_DIRS})
if(NOT HDGLDG_PATHS_INCLUDE)
  message(FATAL_ERROR "Set HDGLDG_PATHS_INCLUDE to Exasim's generated-header directory")
endif()

# Use the same configured METIS/GKlib as Exasim, not an unrelated system copy.
find_path(HDGLDG_METIS_INCLUDE metis.h
  HINTS "${EXASIM_SOURCE}/deps/metis/METIS/build/xinclude")
find_library(HDGLDG_METIS_LIBRARY metis
  HINTS "${EXASIM_SOURCE}/deps/metis/METIS/build/libmetis"
        "${EXASIM_SOURCE}/../exasim_install/lib")
find_library(HDGLDG_GKLIB_LIBRARY GKlib
  HINTS "${EXASIM_SOURCE}/deps/metis/GKlib/build"
        "${EXASIM_SOURCE}/../exasim_install/lib")
if(NOT HDGLDG_METIS_INCLUDE OR NOT HDGLDG_METIS_LIBRARY OR NOT HDGLDG_GKLIB_LIBRARY)
  message(FATAL_ERROR "Rank-separated setup checks need Exasim's METIS and GKlib")
endif()

add_executable(hdgldg_public public.cpp)
target_include_directories(hdgldg_public PRIVATE "${EXASIM_SOURCE}"
  "${HDGLDG_METIS_INCLUDE}" "${HDGLDG_PATHS_INCLUDE}")
target_compile_definitions(hdgldg_public PRIVATE HAVE_PARMETIS _MPI _KOKKOSKERNEL)
target_link_libraries(hdgldg_public PRIVATE Exasim::headers Exasim::cpumpiprelib Kokkos::kokkos)
foreach(ranks 2 3 4)
  add_test(NAME hdgldg_public_np${ranks}
    COMMAND ${CMAKE_COMMAND} "-DMPIEXEC=${MPIEXEC_EXECUTABLE}" "-DNP_FLAG=${MPIEXEC_NUMPROC_FLAG}"
    "-DRANKS=${ranks}" "-DPREPARE=$<TARGET_FILE:hdgldg_solver>" "-DAPP=$<TARGET_FILE:hdgldg_public>"
    "-DROOT=${CMAKE_CURRENT_BINARY_DIR}/public-np${ranks}" -P "${CMAKE_CURRENT_SOURCE_DIR}/public.cmake")
  set_tests_properties(hdgldg_public_np${ranks} PROPERTIES
    ENVIRONMENT "EXASIM_DATA_DIR=${EXASIM_SOURCE}/backend/Preprocessing" TIMEOUT 60)
endforeach()
foreach(mode nonlinear compressible)
  add_test(NAME hdgldg_solver_${mode}_np2
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
    $<TARGET_FILE:hdgldg_solver> "${CMAKE_CURRENT_BINARY_DIR}/solver-${mode}-np2" --${mode})
  set_tests_properties(hdgldg_solver_${mode}_np2 PROPERTIES
    ENVIRONMENT "EXASIM_DATA_DIR=${EXASIM_SOURCE}/backend/Preprocessing" TIMEOUT 60)
endforeach()

add_executable(hdgldg_solver solver.cpp)
target_include_directories(hdgldg_solver PRIVATE "${EXASIM_SOURCE}"
  "${HDGLDG_METIS_INCLUDE}" "${HDGLDG_PATHS_INCLUDE}")
target_compile_definitions(hdgldg_solver PRIVATE HAVE_PARMETIS _MPI)
target_link_libraries(hdgldg_solver PRIVATE Exasim::headers Exasim::cpumpiprelib Kokkos::kokkos)
foreach(ranks 2 3 4)
  add_test(NAME hdgldg_solver_np${ranks}
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
    $<TARGET_FILE:hdgldg_solver> "${CMAKE_CURRENT_BINARY_DIR}/solver-np${ranks}")
  set_tests_properties(hdgldg_solver_np${ranks} PROPERTIES
    ENVIRONMENT "EXASIM_DATA_DIR=${EXASIM_SOURCE}/backend/Preprocessing" TIMEOUT 60)
endforeach()

foreach(target hdgldg_stages hdgldg_pure_lifecycle)
  if(target STREQUAL hdgldg_stages)
    set(source stages.cpp)
  else()
    set(source pure_lifecycle.cpp)
  endif()
  add_executable(${target} "${source}")
  target_include_directories(${target} PRIVATE "${EXASIM_SOURCE}"
    "${HDGLDG_METIS_INCLUDE}" "${HDGLDG_PATHS_INCLUDE}")
  target_compile_definitions(${target} PRIVATE HAVE_PARMETIS _MPI _KOKKOSKERNEL)
  target_link_libraries(${target} PRIVATE Exasim::headers Exasim::cpumpiprelib Kokkos::kokkos)
endforeach()
foreach(ranks 2 3 4)
  foreach(order 2 3)
    add_test(NAME hdgldg_stages_dirk${order}_np${ranks}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
      $<TARGET_FILE:hdgldg_stages> "${CMAKE_CURRENT_BINARY_DIR}/stages-${order}-np${ranks}" ${order})
    set_tests_properties(hdgldg_stages_dirk${order}_np${ranks} PROPERTIES
      ENVIRONMENT "EXASIM_DATA_DIR=${EXASIM_SOURCE}/backend/Preprocessing" TIMEOUT 60)
  endforeach()
endforeach()

add_executable(hdgldg_output output.cpp)
target_include_directories(hdgldg_output PRIVATE "${EXASIM_SOURCE}"
  "${HDGLDG_METIS_INCLUDE}" "${HDGLDG_PATHS_INCLUDE}")
target_compile_definitions(hdgldg_output PRIVATE HAVE_PARMETIS _MPI)
target_link_libraries(hdgldg_output PRIVATE Exasim::headers Exasim::cpumpiprelib Kokkos::kokkos)
foreach(ranks 2 3 4)
  add_test(NAME hdgldg_output_np${ranks}
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
    $<TARGET_FILE:hdgldg_output> "${CMAKE_CURRENT_BINARY_DIR}/output-np${ranks}")
  set_tests_properties(hdgldg_output_np${ranks} PROPERTIES
    ENVIRONMENT "EXASIM_DATA_DIR=${EXASIM_SOURCE}/backend/Preprocessing" TIMEOUT 60)
endforeach()

add_executable(hdgldg_preprocessing preprocessing.cpp)
target_include_directories(hdgldg_preprocessing PRIVATE
  "${EXASIM_SOURCE}" "${HDGLDG_METIS_INCLUDE}")
target_compile_definitions(hdgldg_preprocessing PRIVATE HAVE_METIS)
target_link_libraries(hdgldg_preprocessing PRIVATE
  "${HDGLDG_METIS_LIBRARY}" "${HDGLDG_GKLIB_LIBRARY}")
add_test(NAME hdgldg_preprocessing COMMAND hdgldg_preprocessing)

foreach(target hdgldg_native_regions exasim_hdgldg_preprocess)
  if(target STREQUAL hdgldg_native_regions)
    set(source native_regions.cpp)
  else()
    set(source "${EXASIM_SOURCE}/backend/Main/hdgldgpreprocess.cpp")
  endif()
  add_executable(${target} "${source}")
  target_include_directories(${target} PRIVATE "${EXASIM_SOURCE}" "${HDGLDG_METIS_INCLUDE}"
    "${HDGLDG_PATHS_INCLUDE}")
  target_compile_definitions(${target} PRIVATE HAVE_PARMETIS)
  target_link_libraries(${target} PRIVATE Exasim::headers Exasim::cpuprelib Kokkos::kokkos)
endforeach()

add_executable(hdgldg_derivative derivative.cpp)
target_include_directories(hdgldg_derivative PRIVATE "${EXASIM_SOURCE}"
  "${HDGLDG_METIS_INCLUDE}" "${HDGLDG_PATHS_INCLUDE}")
target_compile_definitions(hdgldg_derivative PRIVATE HAVE_PARMETIS _MPI)
target_link_libraries(hdgldg_derivative PRIVATE Exasim::headers Exasim::cpumpiprelib Kokkos::kokkos)
foreach(ranks 2 3 4)
  add_test(NAME hdgldg_derivative_np${ranks}
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
    $<TARGET_FILE:hdgldg_derivative> "${CMAKE_CURRENT_BINARY_DIR}/derivative-np${ranks}")
  set_tests_properties(hdgldg_derivative_np${ranks} PROPERTIES
    ENVIRONMENT "EXASIM_DATA_DIR=${EXASIM_SOURCE}/backend/Preprocessing" TIMEOUT 60)
endforeach()
add_test(NAME hdgldg_derivative_nonlinear_np2
  COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
  $<TARGET_FILE:hdgldg_derivative> "${CMAKE_CURRENT_BINARY_DIR}/derivative-nonlinear" --nonlinear)
add_test(NAME hdgldg_derivative_3d_np2
  COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
  $<TARGET_FILE:hdgldg_derivative> "${CMAKE_CURRENT_BINARY_DIR}/derivative-3d" --3d)
set_tests_properties(hdgldg_derivative_nonlinear_np2 hdgldg_derivative_3d_np2 PROPERTIES
  ENVIRONMENT "EXASIM_DATA_DIR=${EXASIM_SOURCE}/backend/Preprocessing" TIMEOUT 60)
add_test(NAME hdgldg_native_regions COMMAND hdgldg_native_regions
  "${CMAKE_CURRENT_BINARY_DIR}/native-region-output")
set_tests_properties(hdgldg_native_regions PROPERTIES
  ENVIRONMENT "EXASIM_DATA_DIR=${EXASIM_SOURCE}/backend/Preprocessing" TIMEOUT 60)

add_executable(hdgldg_residual residual.cpp)
target_include_directories(hdgldg_residual PRIVATE "${EXASIM_SOURCE}"
  "${HDGLDG_METIS_INCLUDE}" "${HDGLDG_PATHS_INCLUDE}")
target_compile_definitions(hdgldg_residual PRIVATE HAVE_PARMETIS _MPI)
target_link_libraries(hdgldg_residual PRIVATE Exasim::headers Exasim::cpumpiprelib Kokkos::kokkos)
add_test(NAME hdgldg_residual_np2 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
  $<TARGET_FILE:hdgldg_residual> "${CMAKE_CURRENT_BINARY_DIR}/residual-region-output")
set_tests_properties(hdgldg_residual_np2 PROPERTIES
  ENVIRONMENT "EXASIM_DATA_DIR=${EXASIM_SOURCE}/backend/Preprocessing" TIMEOUT 60)
add_test(NAME hdgldg_residual_3d_np2 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
  $<TARGET_FILE:hdgldg_residual> "${CMAKE_CURRENT_BINARY_DIR}/residual-3d-region-output" --3d)
set_tests_properties(hdgldg_residual_3d_np2 PROPERTIES
  ENVIRONMENT "EXASIM_DATA_DIR=${EXASIM_SOURCE}/backend/Preprocessing" TIMEOUT 60)

foreach(ranks 3 4)
  foreach(dimension 2 3)
    set(arguments "${CMAKE_CURRENT_BINARY_DIR}/residual-${dimension}d-np${ranks}")
    if(dimension EQUAL 3)
      list(APPEND arguments --3d)
    endif()
    add_test(NAME hdgldg_residual_${dimension}d_np${ranks}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
      $<TARGET_FILE:hdgldg_residual> ${arguments})
    set_tests_properties(hdgldg_residual_${dimension}d_np${ranks} PROPERTIES
      ENVIRONMENT "EXASIM_DATA_DIR=${EXASIM_SOURCE}/backend/Preprocessing" TIMEOUT 60)
  endforeach()
endforeach()
add_test(NAME hdgldg_residual_zero_interface_np4
  COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
  $<TARGET_FILE:hdgldg_residual> "${CMAKE_CURRENT_BINARY_DIR}/residual-zero-interface" --no-interface)
set_tests_properties(hdgldg_residual_zero_interface_np4 PROPERTIES
  ENVIRONMENT "EXASIM_DATA_DIR=${EXASIM_SOURCE}/backend/Preprocessing" TIMEOUT 60)
foreach(ranks 2 3 4)
  foreach(dimension 2 3)
    set(arguments "${CMAKE_CURRENT_BINARY_DIR}/periodic-${dimension}d-np${ranks}" --periodic)
    if(dimension EQUAL 3)
      list(APPEND arguments --3d)
    endif()
    add_test(NAME hdgldg_periodic_${dimension}d_np${ranks}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
      $<TARGET_FILE:hdgldg_residual> ${arguments})
    set_tests_properties(hdgldg_periodic_${dimension}d_np${ranks} PROPERTIES
      ENVIRONMENT "EXASIM_DATA_DIR=${EXASIM_SOURCE}/backend/Preprocessing" TIMEOUT 60)
  endforeach()
endforeach()

