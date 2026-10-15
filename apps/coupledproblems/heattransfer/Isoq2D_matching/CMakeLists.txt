cmake_minimum_required(VERSION 3.16)
project(exasim_text2code_coupled_problem CXX)
set(CMAKE_CXX_STANDARD 20)

# A coupled solve always runs under MPI, also with one rank per model.
set(EXASIM_VARIANT cpumpi)
set(EXASIM_SOLVER_TARGET Exasim::cpumpilib)
find_package(Exasim REQUIRED COMPONENTS ${EXASIM_VARIANT})
find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED)
set(EXASIM_TEXT2CODE_EXECUTABLE "" CACHE FILEPATH "Override text2code executable used for coupled app codegen and runtime")
if(EXASIM_TEXT2CODE_EXECUTABLE)
  set(Exasim_TEXT2CODE "${EXASIM_TEXT2CODE_EXECUTABLE}")
endif()

set(_EXASIM_COUPLED_CODEGEN_INPUT_DIR "${CMAKE_BINARY_DIR}/exasim_external_models/coupled_text2code_models/codegen_inputs")
file(MAKE_DIRECTORY "${_EXASIM_COUPLED_CODEGEN_INPUT_DIR}")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pdeapp1.txt" _exasim_coupled_pdeapp_1)
file(WRITE "${_EXASIM_COUPLED_CODEGEN_INPUT_DIR}/pdeapp1.txt" "gendatain = 0;\n${_exasim_coupled_pdeapp_1}")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/pdemodel1.txt" "${_EXASIM_COUPLED_CODEGEN_INPUT_DIR}/pdemodel1.txt" COPYONLY)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pdeapp2.txt" _exasim_coupled_pdeapp_2)
file(WRITE "${_EXASIM_COUPLED_CODEGEN_INPUT_DIR}/pdeapp2.txt" "gendatain = 0;\n${_exasim_coupled_pdeapp_2}")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/pdemodel2.txt" "${_EXASIM_COUPLED_CODEGEN_INPUT_DIR}/pdemodel2.txt" COPYONLY)

exasim_add_external_builtin_model(TARGET coupled_text2code_models
  IDS 100 101
  PDEMODEL ${_EXASIM_COUPLED_CODEGEN_INPUT_DIR}/pdeapp1.txt
           ${_EXASIM_COUPLED_CODEGEN_INPUT_DIR}/pdeapp2.txt)

add_executable(exasimapp main.cpp)
target_compile_definitions(exasimapp PRIVATE _BUILTINLIBRARY)
target_compile_definitions(exasimapp PRIVATE EXASIM_TEXT2CODE_EXECUTABLE="${Exasim_TEXT2CODE}")
target_compile_definitions(exasimapp PRIVATE _MPI)
target_link_libraries(exasimapp PRIVATE
  Exasim::headers ${EXASIM_SOLVER_TARGET} coupled_text2code_models Kokkos::kokkos)
target_link_libraries(exasimapp PRIVATE MPI::MPI_CXX)
