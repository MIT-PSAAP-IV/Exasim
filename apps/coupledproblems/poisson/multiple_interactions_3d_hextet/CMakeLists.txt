cmake_minimum_required(VERSION 3.16)
project(exasim_multiple_interactions CXX)
set(CMAKE_CXX_STANDARD 20)
find_package(Exasim REQUIRED COMPONENTS cpumpi)
find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED)
set(EXASIM_TEXT2CODE_EXECUTABLE "" CACHE FILEPATH "text2code executable")
if(EXASIM_TEXT2CODE_EXECUTABLE)
  set(Exasim_TEXT2CODE "${EXASIM_TEXT2CODE_EXECUTABLE}")
endif()
set(_inputs "${CMAKE_BINARY_DIR}/codegen_inputs")
file(MAKE_DIRECTORY "${_inputs}")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/interaction.txt" "${_inputs}/interaction.txt" COPYONLY)
foreach(i RANGE 1 4)
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pdeapp${i}.txt" _app)
  file(WRITE "${_inputs}/pdeapp${i}.txt"
    "gendatain = 0;\ninteractionfile = \"${_inputs}/interaction.txt\";\nnumdomains = 4;\n${_app}")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/pdemodel.txt" "${_inputs}/pdemodel${i}.txt" COPYONLY)
endforeach()
exasim_add_external_builtin_model(TARGET multiple_interaction_models
  IDS 100 101 102 103
  PDEMODEL "${_inputs}/pdeapp1.txt" "${_inputs}/pdeapp2.txt"
           "${_inputs}/pdeapp3.txt" "${_inputs}/pdeapp4.txt")
add_executable(exasimapp main.cpp)
target_compile_definitions(exasimapp PRIVATE _BUILTINLIBRARY _MPI
  EXASIM_TEXT2CODE_EXECUTABLE="${Exasim_TEXT2CODE}")
target_link_libraries(exasimapp PRIVATE Exasim::headers Exasim::cpumpilib
  multiple_interaction_models Kokkos::kokkos MPI::MPI_CXX)
