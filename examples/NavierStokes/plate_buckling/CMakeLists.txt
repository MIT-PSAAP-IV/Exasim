cmake_minimum_required(VERSION 3.16)
project(plate_buckling LANGUAGES CXX)

find_package(Exasim REQUIRED)

add_executable(exasimapp main.cpp)

# BuiltIn model headers from the in-repo backend (relative to this example, not
# an absolute developer path). Override with -DEXASIM_BUILTIN_DIR=<path> if your
# checkout layout differs.
if(NOT DEFINED EXASIM_BUILTIN_DIR)
  set(EXASIM_BUILTIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../backend/Model/BuiltIn")
endif()
target_include_directories(exasimapp PRIVATE "${EXASIM_BUILTIN_DIR}")
target_compile_definitions(exasimapp PRIVATE HAVE_BUILTINMODEL _BUILTINMODEL)

target_link_libraries(exasimapp PRIVATE
    Exasim::headers
    Exasim::cpulib
    "${Exasim_BuiltinModel_LIBRARY}")
