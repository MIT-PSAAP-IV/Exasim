cmake_minimum_required(VERSION 3.16)
project(plate_buckling LANGUAGES CXX)

find_package(Exasim REQUIRED)

add_executable(exasimapp main.cpp)

target_include_directories(exasimapp PRIVATE /home/tsili/dv/Exasim/backend/Model/BuiltIn)
target_compile_definitions(exasimapp PRIVATE HAVE_BUILTINMODEL _BUILTINMODEL)

target_link_libraries(exasimapp PRIVATE
    Exasim::headers
    Exasim::cpulib
    "${Exasim_BuiltinModel_LIBRARY}")
