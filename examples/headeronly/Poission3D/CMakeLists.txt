cmake_minimum_required(VERSION 3.16)
project(Poission3D LANGUAGES CXX)

# If you installed to a non-standard prefix:
set(CMAKE_PREFIX_PATH "/Users/teoc/projects/exasim/Exasim/export" ${CMAKE_PREFIX_PATH})

find_package(exasim CONFIG REQUIRED)

add_executable(myexasim_tool main.cpp)

# Link the header/source-only variant:
target_link_libraries(myexasim_tool PRIVATE exasim::cpuheader)
