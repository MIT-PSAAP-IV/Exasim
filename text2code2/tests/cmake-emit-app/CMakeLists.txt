# Minimal project proving the exasim_emit_app() cmake helper drives
# `text2code --emit-app` to produce a standalone header-only app at build time.
cmake_minimum_required(VERSION 3.16)
project(cmake_emit_app_test NONE)

# Locate ExasimEmitApp.cmake (repo cmake/ dir, two levels up from tests/).
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake")
include(ExasimEmitApp)

# The built-in pdeapp hardcodes exasimpath; rewrite it to this tree (as the
# real built-in build does) so text2code can locate SymEngine/Kokkos.
set(EXASIM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
get_filename_component(EXASIM_DIR "${EXASIM_DIR}" ABSOLUTE)
file(READ "${EXASIM_DIR}/backend/Model/BuiltIn/pdeapp8.txt" _pde)
string(REGEX REPLACE "exasimpath[ \t]*=[ \t]*\"[^\"]*\"" "exasimpath = \"${EXASIM_DIR}\"" _pde "${_pde}")
if(NOT _pde MATCHES "exasimpath[ \t]*=")
  set(_pde "exasimpath = \"${EXASIM_DIR}\";\n${_pde}")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/pdeapp8.txt" "${_pde}")
# text2code resolves modelfile relative to the pdeapp dir.
file(COPY "${EXASIM_DIR}/backend/Model/BuiltIn/pdemodel8.txt"
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

exasim_emit_app(NAME builtin8_app ALL
                PDEAPP "${CMAKE_CURRENT_BINARY_DIR}/pdeapp8.txt"
                DEST   "${CMAKE_CURRENT_BINARY_DIR}/apps/builtin8"
                MODEL_ID 8 APP_NAME builtin8)
