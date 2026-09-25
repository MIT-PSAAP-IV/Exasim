include(GNUInstallDirs)

set(_elasticity_text2code "")
if(DEFINED EXASIM_TEXT2CODE AND EXISTS "${EXASIM_TEXT2CODE}")
  set(_elasticity_text2code "${EXASIM_TEXT2CODE}")
else()
  find_program(_elasticity_text2code NAMES text2code)
endif()

set(_elasticity_generated_headers "")
if(_elasticity_text2code)
  foreach(_nd RANGE 2 3)
    set(_source_app "${CMAKE_CURRENT_SOURCE_DIR}/pdeapp${_nd}.txt")
    set(_source_model "${CMAKE_CURRENT_SOURCE_DIR}/pdemodel${_nd}.txt")
    set(_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/model${_nd}")
    file(MAKE_DIRECTORY "${_generated_dir}")
    file(READ "${_source_app}" _pdeapp)
    string(REGEX REPLACE "exasimpath[ \t]*=[ \t]*\"[^\"]*\""
           "exasimpath = \"${EXASIM_DIR}\"" _pdeapp "${_pdeapp}")
    file(WRITE "${_generated_dir}/pdeapp${_nd}.txt" "${_pdeapp}")
    configure_file("${_source_model}" "${_generated_dir}/pdemodel${_nd}.txt" COPYONLY)
    set(_model_header "${_generated_dir}/my_model.hpp")
    add_custom_command(
      OUTPUT "${_model_header}"
      COMMAND "${_elasticity_text2code}" "${_generated_dir}/pdeapp${_nd}.txt"
              --out-dir "${_generated_dir}" --gen-only
      DEPENDS "${_source_app}" "${_source_model}"
      COMMENT "text2code: generating ${_nd}D internal linear-elasticity model"
      VERBATIM)
    list(APPEND _elasticity_generated_headers "${_model_header}")
    list(APPEND _elasticity_compile_definitions
      "EXASIM_LINEAR_ELASTICITY_MODEL_${_nd}_HEADER=\"${_model_header}\"")
  endforeach()
  add_custom_target(exasim_linear_elasticity_model_codegen
    DEPENDS ${_elasticity_generated_headers})
else()
  message(STATUS
    "LinearElasticityModels: text2code not found; using checked-in C++ model fallback")
endif()

if(EXASIM_CUDA)
  set(_elasticity_target LinearElasticityModelscuda)
elseif(EXASIM_HIP)
  set(_elasticity_target LinearElasticityModelship)
else()
  set(_elasticity_target LinearElasticityModelsserial)
endif()
set(EXASIM_LINEAR_ELASTICITY_MODEL_TARGET "${_elasticity_target}" PARENT_SCOPE)

add_library(${_elasticity_target} STATIC
  provider2.cpp provider3.cpp linear_elasticity_provider.cpp)
if(TARGET exasim_linear_elasticity_model_codegen)
  add_dependencies(${_elasticity_target} exasim_linear_elasticity_model_codegen)
  target_compile_definitions(${_elasticity_target} PRIVATE
    ${_elasticity_compile_definitions})
endif()
target_include_directories(${_elasticity_target} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${EXASIM_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/backend/Model/LinearElasticity>)
target_link_libraries(${_elasticity_target} PUBLIC ExasimHeaders Kokkos::kokkos)
set_target_properties(${_elasticity_target} PROPERTIES
  OUTPUT_NAME ${_elasticity_target}
  ARCHIVE_OUTPUT_DIRECTORY "${EXASIM_LIB_DIR}"
  POSITION_INDEPENDENT_CODE ON)

install(FILES README.md pdeapp2.txt pdemodel2.txt pdeapp3.txt pdemodel3.txt
  DESTINATION "${CMAKE_INSTALL_DATADIR}/exasim/models/LinearElasticity")
