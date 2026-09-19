include(GNUInstallDirs)

# Regenerate the three dimension-specific model headers when text2code is
# available.  The source-tree HelmholtzModel template remains the bootstrap
# fallback for configurations that intentionally build without text2code.
set(_helmholtz_text2code "")
if(DEFINED EXASIM_TEXT2CODE AND EXISTS "${EXASIM_TEXT2CODE}")
  set(_helmholtz_text2code "${EXASIM_TEXT2CODE}")
else()
  find_program(_helmholtz_text2code NAMES text2code)
endif()

set(_helmholtz_generated_headers "")
if(_helmholtz_text2code)
  foreach(_nd RANGE 1 3)
    set(_source_app "${CMAKE_CURRENT_SOURCE_DIR}/pdeapp${_nd}.txt")
    set(_source_model "${CMAKE_CURRENT_SOURCE_DIR}/pdemodel${_nd}.txt")
    set(_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/model${_nd}")
    file(MAKE_DIRECTORY "${_generated_dir}")

    # text2code resolves modelfile relative to pdeapp and uses exasimpath to
    # locate its generator support headers.  Keep portable sources checked in,
    # then materialize a build-tree copy configured for this checkout.
    file(READ "${_source_app}" _pdeapp)
    string(REGEX REPLACE "exasimpath[ \t]*=[ \t]*\"[^\"]*\""
           "exasimpath = \"${EXASIM_DIR}\"" _pdeapp "${_pdeapp}")
    file(WRITE "${_generated_dir}/pdeapp${_nd}.txt" "${_pdeapp}")
    configure_file("${_source_model}"
                   "${_generated_dir}/pdemodel${_nd}.txt" COPYONLY)

    set(_model_header "${_generated_dir}/my_model.hpp")
    add_custom_command(
      OUTPUT "${_model_header}"
      COMMAND "${_helmholtz_text2code}"
              "${_generated_dir}/pdeapp${_nd}.txt"
              --out-dir "${_generated_dir}" --gen-only
      DEPENDS "${_source_app}" "${_source_model}"
      COMMENT "text2code: generating ${_nd}D internal Helmholtz model"
      VERBATIM)
    list(APPEND _helmholtz_generated_headers "${_model_header}")
    list(APPEND _helmholtz_compile_definitions
      "EXASIM_HELMHOLTZ_MODEL_${_nd}_HEADER=\"${_model_header}\"")
  endforeach()
  add_custom_target(exasim_helmholtz_model_codegen
    DEPENDS ${_helmholtz_generated_headers})
else()
  message(STATUS
    "HelmholtzModels: text2code not found; using checked-in C++ model fallback")
endif()

# Match the built-in model library naming convention.  A configured Exasim
# build has exactly one active execution backend, so it emits exactly one
# Helmholtz archive compiled with the same compiler and Kokkos configuration as
# the solver that consumes it.
if(EXASIM_CUDA)
  set(_helmholtz_target HelmholtzModelscuda)
elseif(EXASIM_HIP)
  set(_helmholtz_target HelmholtzModelship)
else()
  set(_helmholtz_target HelmholtzModelsserial)
endif()
set(EXASIM_HELMHOLTZ_MODEL_TARGET "${_helmholtz_target}" PARENT_SCOPE)

add_library(${_helmholtz_target} STATIC
  provider1.cpp provider2.cpp provider3.cpp helmholtzprovider.cpp)
if(TARGET exasim_helmholtz_model_codegen)
  add_dependencies(${_helmholtz_target} exasim_helmholtz_model_codegen)
  target_compile_definitions(${_helmholtz_target} PRIVATE
    ${_helmholtz_compile_definitions})
endif()
target_include_directories(${_helmholtz_target} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${EXASIM_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/backend/Model/Helmholtz>)
target_link_libraries(${_helmholtz_target} PUBLIC ExasimHeaders Kokkos::kokkos)
set_target_properties(${_helmholtz_target} PROPERTIES
  OUTPUT_NAME ${_helmholtz_target}
  ARCHIVE_OUTPUT_DIRECTORY "${EXASIM_LIB_DIR}"
  POSITION_INDEPENDENT_CODE ON)

# Ship the canonical generator inputs as part of the installation so the
# internal library remains reproducible from its mathematical model sources.
install(FILES
  README.md
  pdeapp1.txt pdemodel1.txt
  pdeapp2.txt pdemodel2.txt
  pdeapp3.txt pdemodel3.txt
  DESTINATION "${CMAKE_INSTALL_DATADIR}/exasim/models/Helmholtz")
