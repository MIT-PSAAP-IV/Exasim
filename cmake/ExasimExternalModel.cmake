# ExasimExternalModel.cmake — installed alongside ExasimConfig.cmake.
# Provides exasim_add_external_builtin_model() for out-of-tree consumers that
# want to register a new model ID without modifying the installed Exasim package.
#
# Usage (text2code / PDEMODEL path):
#
#   exasim_add_external_builtin_model(TARGET my_model_100
#     ID 100
#     PDEMODEL ${CMAKE_CURRENT_SOURCE_DIR}/pdeapp100.txt)
#
#   add_executable(my_solver main.cpp)
#   target_compile_definitions(my_solver PRIVATE _BUILTINLIBRARY)
#   target_link_libraries(my_solver PRIVATE
#     Exasim::headers Exasim::pre my_model_100 Kokkos::kokkos)
#
# The PDEMODEL variant runs text2code at build time to generate kernel .cpp files
# from the given pdeapp.txt into a persistent hidden directory under the consumer's
# cmake binary dir (${CMAKE_BINARY_DIR}/exasim_external_models/<target>/).
#
# Usage (SOURCES / hand-written path):
#
#   exasim_add_external_builtin_model(TARGET my_model_100
#     ID 100
#     SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/model100.hpp
#             ${CMAKE_CURRENT_SOURCE_DIR}/model100.cpp)
#
# For SOURCES, model.hpp must declare and model.cpp must define all kernel
# functions in namespace exasim_model_<ID>. dstype is already defined by the
# time the provider TU is compiled (via driver_abi.hpp -> Kokkos_Core.hpp).
#
# Usage (KERNELS / pre-generated path, used by the language frontends):
#
#   exasim_add_external_builtin_model(TARGET my_model_100
#     ID 100
#     KERNELS ${CMAKE_CURRENT_SOURCE_DIR}/kernels)
#
# KERNELS names a directory that already contains the full kernel .cpp set
# (KokkosFlux.cpp, ..., HdgFextonly.cpp) as produced by the Python/Julia/MATLAB
# gencode step. model.{hpp,cpp} are instantiated from the installed templates
# exactly as in the PDEMODEL path, and the kernel directory is put on the
# include path so model.cpp's quoted includes resolve there. Compiler depfiles
# track the kernel files, so regenerating them triggers a rebuild.
#
# In all cases, the resulting target provides getBuiltInLibraryExasimDriverABI()
# and links to the installed Exasim::builtinmodel{serial,cuda,hip} for fallthrough
# to all other model IDs. Do NOT also link Exasim::builtinmodel in the consumer;
# it will be pulled in transitively through this target.

function(exasim_add_external_builtin_model)
  cmake_parse_arguments(EXT "SHARED" "TARGET;ID;KERNELS" "PDEMODEL;SOURCES" ${ARGN})

  if(NOT EXT_TARGET)
    message(FATAL_ERROR "exasim_add_external_builtin_model: TARGET is required")
  endif()
  if(NOT EXT_ID)
    message(FATAL_ERROR "exasim_add_external_builtin_model: ID is required")
  endif()
  if(NOT EXT_PDEMODEL AND NOT EXT_SOURCES AND NOT EXT_KERNELS)
    message(FATAL_ERROR
      "exasim_add_external_builtin_model: one of PDEMODEL, SOURCES, or KERNELS is required")
  endif()

  set(_id     "${EXT_ID}")
  set(_tgt    "${EXT_TARGET}")
  set(_gendir "${CMAKE_BINARY_DIR}/exasim_external_models/${_tgt}")
  set(_modeldir "${_gendir}/model${_id}")
  file(MAKE_DIRECTORY "${_modeldir}")

  # Configure the C++ provider wrapper from the installed template.
  # configure_file() substitutes @EXT_ID@ at cmake time.
  set(EXT_ID "${_id}")
  configure_file(
    "${Exasim_CMAKE_DIR}/ExternalModelProvider.cpp.in"
    "${_gendir}/ExternalModelProvider.cpp"
    @ONLY)

  if(EXT_PDEMODEL)
    # ---- text2code (PDEMODEL) path ----------------------------------------
    if(NOT EXISTS "${Exasim_TEXT2CODE}")
      message(FATAL_ERROR
        "exasim_add_external_builtin_model(${_tgt}): text2code not found at\n"
        "  ${Exasim_TEXT2CODE}\n"
        "Set Exasim_TEXT2CODE to the path of the text2code binary.")
    endif()

    # Instantiate model.hpp and model.cpp from the installed templates,
    # renaming the namespace and fixing the absolute dstype.hpp path.
    foreach(_tmpl model.hpp model.cpp)
      file(READ "${Exasim_BUILTIN_DIR}/${_tmpl}" _txt)
      string(REPLACE "exasim_model_1" "exasim_model_${_id}" _txt "${_txt}")
      string(REPLACE "\"../dstype.hpp\""
                     "\"${Exasim_BUILTIN_DIR}/dstype.hpp\"" _txt "${_txt}")
      # Write only on change so reconfigures don't dirty mtimes (recompiles).
      set(_prev "")
      if(EXISTS "${_modeldir}/${_tmpl}")
        file(READ "${_modeldir}/${_tmpl}" _prev)
      endif()
      if(NOT _txt STREQUAL _prev)
        file(WRITE "${_modeldir}/${_tmpl}" "${_txt}")
      endif()
    endforeach()

    # Rewrite exasimpath in the pdeapp so text2code finds this Exasim install.
    # text2code looks for backend/ headers at $exasimpath/backend/...; in an
    # installed package those live under include/, so point to Exasim_TEXT2CODE_ROOT
    # ($prefix/include) rather than the prefix itself.
    file(READ "${EXT_PDEMODEL}" _pde)
    string(REGEX REPLACE
      "exasimpath[ \t]*=[ \t]*\"[^\"]*\""
      "exasimpath = \"${Exasim_TEXT2CODE_ROOT}\""
      _pde "${_pde}")
    if(NOT _pde MATCHES "exasimpath[ \t]*=")
      set(_pde "exasimpath = \"${Exasim_TEXT2CODE_ROOT}\";\n${_pde}")
    endif()
    set(_pdeapp "${_modeldir}/pdeapp.txt")
    file(WRITE "${_pdeapp}" "${_pde}")

    # Copy the pdemodel*.txt file (same directory, same base name as pdeapp)
    # so text2code can find it next to the rewritten pdeapp.
    get_filename_component(_pdeapp_dir "${EXT_PDEMODEL}" DIRECTORY)
    get_filename_component(_pdeapp_name "${EXT_PDEMODEL}" NAME)
    string(REGEX REPLACE "pdeapp([0-9]*)\.txt$" "pdemodel\\1.txt"
                         _pdemodel_name "${_pdeapp_name}")
    if(EXISTS "${_pdeapp_dir}/${_pdemodel_name}")
      file(COPY "${_pdeapp_dir}/${_pdemodel_name}" DESTINATION "${_modeldir}")
    endif()

    set(_stamp "${_modeldir}/.text2code.stamp")
    # Depend on both the pdeapp and the pdemodel so that physics changes
    # (e.g. modifying Ubou, Flux, etc.) invalidate the stamp and trigger rerun.
    set(_pdemodel_path "${_pdeapp_dir}/${_pdemodel_name}")
    set(_extra_deps)
    if(EXISTS "${_pdemodel_path}")
      set(_extra_deps "${_pdemodel_path}")
    endif()
    add_custom_command(
      OUTPUT  "${_stamp}"
      COMMAND "${Exasim_TEXT2CODE}" "${_pdeapp}" --out-dir "${_modeldir}" --gen-only
      COMMAND "${CMAKE_COMMAND}" -E touch "${_stamp}"
      DEPENDS "${EXT_PDEMODEL}" ${_extra_deps}
      COMMENT "text2code: generating model ${_id} kernels for target ${_tgt}"
      VERBATIM)
    add_custom_target(_exasim_ext_codegen_${_tgt} DEPENDS "${_stamp}")

  elseif(EXT_KERNELS)
    # ---- pre-generated kernels (KERNELS) path ------------------------------
    # The caller (a language frontend's gencode step) has already produced the
    # full kernel .cpp set in ${EXT_KERNELS}. Instantiate model.{hpp,cpp} from
    # the installed templates; their quoted kernel includes resolve through the
    # kernel directory added to the target's include path below.
    if(NOT IS_DIRECTORY "${EXT_KERNELS}")
      message(FATAL_ERROR
        "exasim_add_external_builtin_model(${_tgt}): KERNELS directory not found:\n"
        "  ${EXT_KERNELS}")
    endif()
    foreach(_tmpl model.hpp model.cpp)
      file(READ "${Exasim_BUILTIN_DIR}/${_tmpl}" _txt)
      string(REPLACE "exasim_model_1" "exasim_model_${_id}" _txt "${_txt}")
      string(REPLACE "\"../dstype.hpp\""
                     "\"${Exasim_BUILTIN_DIR}/dstype.hpp\"" _txt "${_txt}")
      # Write only on change so reconfigures don't dirty mtimes (recompiles).
      set(_prev "")
      if(EXISTS "${_modeldir}/${_tmpl}")
        file(READ "${_modeldir}/${_tmpl}" _prev)
      endif()
      if(NOT _txt STREQUAL _prev)
        file(WRITE "${_modeldir}/${_tmpl}" "${_txt}")
      endif()
    endforeach()
    add_custom_target(_exasim_ext_codegen_${_tgt})

  else()
    # ---- hand-written (SOURCES) path --------------------------------------
    # Copy user-provided sources into model<ID>/ so the relative includes
    # in ExternalModelProvider.cpp ("model<ID>/model.hpp") resolve correctly.
    foreach(_src ${EXT_SOURCES})
      get_filename_component(_sname "${_src}" NAME)
      configure_file("${_src}" "${_modeldir}/${_sname}" COPYONLY)
    endforeach()
    add_custom_target(_exasim_ext_codegen_${_tgt})
  endif()

  # Select the installed built-in model library to link for fallthrough dispatch.
  # Prefer the component-alias if the consumer used COMPONENTS, else detect.
  if(TARGET Exasim::builtinmodel)
    set(_bm_lib Exasim::builtinmodel)
  elseif(TARGET Exasim::builtinmodelcuda)
    set(_bm_lib Exasim::builtinmodelcuda)
  elseif(TARGET Exasim::builtinmodelhip)
    set(_bm_lib Exasim::builtinmodelhip)
  elseif(TARGET Exasim::builtinmodelserial)
    set(_bm_lib Exasim::builtinmodelserial)
  else()
    message(FATAL_ERROR
      "exasim_add_external_builtin_model: no Exasim builtinmodel library found.\n"
      "Use find_package(Exasim REQUIRED COMPONENTS cpu) before calling this function.")
  endif()

  # Build the provider library.
  # getBuiltInLibraryExasimDriverABI() is defined in ExternalModelProvider.cpp and
  # intercepts model ID @EXT_ID@; all other IDs fall through to the builtin dispatchers
  # in ${_bm_lib}. Link order: ext library first so its symbol definition wins over
  # any duplicate in the builtin archive.
  if(EXT_SHARED)
    # Dynamic provider (used by the language frontends): the model lives in
    # libfrontend_model.{so,dylib}; the host executable links it and never has
    # to relink when the model changes. The .so must NOT embed Kokkos — a
    # second copy of Kokkos's global state double-frees at exit on glibc's
    # flat namespace (see backend/Model/BuiltIn/CMakeLists.txt). Take Kokkos
    # as compile flags/includes only and resolve its symbols from the host
    # executable at load time (the host must set ENABLE_EXPORTS).
    add_library(${_tgt} SHARED "${_gendir}/ExternalModelProvider.cpp")
    target_include_directories(${_tgt} PRIVATE
      $<TARGET_PROPERTY:Kokkos::kokkos,INTERFACE_INCLUDE_DIRECTORIES>)
    target_compile_options(${_tgt} PRIVATE
      $<TARGET_PROPERTY:Kokkos::kokkos,INTERFACE_COMPILE_OPTIONS>)
    target_link_libraries(${_tgt} PRIVATE "${_bm_lib}")
    if(APPLE)
      target_link_options(${_tgt} PRIVATE -undefined dynamic_lookup)
    endif()
  else()
    add_library(${_tgt} STATIC "${_gendir}/ExternalModelProvider.cpp")
    target_link_libraries(${_tgt} PRIVATE "${_bm_lib}" Kokkos::kokkos)
  endif()
  add_dependencies(${_tgt} _exasim_ext_codegen_${_tgt})
  # gendir is the #include root: ExternalModelProvider.cpp uses "model<ID>/model.hpp"
  target_include_directories(${_tgt} PRIVATE "${_gendir}")
  if(EXT_KERNELS)
    # model.cpp's quoted kernel includes fall back to this search path.
    target_include_directories(${_tgt} PRIVATE "${EXT_KERNELS}")
  endif()
  target_link_libraries(${_tgt} PUBLIC  Exasim::headers)
  set_target_properties(${_tgt} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()
