# ExasimExternalModel.cmake — installed alongside ExasimConfig.cmake.
# Provides exasim_add_external_builtin_model() for out-of-tree consumers that
# want to register one or more new model IDs without modifying the installed
# Exasim package.
#
# Usage (text2code / PDEMODEL path, single model):
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
# Usage (SOURCES / hand-written path, single model):
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
#   # single model
#   exasim_add_external_builtin_model(TARGET frontend_model
#     ID 100
#     KERNELS ${CMAKE_CURRENT_SOURCE_DIR}/kernels)
#
#   # several generated models in ONE executable (combined multi-PDE): pass
#   # parallel ID and kernel-dir lists. Each model gets its own
#   # exasim_model_<ID> namespace and the single provider dispatches all of them.
#   exasim_add_external_builtin_model(TARGET frontend_model
#     IDS 100 101
#     KERNELS_DIRS "${dir100}" "${dir101}"
#     SHARED)
#
# KERNELS/KERNELS_DIRS name directories that already contain the full kernel
# .cpp set (KokkosFlux.cpp, ..., HdgFextonly.cpp) as produced by the
# Python/Julia/MATLAB gencode step. model.{hpp,cpp} are instantiated from the
# installed templates exactly as in the PDEMODEL path. For a single model the
# kernel directory is put on the include path so model.cpp's quoted includes
# resolve there; for several models each model's kernels are copied into its own
# model<ID>/ directory so the quoted includes resolve per-model (the directory
# of the including file is searched before the include path), with no collision.
#
# In all cases, the resulting target provides getBuiltInLibraryExasimDriverABI()
# and links to the installed Exasim::builtinmodel{serial,cuda,hip} for fallthrough
# to all other model IDs. Do NOT also link Exasim::builtinmodel in the consumer;
# it will be pulled in transitively through this target.

function(exasim_add_external_builtin_model)
  cmake_parse_arguments(EXT "SHARED" "TARGET;ID;KERNELS"
    "IDS;KERNELS_DIRS;PDEMODEL;SOURCES" ${ARGN})

  if(NOT EXT_TARGET)
    message(FATAL_ERROR "exasim_add_external_builtin_model: TARGET is required")
  endif()

  # Back-compat: the singular ID/KERNELS map onto the list forms, so existing
  # single-model callers (and the out-of-tree consumer tests) are unchanged.
  if(NOT EXT_IDS AND NOT "${EXT_ID}" STREQUAL "")
    set(EXT_IDS "${EXT_ID}")
  endif()
  if(NOT EXT_KERNELS_DIRS AND EXT_KERNELS)
    set(EXT_KERNELS_DIRS "${EXT_KERNELS}")
  endif()

  if(NOT EXT_IDS)
    message(FATAL_ERROR "exasim_add_external_builtin_model: ID (or IDS) is required")
  endif()
  if(NOT EXT_PDEMODEL AND NOT EXT_SOURCES AND NOT EXT_KERNELS_DIRS)
    message(FATAL_ERROR
      "exasim_add_external_builtin_model: one of PDEMODEL, SOURCES, or KERNELS/KERNELS_DIRS is required")
  endif()

  set(_tgt    "${EXT_TARGET}")
  set(_gendir "${CMAKE_BINARY_DIR}/exasim_external_models/${_tgt}")
  list(LENGTH EXT_IDS _nids)

  # ---- Configure the C++ provider over ALL ids --------------------------------
  # The provider TU #includes every model<id>/model.{hpp,cpp} and its 43 ext*
  # functions dispatch by id via the EXASIM_EXT_MODEL_IDS(X) X-macro list.
  set(_includes "")
  set(_idmacro "#define EXASIM_EXT_MODEL_IDS(X)")
  foreach(_id IN LISTS EXT_IDS)
    string(APPEND _includes
      "#include \"model${_id}/model.hpp\"\n#include \"model${_id}/model.cpp\"\n")
    string(APPEND _idmacro " X(${_id})")
  endforeach()
  set(EXT_MODEL_INCLUDES "${_includes}")
  set(EXT_MODEL_IDS_MACRO "${_idmacro}")
  configure_file(
    "${Exasim_CMAKE_DIR}/ExternalModelProvider.cpp.in"
    "${_gendir}/ExternalModelProvider.cpp"
    @ONLY)

  # Helper: instantiate model.hpp/model.cpp for one id from the installed
  # templates, renaming the namespace and fixing the absolute dstype.hpp path.
  # (Inlined per call site below because CMake lacks closures.)

  if(EXT_PDEMODEL)
    # ---- text2code (PDEMODEL) path — single model only --------------------
    if(NOT _nids EQUAL 1)
      message(FATAL_ERROR
        "exasim_add_external_builtin_model: PDEMODEL supports a single ID")
    endif()
    set(_id "${EXT_IDS}")
    set(_modeldir "${_gendir}/model${_id}")
    file(MAKE_DIRECTORY "${_modeldir}")

    if(NOT EXISTS "${Exasim_TEXT2CODE}")
      message(FATAL_ERROR
        "exasim_add_external_builtin_model(${_tgt}): text2code not found at\n"
        "  ${Exasim_TEXT2CODE}\n"
        "Set Exasim_TEXT2CODE to the path of the text2code binary.")
    endif()

    foreach(_tmpl model.hpp model.cpp)
      file(READ "${Exasim_BUILTIN_DIR}/${_tmpl}" _txt)
      string(REPLACE "exasim_model_1" "exasim_model_${_id}" _txt "${_txt}")
      string(REPLACE "\"../dstype.hpp\""
                     "\"${Exasim_BUILTIN_DIR}/dstype.hpp\"" _txt "${_txt}")
      set(_prev "")
      if(EXISTS "${_modeldir}/${_tmpl}")
        file(READ "${_modeldir}/${_tmpl}" _prev)
      endif()
      if(NOT _txt STREQUAL _prev)
        file(WRITE "${_modeldir}/${_tmpl}" "${_txt}")
      endif()
    endforeach()

    # Rewrite exasimpath in the pdeapp so text2code finds this Exasim install.
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

    get_filename_component(_pdeapp_dir "${EXT_PDEMODEL}" DIRECTORY)
    get_filename_component(_pdeapp_name "${EXT_PDEMODEL}" NAME)
    string(REGEX REPLACE "pdeapp([0-9]*)\.txt$" "pdemodel\\1.txt"
                         _pdemodel_name "${_pdeapp_name}")
    if(EXISTS "${_pdeapp_dir}/${_pdemodel_name}")
      file(COPY "${_pdeapp_dir}/${_pdemodel_name}" DESTINATION "${_modeldir}")
    endif()

    set(_stamp "${_modeldir}/.text2code.stamp")
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

  elseif(EXT_KERNELS_DIRS)
    # ---- pre-generated kernels (KERNELS/KERNELS_DIRS) path ----------------
    list(LENGTH EXT_KERNELS_DIRS _nk)
    if(NOT _nk EQUAL _nids)
      message(FATAL_ERROR
        "exasim_add_external_builtin_model(${_tgt}): IDS (${_nids}) and "
        "KERNELS_DIRS (${_nk}) must have the same length")
    endif()
    math(EXPR _last "${_nids} - 1")
    foreach(_i RANGE ${_last})
      list(GET EXT_IDS ${_i} _id)
      list(GET EXT_KERNELS_DIRS ${_i} _kdir)
      if(NOT IS_DIRECTORY "${_kdir}")
        message(FATAL_ERROR
          "exasim_add_external_builtin_model(${_tgt}): KERNELS directory not found:\n"
          "  ${_kdir}")
      endif()
      set(_modeldir "${_gendir}/model${_id}")
      file(MAKE_DIRECTORY "${_modeldir}")
      foreach(_tmpl model.hpp model.cpp)
        file(READ "${Exasim_BUILTIN_DIR}/${_tmpl}" _txt)
        string(REPLACE "exasim_model_1" "exasim_model_${_id}" _txt "${_txt}")
        string(REPLACE "\"../dstype.hpp\""
                       "\"${Exasim_BUILTIN_DIR}/dstype.hpp\"" _txt "${_txt}")
        set(_prev "")
        if(EXISTS "${_modeldir}/${_tmpl}")
          file(READ "${_modeldir}/${_tmpl}" _prev)
        endif()
        if(NOT _txt STREQUAL _prev)
          file(WRITE "${_modeldir}/${_tmpl}" "${_txt}")
        endif()
      endforeach()
      # With several models the kernel dirs cannot all sit on one include path
      # (model.cpp's quoted "KokkosFlux.cpp" would resolve ambiguously). Copy
      # each model's kernels into its own model<ID>/ so the quoted includes
      # resolve to the directory of the including file. A single model keeps
      # the historical include-path resolution (see include dirs below) for a
      # byte-identical build.
      if(_nids GREATER 1)
        file(GLOB _kfiles "${_kdir}/*")
        foreach(_kf IN LISTS _kfiles)
          if(NOT IS_DIRECTORY "${_kf}")
            get_filename_component(_kfn "${_kf}" NAME)
            configure_file("${_kf}" "${_modeldir}/${_kfn}" COPYONLY)
          endif()
        endforeach()
      endif()
    endforeach()
    add_custom_target(_exasim_ext_codegen_${_tgt})

  else()
    # ---- hand-written (SOURCES) path — single model only ------------------
    if(NOT _nids EQUAL 1)
      message(FATAL_ERROR
        "exasim_add_external_builtin_model: SOURCES supports a single ID")
    endif()
    set(_id "${EXT_IDS}")
    set(_modeldir "${_gendir}/model${_id}")
    file(MAKE_DIRECTORY "${_modeldir}")
    foreach(_src ${EXT_SOURCES})
      get_filename_component(_sname "${_src}" NAME)
      configure_file("${_src}" "${_modeldir}/${_sname}" COPYONLY)
    endforeach()
    add_custom_target(_exasim_ext_codegen_${_tgt})
  endif()

  # Select the installed built-in model library to link for fallthrough dispatch.
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

  # Build the provider library. getBuiltInLibraryExasimDriverABI() is defined in
  # ExternalModelProvider.cpp and intercepts every ID in EXASIM_EXT_MODEL_IDS;
  # all other IDs fall through to the builtin dispatchers in ${_bm_lib}. Link
  # order: ext library first so its symbol definition wins over any duplicate.
  if(EXT_SHARED)
    add_library(${_tgt} SHARED "${_gendir}/ExternalModelProvider.cpp")
    set_target_properties(${_tgt} PROPERTIES
      BUILD_WITH_INSTALL_NAME_DIR TRUE
      INSTALL_NAME_DIR "@rpath")
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
  # For a single model, model.cpp's quoted kernel includes fall back to this
  # search path (no per-model copy is made). For several models the per-model
  # copies above take precedence, and these entries are a harmless fallback.
  foreach(_kdir IN LISTS EXT_KERNELS_DIRS)
    target_include_directories(${_tgt} PRIVATE "${_kdir}")
  endforeach()
  target_link_libraries(${_tgt} PUBLIC  Exasim::headers)
  set_target_properties(${_tgt} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()
