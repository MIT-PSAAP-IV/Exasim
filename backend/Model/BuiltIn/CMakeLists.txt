# =============================================================================
#  Built-in model library (libbuiltinmodel{serial,cuda,hip})
#
#  Replaces the standalone Makefile and the MATLAB genmodel*/editlibbuiltinmodel
#  scripts. Models are registered with exasim_add_builtin_model(); those with a
#  text2code SOURCE are regenerated at build time, legacy models reuse their
#  checked-in kernels. The dispatcher (libbuiltinmodel.cpp) consumes a generated
#  registry header, so adding a model never edits hand-written C++.
#
#  External consumers can pull in this directory (add_subdirectory) and call
#  exasim_add_builtin_model(<id> SOURCE <pdemodel.txt>) to inject their own model.
# =============================================================================
cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED EXASIM_DIR)
  get_filename_component(EXASIM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
endif()
set(EXASIM_BUILTIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(EXASIM_BUILTIN_GENROOT "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(EXASIM_BUILTIN_INCDIR  "${EXASIM_BUILTIN_GENROOT}/include")
file(MAKE_DIRECTORY "${EXASIM_BUILTIN_INCDIR}")

# text2code binary used to (re)generate model kernels from a pdemodel source.
if(NOT DEFINED EXASIM_TEXT2CODE)
  set(EXASIM_TEXT2CODE "${EXASIM_DIR}/build/text2code")
endif()

# Accumulators (parent scope so exasim_add_builtin_model can append from anywhere)
set_property(GLOBAL PROPERTY EXASIM_BM_IDS "")
set_property(GLOBAL PROPERTY EXASIM_BM_DIRS "")
set_property(GLOBAL PROPERTY EXASIM_BM_GENFILES "")

# -----------------------------------------------------------------------------
# exasim_add_builtin_model(<id> [SOURCE <pdemodel.txt>] [MODEL_DIR <dir>])
#   SOURCE     : text2code source; kernels + namespaced wrapper are regenerated
#                into the build tree.
#   MODEL_DIR  : for legacy models without a source, the checked-in modelN dir.
# -----------------------------------------------------------------------------
function(exasim_add_builtin_model ID)
  cmake_parse_arguments(BM "" "SOURCE;MODEL_DIR" "" ${ARGN})

  if(BM_SOURCE AND NOT EXISTS "${EXASIM_TEXT2CODE}")
    message(STATUS "exasim_add_builtin_model(${ID}): text2code not found at "
                   "${EXASIM_TEXT2CODE}; using checked-in kernels in model${ID}.")
    set(BM_SOURCE "")
  endif()

  if(BM_SOURCE)
    set(_gendir "${EXASIM_BUILTIN_GENROOT}/model${ID}")
    file(MAKE_DIRECTORY "${_gendir}")

    # Instantiate the namespaced wrapper (model.hpp/.cpp) from the template,
    # renaming exasim_model_1 -> exasim_model_<ID> and fixing the dstype include.
    # (This is the editmodelhppcpp.m step, done in pure CMake.)
    foreach(_f model.hpp model.cpp)
      file(READ "${EXASIM_BUILTIN_DIR}/${_f}" _txt)
      string(REPLACE "exasim_model_1" "exasim_model_${ID}" _txt "${_txt}")
      string(REPLACE "\"../dstype.hpp\"" "\"${EXASIM_BUILTIN_DIR}/dstype.hpp\"" _txt "${_txt}")
      file(WRITE "${_gendir}/${_f}" "${_txt}")
    endforeach()

    # Regenerate the kernels from the text2code source at build time.
    # text2code emits the Kokkos*/Hdg*/cpu* kernel files that model.cpp includes.
    # SOURCE is the pdeapp file, which references the model file (pdemodel*.txt).
    set(_dep_model "")
    string(REGEX REPLACE "pdeapp([0-9]+)\\.txt$" "pdemodel\\1.txt" _dep_model "${BM_SOURCE}")

    # The checked-in pdeapp sources hardcode exasimpath to the original author's
    # machine; text2code uses it to locate SymEngine/Kokkos. Rewrite it to this
    # tree so regeneration is portable. (Commented lines stay commented.)
    file(READ "${BM_SOURCE}" _pde)
    string(REGEX REPLACE "exasimpath[ \t]*=[ \t]*\"[^\"]*\""
           "exasimpath = \"${EXASIM_DIR}\"" _pde "${_pde}")
    # Some checked-in pdeapp sources (e.g. models 12-15) omit exasimpath entirely;
    # the replace above is then a no-op. Inject it so text2code can locate this tree.
    if(NOT _pde MATCHES "exasimpath[ \t]*=")
      set(_pde "exasimpath = \"${EXASIM_DIR}\";\n${_pde}")
    endif()
    set(_pdeapp "${_gendir}/pdeapp.txt")
    file(WRITE "${_pdeapp}" "${_pde}")
    # text2code resolves modelfile relative to the pdeapp's directory, so place
    # the referenced model file next to the rewritten pdeapp.
    if(_dep_model AND EXISTS "${_dep_model}")
      file(COPY "${_dep_model}" DESTINATION "${_gendir}")
    endif()

    set(_stamp "${_gendir}/.text2code.stamp")
    add_custom_command(
      OUTPUT "${_stamp}"
      COMMAND "${EXASIM_TEXT2CODE}" "${_pdeapp}" --out-dir "${_gendir}" --gen-only
      COMMAND ${CMAKE_COMMAND} -E touch "${_stamp}"
      DEPENDS "${BM_SOURCE}" "$<$<BOOL:${_dep_model}>:${_dep_model}>"
      WORKING_DIRECTORY "${EXASIM_BUILTIN_DIR}"
      COMMENT "text2code: regenerating built-in model ${ID} from ${BM_SOURCE}"
      VERBATIM)
    set_property(GLOBAL APPEND PROPERTY EXASIM_BM_GENFILES "${_stamp}")
    set(_modeldir "${_gendir}")
  elseif(BM_MODEL_DIR)
    set(_modeldir "${BM_MODEL_DIR}")
  else()
    set(_modeldir "${EXASIM_BUILTIN_DIR}/model${ID}")
  endif()

  set_property(GLOBAL APPEND PROPERTY EXASIM_BM_IDS  "${ID}")
  set_property(GLOBAL APPEND PROPERTY EXASIM_BM_DIRS "${_modeldir}")
endfunction()

# -----------------------------------------------------------------------------
# exasim_finalize_builtin_models()
#   Generate builtin_model_registry.h from the registered models and define the
#   library targets.
# -----------------------------------------------------------------------------
function(exasim_finalize_builtin_models)
  get_property(_ids  GLOBAL PROPERTY EXASIM_BM_IDS)
  get_property(_dirs GLOBAL PROPERTY EXASIM_BM_DIRS)
  get_property(_gen  GLOBAL PROPERTY EXASIM_BM_GENFILES)

  # --- generate the registry header (absolute include paths) ---
  set(_hdr_inc "")
  set(_impl_inc "")
  set(_macro "#define EXASIM_BUILTIN_MODELS(X)")
  list(LENGTH _ids _n)
  math(EXPR _last "${_n} - 1")
  foreach(_i RANGE ${_last})
    list(GET _ids  ${_i} _id)
    list(GET _dirs ${_i} _dir)
    string(APPEND _hdr_inc  "#include \"${_dir}/model.hpp\"\n")
    string(APPEND _impl_inc "#include \"${_dir}/model.cpp\"\n")
    string(APPEND _macro " X(${_id})")
  endforeach()
  set(EXASIM_BM_HEADER_INCLUDES "${_hdr_inc}")
  set(EXASIM_BM_IMPL_INCLUDES "${_impl_inc}")
  set(EXASIM_BM_MACRO "${_macro}")
  configure_file(
    "${EXASIM_BUILTIN_DIR}/builtin_model_registry.h.in"
    "${EXASIM_BUILTIN_INCDIR}/builtin_model_registry.h"
    @ONLY)

  # force regeneration of kernels before compiling the dispatcher
  add_custom_target(exasim_builtin_model_codegen DEPENDS ${_gen})

  # --- define the library variant matching the active backend ---
  # The inner build uses a single CXX compiler (g++ for CPU, nvcc_wrapper for
  # CUDA, hipcc for HIP), so the built-in model library is compiled for whichever
  # backend this build targets — exactly like the cpu*/gpu* EXASIM executables.
  # The matching Kokkos build dir supplies the headers (and selects the default
  # execution space: Serial / Cuda / HIP).
  #
  # STATIC, and deliberately *not* linking libkokkoscore.a: a SHARED model lib
  # that statically embeds its own Kokkos produces a second copy of Kokkos's
  # global state (SharedAllocationRecord list, the default spaces, the atexit
  # finalizers) alongside the copy in the consumer executable. On glibc's flat
  # symbol namespace both finalizers run at _dl_fini and double-free the shared
  # singleton (a latent bug macOS's two-level namespace masked). As a static
  # archive linked into the consumer — which already links exactly one Kokkos —
  # there is a single copy of Kokkos and a single finalizer. Only the headers
  # are needed to compile the kernels.
  if(EXASIM_CUDA)
    set(_bm_name builtinmodelcuda)
    set(_bm_kokkos "${EXASIM_DIR}/kokkos/buildcuda")
  elseif(EXASIM_HIP)
    set(_bm_name builtinmodelhip)
    set(_bm_kokkos "${EXASIM_DIR}/kokkos/buildhip")
  else()
    set(_bm_name builtinmodelserial)
    set(_bm_kokkos "${EXASIM_DIR}/kokkos/buildserial")
  endif()
  add_library(${_bm_name} STATIC "${EXASIM_BUILTIN_DIR}/libbuiltinmodel.cpp")
  add_dependencies(${_bm_name} exasim_builtin_model_codegen)
  target_include_directories(${_bm_name} PRIVATE
    "${EXASIM_BUILTIN_INCDIR}" "${EXASIM_BUILTIN_DIR}" "${_bm_kokkos}/include")
  set_target_properties(${_bm_name} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${EXASIM_DIR}/lib"
    POSITION_INDEPENDENT_CODE ON)
endfunction()

# -----------------------------------------------------------------------------
# Register the default model set: legacy 1-6 (checked-in kernels), 7-15 from
# their text2code sources. Skipped if the caller opts to register models itself.
# -----------------------------------------------------------------------------
if(NOT DEFINED EXASIM_BUILTIN_SKIP_DEFAULTS)
  foreach(_id RANGE 1 6)
    exasim_add_builtin_model(${_id})
  endforeach()
  foreach(_id RANGE 7 15)
    if(EXISTS "${EXASIM_BUILTIN_DIR}/pdeapp${_id}.txt")
      exasim_add_builtin_model(${_id} SOURCE "${EXASIM_BUILTIN_DIR}/pdeapp${_id}.txt")
    else()
      exasim_add_builtin_model(${_id})
    endif()
  endforeach()
  exasim_finalize_builtin_models()
endif()
