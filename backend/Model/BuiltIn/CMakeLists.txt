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
# Published for tests that compare regenerated kernels against checked-in ones
# (see tests/builtin-models/).
set_property(GLOBAL PROPERTY EXASIM_BUILTIN_GENROOT "${EXASIM_BUILTIN_GENROOT}")

# text2code binary used to (re)generate model kernels from a pdemodel source.
# The superbuild forwards its out-of-tree location via EXASIM_TEXT2CODE.
if(NOT DEFINED EXASIM_TEXT2CODE)
  find_program(EXASIM_TEXT2CODE text2code)
endif()

# Accumulators (parent scope so exasim_add_builtin_model can append from anywhere)
set_property(GLOBAL PROPERTY EXASIM_BM_IDS "")
set_property(GLOBAL PROPERTY EXASIM_BM_DIRS "")
set_property(GLOBAL PROPERTY EXASIM_BM_GENFILES "")
set_property(GLOBAL PROPERTY EXASIM_BM_HAS_SIZES "")
# set_property(GLOBAL PROPERTY EXASIM_BM_SIZES "")   # no longer used

# -----------------------------------------------------------------------------
# exasim_add_builtin_model(<id> [SOURCE <pdemodel.txt>] [MODEL_DIR <dir>])
#   SOURCE     : text2code source; kernels + namespaced wrapper are regenerated
#                into the build tree.
#   MODEL_DIR  : for legacy models without a source, the checked-in modelN dir.
# -----------------------------------------------------------------------------
function(exasim_add_builtin_model ID)
  cmake_parse_arguments(BM "" "SOURCE;MODEL_DIR" "" ${ARGN})

  # Model sizes are provided by text2code-generated model_sizes.hpp.
  if(BM_SOURCE AND NOT EXISTS "${EXASIM_TEXT2CODE}")
    message(STATUS "exasim_add_builtin_model(${ID}): text2code not found at "
                   "${EXASIM_TEXT2CODE}; using checked-in kernels in model${ID}.")
    set(BM_SOURCE "")
  endif()

  if(BM_SOURCE)
    set(_gendir "${EXASIM_BUILTIN_GENROOT}/model${ID}")
    file(MAKE_DIRECTORY "${_gendir}")
    # Remove stale model_sizes.hpp and any leftover stamp so regeneration
    # happens on the next build (configure removes the outputs that the stamp
    # refers to).
    file(REMOVE "${_gendir}/model_sizes.hpp" "${_gendir}/.text2code.stamp")

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
    set(_has_sizes 1)
  elseif(BM_MODEL_DIR)
    set(_modeldir "${BM_MODEL_DIR}")
    if(EXISTS "${_modeldir}/model_sizes.hpp")
      set(_has_sizes 1)
    else()
      set(_has_sizes 0)
    endif()
  else()
    set(_modeldir "${EXASIM_BUILTIN_DIR}/model${ID}")
    if(EXISTS "${_modeldir}/model_sizes.hpp")
      set(_has_sizes 1)
    else()
      set(_has_sizes 0)
    endif()
  endif()

  set_property(GLOBAL APPEND PROPERTY EXASIM_BM_IDS  "${ID}")
  set_property(GLOBAL APPEND PROPERTY EXASIM_BM_DIRS "${_modeldir}")
  set_property(GLOBAL APPEND PROPERTY EXASIM_BM_HAS_SIZES "${_has_sizes}")
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

  # --- generate the model size lookup table (.cpp, linked into the library) ---
  # Text2Code-generated models provide model_sizes.hpp. Legacy checked-in
  # fallback kernels do not, so return an empty ModelSizes record for them and
  # let app/runtime metadata remain authoritative.
  get_property(_ids GLOBAL PROPERTY EXASIM_BM_IDS)
  get_property(_dirs GLOBAL PROPERTY EXASIM_BM_DIRS)
  get_property(_has_sizes GLOBAL PROPERTY EXASIM_BM_HAS_SIZES)
  set(_sizes_file "${EXASIM_BUILTIN_GENROOT}/builtin_model_sizes.cpp")
  file(WRITE "${_sizes_file}"
    "// Auto-generated by CMake from text2code model_sizes.hpp headers when available.\n"
    "#include \"driver_abi.hpp\"\n")
  list(LENGTH _ids _n_sizes)
  math(EXPR _last_sizes "${_n_sizes} - 1")
  foreach(_i RANGE ${_last_sizes})
    list(GET _ids ${_i} _id)
    list(GET _dirs ${_i} _dir)
    list(GET _has_sizes ${_i} _sizes_available)
    if(_sizes_available)
      file(APPEND "${_sizes_file}"
        "namespace builtin_model_${_id} {\n"
        "#include \"${_dir}/model_sizes.hpp\"\n"
        "}\n")
    endif()
  endforeach()
  file(APPEND "${_sizes_file}"
    "\n"
    "ModelSizes builtinGetModelSizes(int modelnumber) {\n"
    "    switch (modelnumber) {\n")
  foreach(_i RANGE ${_last_sizes})
    list(GET _ids ${_i} _id)
    list(GET _has_sizes ${_i} _sizes_available)
    if(_sizes_available)
      file(APPEND "${_sizes_file}"
        "        case ${_id}: return {builtin_model_${_id}::ncu, builtin_model_${_id}::nco, builtin_model_${_id}::ncw, builtin_model_${_id}::nsca, builtin_model_${_id}::nvec, builtin_model_${_id}::nten, builtin_model_${_id}::nsurf, builtin_model_${_id}::nvqoi, builtin_model_${_id}::nmaterialstate};\n")
    else()
      file(APPEND "${_sizes_file}"
        "        case ${_id}: return {0, 0, 0, 0, 0, 0, 0, 0, 0};\n")
    endif()
  endforeach()
  file(APPEND "${_sizes_file}"
    "        default: return {0, 0, 0, 0, 0, 0, 0, 0, 0};\n"
    "    }\n"
    "}\n")

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
    set(_bm_kokkos "${EXASIM_DIR}/deps/kokkos/buildcuda")
  elseif(EXASIM_HIP)
    set(_bm_name builtinmodelhip)
    set(_bm_kokkos "${EXASIM_DIR}/deps/kokkos/buildhip")
  else()
    set(_bm_name builtinmodelserial)
    set(_bm_kokkos "${EXASIM_DIR}/deps/kokkos/buildserial")
  endif()
  # Pull in the matching Kokkos as a usage requirement. For a STATIC library this
  # applies Kokkos's *compile* flags — include dirs, the GPU arch, and (crucially
  # for CUDA/HIP) --expt-extended-lambda / relaxed-constexpr for the device
  # lambdas in the kernels — without archiving Kokkos into the .a (so still no
  # duplicate Kokkos globals). CMake propagates the Kokkos dependency to the
  # consumer, which links the very same Kokkos: one copy overall. find_package
  # here because the parent build only locates Kokkos after this subdirectory.
  # Prefer the Kokkos_DIR the superbuild forwards (it builds Kokkos out of the
  # source tree); fall back to the in-source vendored location for a standalone
  # `cmake -S install` build.
  if(NOT TARGET Kokkos::kokkos)
    if(Kokkos_DIR)
      find_package(Kokkos REQUIRED)
    else()
      find_package(Kokkos REQUIRED PATHS "${_bm_kokkos}" NO_DEFAULT_PATH)
    endif()
  endif()
  set(_bm_sizes_cpp "${EXASIM_BUILTIN_GENROOT}/builtin_model_sizes.cpp")
  add_library(${_bm_name} STATIC
    "${EXASIM_BUILTIN_DIR}/libbuiltinmodel.cpp"
    "${_bm_sizes_cpp}")
  add_dependencies(${_bm_name} exasim_builtin_model_codegen)
  target_include_directories(${_bm_name} PRIVATE
    "${EXASIM_BUILTIN_INCDIR}" "${EXASIM_BUILTIN_DIR}" "${EXASIM_DIR}/include")
  target_link_libraries(${_bm_name} PRIVATE Kokkos::kokkos)
  set_target_properties(${_bm_name} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${EXASIM_LIB_DIR}"
    POSITION_INDEPENDENT_CODE ON)
endfunction()

# -----------------------------------------------------------------------------
# Register the default model set. The library used to build all 15 built-in
# models by default, which made ordinary builds pay for application models that
# the test suite does not exercise. Keep the tested minimum as the default and
# let release/application builds opt into additional models through the cache.
# Skipped if the caller opts to register models itself.
# -----------------------------------------------------------------------------
set(EXASIM_BUILTIN_MODEL_IDS "1;4" CACHE STRING
    "Built-in model IDs to compile into libbuiltinmodel, or ALL for models 1-15.")

function(exasim_add_default_builtin_model ID)
  if(EXISTS "${EXASIM_BUILTIN_DIR}/pdeapp${ID}.txt")
    exasim_add_builtin_model(${ID} SOURCE "${EXASIM_BUILTIN_DIR}/pdeapp${ID}.txt")
  else()
    exasim_add_builtin_model(${ID})
  endif()
endfunction()

if(NOT DEFINED EXASIM_BUILTIN_SKIP_DEFAULTS)
  if(EXASIM_BUILTIN_MODEL_IDS STREQUAL "ALL")
    set(_exasim_builtin_model_ids 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
  else()
    set(_exasim_builtin_model_ids ${EXASIM_BUILTIN_MODEL_IDS})
  endif()

  if(NOT _exasim_builtin_model_ids)
    message(FATAL_ERROR "EXASIM_BUILTIN_MODEL_IDS must not be empty.")
  endif()

  foreach(_id IN LISTS _exasim_builtin_model_ids)
    if(NOT _id MATCHES "^[0-9]+$")
      message(FATAL_ERROR
        "Invalid EXASIM_BUILTIN_MODEL_IDS entry '${_id}'. Use numeric IDs or ALL.")
    endif()
    if(_id LESS 1 OR _id GREATER 15)
      message(FATAL_ERROR
        "Invalid built-in model ID ${_id}; supported default model IDs are 1 through 15.")
    endif()
    exasim_add_default_builtin_model(${_id})
  endforeach()

  message(STATUS "EXASIM_BUILTIN_MODEL_IDS = ${_exasim_builtin_model_ids}")
  exasim_finalize_builtin_models()
endif()

# -----------------------------------------------------------------------------
# Optional: also emit standalone header-only, C++-driven apps for the built-in
# models via `text2code --emit-app` (the same text2code already used to
# regenerate kernels above). Off by default; build the aggregate `builtin-apps`
# target to produce them. No-op when text2code is unavailable. Demonstrates the
# emit-app path over the whole built-in corpus and gives a runnable app per model.
# -----------------------------------------------------------------------------
option(EXASIM_EMIT_BUILTIN_APPS
       "Emit standalone header-only apps for the built-in models (needs text2code)" OFF)
if(EXASIM_EMIT_BUILTIN_APPS)
  include("${EXASIM_DIR}/cmake/ExasimEmitApp.cmake")
  set(_bapps "")
  foreach(_id RANGE 1 15)
    set(_src "${EXASIM_BUILTIN_DIR}/pdeapp${_id}.txt")
    if(EXISTS "${_src}")
      set(_adir "${EXASIM_BUILTIN_GENROOT}/apps/model${_id}")
      file(MAKE_DIRECTORY "${_adir}")
      # Rewrite exasimpath to this tree + force gendatain=0 (the emitted app reads
      # datain/ at run time; codegen here needs no mesh), mirroring the kernel path.
      file(READ "${_src}" _pde)
      string(REGEX REPLACE "exasimpath[ \t]*=[ \t]*\"[^\"]*\""
             "exasimpath = \"${EXASIM_DIR}\"" _pde "${_pde}")
      if(NOT _pde MATCHES "exasimpath[ \t]*=")
        set(_pde "exasimpath = \"${EXASIM_DIR}\";\n${_pde}")
      endif()
      string(REGEX REPLACE "gendatain[ \t]*=[ \t]*[0-9]+" "gendatain = 0" _pde "${_pde}")
      if(NOT _pde MATCHES "gendatain[ \t]*=")
        set(_pde "gendatain = 0;\n${_pde}")
      endif()
      file(WRITE "${_adir}/pdeapp.txt" "${_pde}")
      file(COPY "${EXASIM_BUILTIN_DIR}/pdemodel${_id}.txt" DESTINATION "${_adir}")
      exasim_emit_app(NAME builtin_app_${_id}
                      PDEAPP "${_adir}/pdeapp.txt"
                      DEST   "${_adir}/app"
                      MODEL_ID ${_id} APP_NAME builtin${_id}
                      TEXT2CODE "${EXASIM_TEXT2CODE}")
      if(TARGET builtin_app_${_id})
        list(APPEND _bapps builtin_app_${_id})
      endif()
    endif()
  endforeach()
  if(_bapps)
    add_custom_target(builtin-apps DEPENDS ${_bapps})
    message(STATUS "EXASIM_EMIT_BUILTIN_APPS: 'builtin-apps' target emits ${_bapps}")
  endif()
endif()
