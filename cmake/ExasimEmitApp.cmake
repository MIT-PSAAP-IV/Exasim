# =============================================================================
#  ExasimEmitApp.cmake — generate a standalone, header-only, C++-driven app from a
#  text2code model, via the installed `text2code --emit-app`.
#
#  The app builds CSolution<PdeModel> from datain/ and solves through Exasim's
#  exported PETSc operator (exasim::petsc::solve_steady) — no runtime-loaded .so
#  model ABI and no hand-rolled PETSc glue. It is the C++-driven form of a
#  text2code-generated model, complementary to the frontends' `exportapp` (which
#  ships the generated kernel .cpp set + the ExasimSolver driver instead).
#
#  Requires a `text2code` binary: pass TEXT2CODE, set EXASIM_TEXT2CODE, or rely on
#  find_program on PATH. When none is found the target is skipped with a STATUS
#  message (so a configure never fails just because text2code is not built yet).
#
#  Usage:
#    include(ExasimEmitApp)
#    exasim_emit_app(NAME poisson2d_app
#                    PDEAPP ${CMAKE_CURRENT_SOURCE_DIR}/pdeapp.txt
#                    DEST   ${CMAKE_CURRENT_BINARY_DIR}/apps/poisson2d
#                    MODEL_ID 8 [APP_NAME poisson2d] [TEXT2CODE /path/to/text2code] [ALL])
# =============================================================================

function(exasim_emit_app)
  cmake_parse_arguments(EA "ALL" "NAME;PDEAPP;DEST;MODEL_ID;APP_NAME;TEXT2CODE" "" ${ARGN})
  if(NOT EA_NAME OR NOT EA_PDEAPP OR NOT EA_DEST)
    message(FATAL_ERROR "exasim_emit_app: NAME, PDEAPP and DEST are required.")
  endif()
  if(NOT EA_MODEL_ID)
    set(EA_MODEL_ID 100)
  endif()
  if(NOT EA_APP_NAME)
    get_filename_component(EA_APP_NAME "${EA_DEST}" NAME)
  endif()

  # Resolve a text2code binary: explicit arg -> cache/var -> PATH.
  set(_t2c "${EA_TEXT2CODE}")
  if(NOT _t2c AND DEFINED EXASIM_TEXT2CODE)
    set(_t2c "${EXASIM_TEXT2CODE}")
  endif()
  if(NOT _t2c)
    find_program(_t2c text2code)
  endif()
  if(NOT _t2c OR NOT EXISTS "${_t2c}")
    message(STATUS "exasim_emit_app(${EA_NAME}): no text2code found; skipping "
                   "(build/install text2code, or pass TEXT2CODE=<path>).")
    return()
  endif()

  set(_stamp "${EA_DEST}/.emit-app.stamp")
  file(MAKE_DIRECTORY "${EA_DEST}")
  # Run in the pdeapp's directory so text2code resolves `modelfile` (the referenced
  # pdemodel*.txt) relative to it -- the default datapath is the working directory.
  get_filename_component(_pdeapp_dir "${EA_PDEAPP}" DIRECTORY)
  add_custom_command(
    OUTPUT "${_stamp}"
    COMMAND "${_t2c}" "${EA_PDEAPP}" --emit-app "${EA_DEST}"
            --app-name "${EA_APP_NAME}" --model-id "${EA_MODEL_ID}"
    COMMAND ${CMAKE_COMMAND} -E touch "${_stamp}"
    DEPENDS "${EA_PDEAPP}"
    WORKING_DIRECTORY "${_pdeapp_dir}"
    COMMENT "text2code --emit-app: standalone header-only app '${EA_APP_NAME}' -> ${EA_DEST}"
    VERBATIM)

  if(EA_ALL)
    add_custom_target(${EA_NAME} ALL DEPENDS "${_stamp}")
  else()
    add_custom_target(${EA_NAME} DEPENDS "${_stamp}")
  endif()
endfunction()
