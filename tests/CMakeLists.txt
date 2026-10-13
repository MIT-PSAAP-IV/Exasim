# Test registration (gated by EXASIM_BUILD_TESTS; wired from install/CMakeLists.txt).
#
# B1-B4: out-of-tree consumer tests. run-consumer-tests.sh installs this build to a
# temp prefix, then for each project under tests/consumers/:
#   B3  asserts the consumer CMakeLists does NOT reference the Exasim source tree;
#   B2  configures it ONLY via find_package(Exasim) and builds it;
#   B4  runs it (if it ships a pdeapp.txt) and checks the QoI gate.
# This catches export/packaging regressions a normal in-tree build would miss.
#
# Run everything with `bash tests/run-tests.sh` (configures with
# EXASIM_BUILD_TESTS=ON, builds, runs ctest).
cmake_minimum_required(VERSION 3.16)


# Backend material-database appstruct conversion.  This is intentionally scoped
# to the appstruct-based reader used by the backend initialization path; PSAAP
# does not build the older standalone backend/Materials module.
add_executable(appstruct_materialdatabase_test
  "${EXASIM_DIR}/tests/materials/appstruct_materialdatabase_test.cpp")
target_include_directories(appstruct_materialdatabase_test PRIVATE
  "${EXASIM_DIR}")
target_compile_features(appstruct_materialdatabase_test PRIVATE cxx_std_17)
add_test(NAME appstruct_materialdatabase
  COMMAND appstruct_materialdatabase_test
    "${CMAKE_CURRENT_BINARY_DIR}")
set_tests_properties(appstruct_materialdatabase PROPERTIES TIMEOUT 60)

if(TARGET Kokkos::kokkos)
  add_executable(helmholtz_model_test
    "${EXASIM_DIR}/tests/unit/helmholtz_model.cpp")
  target_include_directories(helmholtz_model_test PRIVATE
    "${EXASIM_DIR}" "${EXASIM_DIR}/include")
  target_link_libraries(helmholtz_model_test PRIVATE
    ${EXASIM_HELMHOLTZ_MODEL_TARGET} Kokkos::kokkos)
  target_compile_features(helmholtz_model_test PRIVATE cxx_std_17)
  add_test(NAME helmholtz_model COMMAND helmholtz_model_test)
  set_tests_properties(helmholtz_model PROPERTIES TIMEOUT 60)

  add_executable(material_properties_nonuniform_test
    "${EXASIM_DIR}/tests/materials/material_properties_nonuniform_test.cpp")
  target_include_directories(material_properties_nonuniform_test PRIVATE
    "${EXASIM_DIR}")
  target_link_libraries(material_properties_nonuniform_test PRIVATE Kokkos::kokkos)
  target_compile_features(material_properties_nonuniform_test PRIVATE cxx_std_17)
  add_test(NAME material_properties_nonuniform
    COMMAND material_properties_nonuniform_test)
  set_tests_properties(material_properties_nonuniform PROPERTIES TIMEOUT 60)

  add_executable(wequation_material_split_test
    "${EXASIM_DIR}/tests/materials/wequation_material_split_test.cpp")
  target_include_directories(wequation_material_split_test PRIVATE
    "${EXASIM_INCLUDE_DIR}" "${EXASIM_DIR}")
  target_link_libraries(wequation_material_split_test PRIVATE
    Kokkos::kokkos ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
  target_compile_features(wequation_material_split_test PRIVATE cxx_std_17)
  add_test(NAME wequation_material_split
    COMMAND wequation_material_split_test)
  set_tests_properties(wequation_material_split PROPERTIES TIMEOUT 60)
endif()

# CPU-only consumer (non-MPI, single process).
add_test(NAME consumer_builtin_cpu
  COMMAND ${CMAKE_COMMAND} -E env
    "EXASIM_BUILD=${CMAKE_BINARY_DIR}"
    "KOKKOS_DIR=${Kokkos_DIR}"
    "EXASIM_ROOT=${EXASIM_DIR}"
    "CC=${CMAKE_C_COMPILER}"
    "CXX=${CMAKE_CXX_COMPILER}"
    "NP=1"
    "EXASIM_MPI_VARIANT=OFF"
    "INSTALL_PREFIX=/tmp/exasim_install_cpu"
    bash "${CMAKE_CURRENT_SOURCE_DIR}/run-consumer-tests.sh")
set_tests_properties(consumer_builtin_cpu PROPERTIES TIMEOUT 1200)

# CPU+MPI consumer (2 ranks).
add_test(NAME consumer_builtin_mpi
  COMMAND ${CMAKE_COMMAND} -E env
    "EXASIM_BUILD=${CMAKE_BINARY_DIR}"
    "KOKKOS_DIR=${Kokkos_DIR}"
    "EXASIM_ROOT=${EXASIM_DIR}"
    "CC=${CMAKE_C_COMPILER}"
    "CXX=${CMAKE_CXX_COMPILER}"
    "NP=2"
    "EXASIM_MPI_VARIANT=ON"
    "INSTALL_PREFIX=/tmp/exasim_install_mpi"
    bash "${CMAKE_CURRENT_SOURCE_DIR}/run-consumer-tests.sh")
set_tests_properties(consumer_builtin_mpi PROPERTIES TIMEOUT 1200)

# MMS analytical-solution smoke for the header-only emit-app / solve_steady path:
# a small 2-species coupled reaction-diffusion model whose exact (manufactured)
# solution is u0=sin(pi x)sin(pi y), u1=0.5 sin(pi x)sin(pi y). run.sh generates datain,
# builds a solve_steady verify harness via find_package(Exasim), runs it, and checks the
# field against the exact solution. PETSc-gated: exits 77 (clean SKIP) when text2code /
# PETSc / a complete install is unavailable, like the other petsc consumer tests.
add_test(NAME consumer_mms_reactdiff
  COMMAND ${CMAKE_COMMAND} -E env
    "EXASIM_INSTALL=${CMAKE_INSTALL_PREFIX}"
    "EXASIM_ROOT=${EXASIM_DIR}"
    "KOKKOS_DIR=${Kokkos_DIR}"
    "CXX=${CMAKE_CXX_COMPILER}"
    bash "${CMAKE_CURRENT_SOURCE_DIR}/mms-reactdiff/run.sh")
set_tests_properties(consumer_mms_reactdiff PROPERTIES TIMEOUT 1200 SKIP_RETURN_CODE 77)

# Language-frontend end-to-end tests (Poisson 2D via the installed packages).
# Each builds a small app against this build's install (CMAKE_INSTALL_PREFIX)
# through the external-model path and gates the QoI. The wrapper exits 77 —
# a clean ctest SKIP — when the interpreter or its symbolic stack is missing
# (e.g. julia/matlab on CI runners), so registration is unconditional.
find_package(Python3 COMPONENTS Interpreter QUIET)
foreach(_fe python julia matlab)
  add_test(NAME frontend_${_fe}
    COMMAND ${CMAKE_COMMAND} -E env
      "EXASIM_ROOT=${EXASIM_DIR}"
      "EXASIM_INSTALL=${CMAKE_INSTALL_PREFIX}"
      "FRONTEND=${_fe}"
      "PYTHON3=${Python3_EXECUTABLE}"
      bash "${EXASIM_DIR}/tests/frontends/run-frontend-test.sh")
  set_tests_properties(frontend_${_fe} PROPERTIES
    TIMEOUT 2400 SKIP_RETURN_CODE 77)

  # Data-transfer app export: the run also packages a relocatable bundle, which
  # must be pristine and build+run on its own (copied elsewhere) via run.sh.
  add_test(NAME frontend_${_fe}_export
    COMMAND ${CMAKE_COMMAND} -E env
      "EXASIM_ROOT=${EXASIM_DIR}"
      "EXASIM_INSTALL=${CMAKE_INSTALL_PREFIX}"
      "FRONTEND=${_fe}"
      "EXPORT_TEST=1"
      "PYTHON3=${Python3_EXECUTABLE}"
      bash "${EXASIM_DIR}/tests/frontends/run-frontend-test.sh")
  set_tests_properties(frontend_${_fe}_export PROPERTIES
    TIMEOUT 2400 SKIP_RETURN_CODE 77)
endforeach()

# [nonmatching] Phase 9: interfacetype field, validation and app.bin slot 34 in every
# frontend and in text2code, plus the MATLAB coupled verification CI subset (matching
# type-0/type-1 solve at 1+1 through the production MPI build, the backend matching
# bypass, and the unequal-h convergence subset). Each wrapper exits 77 (skip) when its
# interpreter, MATLAB, mpirun or the install is unavailable.
add_test(NAME frontend_python_interfacetype # [nonmatching]
  COMMAND ${Python3_EXECUTABLE} "${EXASIM_DIR}/tests/frontends/python/test_interfacetype.py") # [nonmatching]
set_tests_properties(frontend_python_interfacetype PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77) # [nonmatching]
add_test(NAME frontend_julia_interfacetype # [nonmatching]
  COMMAND bash "${EXASIM_DIR}/tests/frontends/julia/run-test-interfacetype.sh") # [nonmatching]
set_tests_properties(frontend_julia_interfacetype PROPERTIES TIMEOUT 900 SKIP_RETURN_CODE 77) # [nonmatching]
add_test(NAME frontend_matlab_interfacetype # [nonmatching]
  COMMAND ${CMAKE_COMMAND} -E env "EXASIM_ROOT=${EXASIM_DIR}" # [nonmatching]
          bash "${EXASIM_DIR}/tests/nonmatching/run-phase9-matlab.sh" serialization) # [nonmatching]
set_tests_properties(frontend_matlab_interfacetype PROPERTIES TIMEOUT 900 SKIP_RETURN_CODE 77) # [nonmatching]
add_test(NAME text2code_interfacetype # [nonmatching]
  COMMAND ${CMAKE_COMMAND} -E env "EXASIM_ROOT=${EXASIM_DIR}" "EXASIM_INSTALL=${CMAKE_INSTALL_PREFIX}" # [nonmatching]
          bash "${EXASIM_DIR}/tests/text2code/interfacetype-test.sh") # [nonmatching]
set_tests_properties(text2code_interfacetype PROPERTIES # [nonmatching]
  TIMEOUT 600 SKIP_RETURN_CODE 77 RESOURCE_LOCK exasim_text2code_install) # [nonmatching]
# [nonmatching] Phase 9 Part B: text2code initializeDMD, single-rank coupled branch. Direct
# unit test (no interface elements, ordinary interface elements, duplicate element IDs from
# two coupled faces) and an installed-text2code CLI case with an L-shaped coupled interface.
add_executable(text2code_initializedmd_test ${CMAKE_CURRENT_SOURCE_DIR}/text2code/initializedmd_test.cpp) # [nonmatching]
target_include_directories(text2code_initializedmd_test PRIVATE ${EXASIM_DIR}/text2code/text2code) # [nonmatching]
target_compile_features(text2code_initializedmd_test PRIVATE cxx_std_17) # [nonmatching] tinyexpr needs C++17
target_link_libraries(text2code_initializedmd_test PRIVATE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES}) # [nonmatching]
add_test(NAME text2code_initializedmd COMMAND text2code_initializedmd_test) # [nonmatching]
set_tests_properties(text2code_initializedmd PROPERTIES TIMEOUT 120) # [nonmatching]
add_test(NAME text2code_initializedmd_cli # [nonmatching]
  COMMAND ${CMAKE_COMMAND} -E env "EXASIM_ROOT=${EXASIM_DIR}" "EXASIM_INSTALL=${CMAKE_INSTALL_PREFIX}" # [nonmatching]
          bash "${EXASIM_DIR}/tests/text2code/initializedmd-cli-test.sh") # [nonmatching]
set_tests_properties(text2code_initializedmd_cli PROPERTIES # [nonmatching]
  TIMEOUT 600 SKIP_RETURN_CODE 77 RESOURCE_LOCK exasim_text2code_install) # [nonmatching]
add_test(NAME nonmatching_phase9_matlab_ci # [nonmatching]
  COMMAND ${CMAKE_COMMAND} -E env "EXASIM_ROOT=${EXASIM_DIR}" "EXASIM_INSTALL=${CMAKE_INSTALL_PREFIX}" # [nonmatching]
          bash "${EXASIM_DIR}/tests/nonmatching/run-phase9-matlab.sh" ci) # [nonmatching]
set_tests_properties(nonmatching_phase9_matlab_ci PROPERTIES TIMEOUT 3600 SKIP_RETURN_CODE 77) # [nonmatching]
# [nonmatching] Phase 9 B2: MATLAB exporttext2code round trip of the coupled non-matching
# example (export, both pdeapp files carry interfacetype = 1, bundle build with Text2Code
# code generation, text2code datain generation, coupled run, both donor directions,
# solution equal to the MATLAB run). 1+1 is the acceptance run; 2+2 covers the
# partitioned text2code data path.
foreach(k 1 2) # [nonmatching]
  add_test(NAME nonmatching_phase9_text2code_${k}p${k} # [nonmatching]
    COMMAND ${CMAKE_COMMAND} -E env "EXASIM_ROOT=${EXASIM_DIR}" "EXASIM_INSTALL=${CMAKE_INSTALL_PREFIX}" # [nonmatching]
            bash "${EXASIM_DIR}/tests/nonmatching/run-phase9-matlab.sh" text2code ${k}) # [nonmatching]
  set_tests_properties(nonmatching_phase9_text2code_${k}p${k} PROPERTIES # [nonmatching]
    TIMEOUT 2400 SKIP_RETURN_CODE 77 RESOURCE_LOCK exasim_text2code_install) # [nonmatching]
endforeach() # [nonmatching]
add_test(NAME nonmatching_phase10_matlab_ci # [nonmatching]
  COMMAND ${CMAKE_COMMAND} -E env "EXASIM_ROOT=${EXASIM_DIR}" "EXASIM_INSTALL=${CMAKE_INSTALL_PREFIX}" # [nonmatching]
          bash "${EXASIM_DIR}/tests/nonmatching/run-phase9-matlab.sh" phase10-ci) # [nonmatching]
set_tests_properties(nonmatching_phase10_matlab_ci PROPERTIES TIMEOUT 3600 SKIP_RETURN_CODE 77) # [nonmatching]
add_test(NAME nonmatching_phase10_partition # [nonmatching]
  COMMAND ${CMAKE_COMMAND} -E env "EXASIM_ROOT=${EXASIM_DIR}" "EXASIM_INSTALL=${CMAKE_INSTALL_PREFIX}" # [nonmatching]
          bash "${EXASIM_DIR}/tests/nonmatching/run-phase9-matlab.sh" phase10-partition) # [nonmatching]
set_tests_properties(nonmatching_phase10_partition PROPERTIES TIMEOUT 3600 SKIP_RETURN_CODE 77) # [nonmatching]
foreach(k 1 2) # [nonmatching]
  add_test(NAME nonmatching_phase10_text2code_${k}p${k} # [nonmatching]
    COMMAND ${CMAKE_COMMAND} -E env "EXASIM_ROOT=${EXASIM_DIR}" "EXASIM_INSTALL=${CMAKE_INSTALL_PREFIX}" # [nonmatching]
            bash "${EXASIM_DIR}/tests/nonmatching/run-phase9-matlab.sh" phase10-text2code ${k}) # [nonmatching]
  set_tests_properties(nonmatching_phase10_text2code_${k}p${k} PROPERTIES # [nonmatching]
    TIMEOUT 3600 SKIP_RETURN_CODE 77 RESOURCE_LOCK exasim_text2code_install) # [nonmatching]
endforeach() # [nonmatching]

# Cheap extra Poisson app-regression cases through the Python frontend -- bring app
# numerical coverage into CI beyond the default 2D HDG case. CASE=<name> selects
# pdeapp_<name>.py (+ pdemodel_<name>.py if present); each gates the analytic
# (u-uexact)^2 QoI so no golden baseline is needed. Python-only (SKIPs 77 otherwise).
#
# LDG (hybrid=0): exercises the LDG discretization end-to-end through the frontend --
# coverage nothing else in CI provides (CPU LDG app-regression + GPU LDG are
# local/remote only). Reuses pdemodel.py; same cheap 16x16 p3 mesh -> tight 1e-8 gate.
add_test(NAME frontend_python_ldg
  COMMAND ${CMAKE_COMMAND} -E env
    "EXASIM_ROOT=${EXASIM_DIR}"
    "EXASIM_INSTALL=${CMAKE_INSTALL_PREFIX}"
    "FRONTEND=python"
    "CASE=ldg"
    "PYTHON3=${Python3_EXECUTABLE}"
    bash "${EXASIM_DIR}/tests/frontends/run-frontend-test.sh")
set_tests_properties(frontend_python_ldg PROPERTIES
  TIMEOUT 2400 SKIP_RETURN_CODE 77)

# 3D coverage (the other frontend cases are 2D SquareMesh). Small 6^3 hex mesh at
# porder 3; the coarser 3D discretization leaves error^2 ~ 6e-7, so gate at 1e-6 (a
# real broken 3D solve gives O(1) error and still fails loudly).
add_test(NAME frontend_python_poisson3d
  COMMAND ${CMAKE_COMMAND} -E env
    "EXASIM_ROOT=${EXASIM_DIR}"
    "EXASIM_INSTALL=${CMAKE_INSTALL_PREFIX}"
    "FRONTEND=python"
    "CASE=poisson3d"
    "QOI_TOL=1e-6"
    "PYTHON3=${Python3_EXECUTABLE}"
    bash "${EXASIM_DIR}/tests/frontends/run-frontend-test.sh")
set_tests_properties(frontend_python_poisson3d PROPERTIES
  TIMEOUT 2400 SKIP_RETURN_CODE 77)

# Coexistence: build + run two independent generated models from ONE working
# directory. Asserts per-model isolation (distinct .exasim build trees, datain
# and dataout subdirs, modelid 100 vs 101) so they no longer clobber. Python
# only for now; the wrapper SKIPs (77) for other frontends.
add_test(NAME frontend_coexistence
  COMMAND ${CMAKE_COMMAND} -E env
    "EXASIM_ROOT=${EXASIM_DIR}"
    "EXASIM_INSTALL=${CMAKE_INSTALL_PREFIX}"
    "FRONTEND=python"
    "COEXIST_TEST=1"
    "PYTHON3=${Python3_EXECUTABLE}"
    bash "${EXASIM_DIR}/tests/frontends/run-frontend-test.sh")
set_tests_properties(frontend_coexistence PROPERTIES
  TIMEOUT 2400 SKIP_RETURN_CODE 77)

# Combined multi-PDE: build two generated models into ONE exasimapp and run
# them together via the solver's multi-model CLI (one datain/dataout pair per
# model). Asserts the single provider dispatches both ids (100, 101) and both
# output sets are produced. Python only for now; SKIPs (77) for others.
add_test(NAME frontend_combined_multimodel
  COMMAND ${CMAKE_COMMAND} -E env
    "EXASIM_ROOT=${EXASIM_DIR}"
    "EXASIM_INSTALL=${CMAKE_INSTALL_PREFIX}"
    "FRONTEND=python"
    "COMBINED_TEST=1"
    "PYTHON3=${Python3_EXECUTABLE}"
    bash "${EXASIM_DIR}/tests/frontends/run-frontend-test.sh")
set_tests_properties(frontend_combined_multimodel PROPERTIES
  TIMEOUT 2400 SKIP_RETURN_CODE 77)

# Combined multi-PDE export: the combined run also packages a relocatable
# data-transfer bundle (per-model datain<n>/kernels<n>, id list, N-pair run.sh)
# which must be pristine and build+run on its own elsewhere. Python only.
add_test(NAME frontend_combined_export
  COMMAND ${CMAKE_COMMAND} -E env
    "EXASIM_ROOT=${EXASIM_DIR}"
    "EXASIM_INSTALL=${CMAKE_INSTALL_PREFIX}"
    "FRONTEND=python"
    "COMBINED_EXPORT=1"
    "PYTHON3=${Python3_EXECUTABLE}"
    bash "${EXASIM_DIR}/tests/frontends/run-frontend-test.sh")
set_tests_properties(frontend_combined_export PROPERTIES
  TIMEOUT 2400 SKIP_RETURN_CODE 77)

# Explicit standalone postprocess mode: solve a small frontend-generated
# Poisson case, rerun the generated executable in postprocess mode on a copied
# output tree, and assert the postprocessed QoI/VTK/solution files match.
add_test(NAME frontend_python_postprocess
  COMMAND ${CMAKE_COMMAND} -E env
    "EXASIM_ROOT=${EXASIM_DIR}"
    "EXASIM_INSTALL=${CMAKE_INSTALL_PREFIX}"
    "FRONTEND=python"
    "POSTPROCESS_TEST=1"
    "PYTHON3=${Python3_EXECUTABLE}"
    bash "${EXASIM_DIR}/tests/frontends/run-frontend-test.sh")
set_tests_properties(frontend_python_postprocess PROPERTIES
  TIMEOUT 2400 SKIP_RETURN_CODE 77)

# NOTE: frontend_python_modelcache (CACHE_TEST=1 in run-frontend-test.sh) is
# intentionally NOT wired: the second app dir does not currently hit the model
# cache ("FAIL: second app dir did not hit the model cache"), so the cache-hit
# path needs debugging in the frontend before this can be a CI gate. The README /
# smoke-cpu.yml previously claimed it ran -- that was documentation drift.

# Frontend Text2Code-source export: Python exports pdemodel.txt, pdeapp.txt,
# grid.bin, and optional field binaries; installed text2code must parse the
# package and regenerate datain without any frontend runtime.
add_test(NAME frontend_python_exporttext2code
  COMMAND ${CMAKE_COMMAND} -E env
    "EXASIM_ROOT=${EXASIM_DIR}"
    "EXASIM_INSTALL=${CMAKE_INSTALL_PREFIX}"
    "FRONTEND=python"
    "EXPORT_TEXT2CODE_TEST=1"
    "PYTHON3=${Python3_EXECUTABLE}"
    bash "${EXASIM_DIR}/tests/frontends/run-frontend-test.sh")
set_tests_properties(frontend_python_exporttext2code PROPERTIES
  TIMEOUT 2400 SKIP_RETURN_CODE 77 RESOURCE_LOCK exasim_text2code_install)

# Same package contract for the MATLAB exporter (tests/frontends/matlab/pdeapp_exporttext2code.m),
# which writes pdeapp.txt through its own key whitelist (writeinputfile.m), so a key it drops
# fails here. SKIPs (77) without MATLAB + the Symbolic Math Toolbox, like frontend_matlab.
add_test(NAME frontend_matlab_exporttext2code
  COMMAND ${CMAKE_COMMAND} -E env
    "EXASIM_ROOT=${EXASIM_DIR}"
    "EXASIM_INSTALL=${CMAKE_INSTALL_PREFIX}"
    "FRONTEND=matlab"
    "EXPORT_TEXT2CODE_TEST=1"
    bash "${EXASIM_DIR}/tests/frontends/run-frontend-test.sh")
set_tests_properties(frontend_matlab_exporttext2code PROPERTIES
  TIMEOUT 2400 SKIP_RETURN_CODE 77 RESOURCE_LOCK exasim_text2code_install)

# Direct Text2Code CLI regression (FRONTEND-INDEPENDENT): `text2code pdeapp.txt --out-dir <dir>`
# must emit the concrete-model header my_model.hpp + the generated symbolic sources. Run on a
# checked-in builtin fixture (Poisson / pdemodel1.txt) with no frontend, so a MATLAB/Julia or
# hand-written pdeapp cannot regress this CLI contract while frontend_python_exporttext2code (which
# only reaches --out-dir via the Python export) stays green. Shares the install-prefix write lock
# with that test (both build libt2cmodel*.{so,dylib} into the install).
add_test(NAME text2code_cli_out_dir
  COMMAND ${CMAKE_COMMAND} -E env
    "EXASIM_ROOT=${EXASIM_DIR}"
    "EXASIM_INSTALL=${CMAKE_INSTALL_PREFIX}"
    "EXASIM_TEXT2CODE=${EXASIM_TEXT2CODE}"
    bash "${EXASIM_DIR}/tests/text2code/cli-out-dir-test.sh")
set_tests_properties(text2code_cli_out_dir PROPERTIES
  TIMEOUT 1200 SKIP_RETURN_CODE 77 RESOURCE_LOCK exasim_text2code_install)

add_test(NAME text2code_poisson_variants
  COMMAND ${CMAKE_COMMAND} -E env
    "EXASIM_INSTALL=${CMAKE_INSTALL_PREFIX}"
    "EXASIM_ROOT=${EXASIM_DIR}"
    "KOKKOS_DIR=${Kokkos_DIR}"
    "CC=${CMAKE_C_COMPILER}"
    "CXX=${CMAKE_CXX_COMPILER}"
    bash "${EXASIM_DIR}/tests/text2code/poisson-variants-test.sh")
set_tests_properties(text2code_poisson_variants PROPERTIES
  TIMEOUT 1200 SKIP_RETURN_CODE 77 RESOURCE_LOCK exasim_text2code_install
  ENVIRONMENT "EXASIM_TEST_MAX_MPI_PROCS=2")

# Builtin model 4 kernel equivalence: model 4 (CNS5air reacting NS) switched
# from MATLAB-generated kernels to a text2code source (pdemodel4.txt). This
# test compiles the legacy checked-in kernels (backend/Model/BuiltIn/model4/)
# and the kernels THIS build regenerated into libbuiltinmodel side by side and
# checks 35 outputs — every kernel family, all 9 HDG BCs, and the analytic
# jacobians — agree to machine precision at a physical reacting-flow state.
# See tests/builtin-models/compare_model4.cpp for the full story.
get_property(_bm_genroot GLOBAL PROPERTY EXASIM_BUILTIN_GENROOT)
if(_bm_genroot AND EXISTS "${EXASIM_TEXT2CODE}"
   AND EXISTS "${EXASIM_DIR}/backend/Model/BuiltIn/pdeapp4.txt")
  # Stable include roots for the harness's "ref/..." / "gen/..." includes.
  # gen/ dangles until the codegen step runs; the target dependency below
  # guarantees ordering.
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/model4eq")
  file(CREATE_LINK "${EXASIM_DIR}/backend/Model/BuiltIn/model4"
       "${CMAKE_CURRENT_BINARY_DIR}/model4eq/ref" SYMBOLIC)
  file(CREATE_LINK "${_bm_genroot}/model4"
       "${CMAKE_CURRENT_BINARY_DIR}/model4eq/gen" SYMBOLIC)
  add_executable(model4_kernel_equivalence
    "${EXASIM_DIR}/tests/builtin-models/compare_model4.cpp")
  target_include_directories(model4_kernel_equivalence PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/model4eq")
  add_dependencies(model4_kernel_equivalence exasim_builtin_model_codegen)
  add_test(NAME builtin_model4_kernel_equivalence
    COMMAND model4_kernel_equivalence)
  set_tests_properties(builtin_model4_kernel_equivalence PROPERTIES TIMEOUT 60)
endif()


# Multi-domain HDG interface-coupling (Fint/Fext) kernel equivalence: prove the
# templated exasim::fint_kernel<M>/fext_kernel<M> assemble byte-identical
# residual + Jacobian buffers to the libpdemodel ABI HdgFint/HdgFext for a
# coupled model (apps/poisson/poisson2d/pdemodel2.txt + a non-trivial-Jacobian
# probe). Links real Kokkos; the ABI reference kernels are checked in under
# tests/coupling-models/reference/. See tests/coupling-models/compare_fint_fext.cpp.
if(TARGET Kokkos::kokkos)
  add_executable(fint_fext_equivalence
    "${EXASIM_DIR}/tests/coupling-models/compare_fint_fext.cpp")
  target_include_directories(fint_fext_equivalence PRIVATE
    "${EXASIM_INCLUDE_DIR}" "${EXASIM_DIR}"
    "${EXASIM_DIR}/tests/coupling-models")
  target_link_libraries(fint_fext_equivalence PRIVATE Kokkos::kokkos)
  target_compile_features(fint_fext_equivalence PRIVATE cxx_std_17)
  add_test(NAME coupling_fint_fext_equivalence
    COMMAND fint_fext_equivalence)
  set_tests_properties(coupling_fint_fext_equivalence PROPERTIES TIMEOUT 120)
endif()


# Generated-model coupling equivalence (Gap 1): prove text2code EMITS the Fint/Fext
# coupling surface into the generated `PdeModel` (has_external_coupling +
# nfint/nfext/ncuext + concrete fint/fext + Jacobians) so a text2code-GENERATED
# coupled model runs pure-template with NO loaded ABI. Drives the GENERATED PdeModel
# (checked in under tests/coupling-models/generated/, regenerated from pdemodel2.txt
# + probe_model.txt) through the templated exasim::fint_kernel<M>/fext_kernel<M> and
# asserts byte-identical residual + Jacobian buffers vs the SAME ABI HdgFint/HdgFext
# references in tests/coupling-models/reference/. The generated-model analog of
# coupling_fint_fext_equivalence (which used hand-written coupled models).
# See tests/coupling-models/compare_generated_fint_fext.cpp.
if(TARGET Kokkos::kokkos)
  add_executable(generated_fint_fext_equivalence
    "${EXASIM_DIR}/tests/coupling-models/compare_generated_fint_fext.cpp")
  target_include_directories(generated_fint_fext_equivalence PRIVATE
    "${EXASIM_INCLUDE_DIR}" "${EXASIM_DIR}"
    "${EXASIM_DIR}/tests/coupling-models")
  target_link_libraries(generated_fint_fext_equivalence PRIVATE Kokkos::kokkos)
  target_compile_features(generated_fint_fext_equivalence PRIVATE cxx_std_17)
  add_test(NAME generated_coupling_equivalence
    COMMAND generated_fint_fext_equivalence)
  set_tests_properties(generated_coupling_equivalence PROPERTIES TIMEOUT 120)
endif()


# Gap-3 dispatch regression: prove exasim::FintDriver<M>/FextDriver<M> (the OVERLOAD
# SET the concrete assembly sites call via EXASIM_DRIVER_CALL, uequation.hpp:409/1030)
# select the TYPED dstype* overload -- not the variadic precision fallback -- for the
# non-const dstype* buffers those sites pass. Without the Gap-3 SFINAE constraint the
# variadic wins overload resolution and forwards to the No-ABI multidomain stub, so the
# coupling residual is never assembled (SEGV / silent no-op). This test drives
# FintDriver<M>/FextDriver<M> for a GENERATED coupled model (has_external_coupling=true)
# and asserts the dispatched buffers equal the direct fint_kernel<M>/fext_kernel<M>
# output -- it FAILS without the fix. See tests/coupling-models/compare_dispatch_fint_fext.cpp.
if(TARGET Kokkos::kokkos)
  add_executable(dispatch_fint_fext_equivalence
    "${EXASIM_DIR}/tests/coupling-models/compare_dispatch_fint_fext.cpp")
  target_include_directories(dispatch_fint_fext_equivalence PRIVATE
    "${EXASIM_INCLUDE_DIR}" "${EXASIM_DIR}"
    "${EXASIM_DIR}/tests/coupling-models")
  target_link_libraries(dispatch_fint_fext_equivalence PRIVATE Kokkos::kokkos)
  target_compile_features(dispatch_fint_fext_equivalence PRIVATE cxx_std_17)
  add_test(NAME dispatch_coupling_equivalence
    COMMAND dispatch_fint_fext_equivalence)
  set_tests_properties(dispatch_coupling_equivalence PROPERTIES TIMEOUT 120)
endif()


# Templated interface-flux sampler equivalence (Gap 2): CInterfaceSampler's interface
# flux EXTRACTION now has a concrete-model path (getInterfaceFluxesAt*For<M>) that routes
# through exasim::FintDriver<M> / fint_only_kernel<M> with NO loaded ABI, while the
# AbiAdapter path stays byte-identical. The sampler's field-sampling half is model-free, so
# the concrete-vs-ABI flux equivalence reduces to fint_only_kernel<M> == the ABI HdgFintonly
# (the value-only kernel the ABI sampler calls) on the same interface-Gauss buffers. Drives
# the GENERATED coupled PdeModels through fint_only_kernel<M> and asserts equality to the
# checked-in ABI HdgFintonly references (tests/coupling-models/reference/*_HdgFintonly.cpp).
# The concrete-M sampler dispatch itself is compile-locked by tests/consumers/operators
# (explicit instantiation, built under consumer_builtin_cpu).
# See tests/coupling-models/compare_interface_sampler.cpp.
if(TARGET Kokkos::kokkos)
  add_executable(interface_sampler_equivalence_exe
    "${EXASIM_DIR}/tests/coupling-models/compare_interface_sampler.cpp")
  target_include_directories(interface_sampler_equivalence_exe PRIVATE
    "${EXASIM_INCLUDE_DIR}" "${EXASIM_DIR}"
    "${EXASIM_DIR}/tests/coupling-models")
  target_link_libraries(interface_sampler_equivalence_exe PRIVATE Kokkos::kokkos)
  target_compile_features(interface_sampler_equivalence_exe PRIVATE cxx_std_17)
  add_test(NAME interface_sampler_equivalence
    COMMAND interface_sampler_equivalence_exe)
  set_tests_properties(interface_sampler_equivalence PROPERTIES TIMEOUT 120)
endif()


# Implicit HDG w-equation source (Sourcew) kernel equivalence: prove the
# templated exasim::hdg_sourcew_kernel<M> / hdg_sourcewonly_kernel<M> assemble
# byte-identical value + ∂sw/∂udg + ∂sw/∂w buffers to the libpdemodel ABI
# HdgSourcew / HdgSourcewonly for an ncw>0 model (a 2-w probe that locks the
# input-index-outer Jacobian SoA layout). Links real Kokkos; the ABI reference
# kernels are checked in under tests/w-models/reference/.
# See tests/w-models/compare_hdg_sourcew.cpp.
if(TARGET Kokkos::kokkos)
  add_executable(hdg_sourcew_equivalence
    "${EXASIM_DIR}/tests/w-models/compare_hdg_sourcew.cpp")
  target_include_directories(hdg_sourcew_equivalence PRIVATE
    "${EXASIM_INCLUDE_DIR}" "${EXASIM_DIR}"
    "${EXASIM_DIR}/tests/w-models")
  target_link_libraries(hdg_sourcew_equivalence PRIVATE Kokkos::kokkos)
  target_compile_features(hdg_sourcew_equivalence PRIVATE cxx_std_17)
  add_test(NAME hdg_sourcew_equivalence
    COMMAND hdg_sourcew_equivalence)
  set_tests_properties(hdg_sourcew_equivalence PROPERTIES TIMEOUT 120)
endif()

# Templated boundary-output kernels (surface_quantities / qoi_boundary): the model gets the
# full tau (ntau != ncu), the real uinf and the real boundary id, and nsurfq beyond the
# per-point buffer is refused before launch. See tests/unit/surface_quantities_kernel.cpp.
if(TARGET Kokkos::kokkos)
  add_executable(unit_surface_quantities_kernel
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/surface_quantities_kernel.cpp")
  target_include_directories(unit_surface_quantities_kernel PRIVATE
    "${EXASIM_INCLUDE_DIR}" "${EXASIM_DIR}")
  target_link_libraries(unit_surface_quantities_kernel PRIVATE Kokkos::kokkos)
  target_compile_features(unit_surface_quantities_kernel PRIVATE cxx_std_17)
  add_test(NAME unit_surface_quantities_kernel COMMAND unit_surface_quantities_kernel)
  set_tests_properties(unit_surface_quantities_kernel PROPERTIES TIMEOUT 120)
endif()


# Shared-library app test: install Exasim, build apps/sharedlibrary against the
# installed package, generate libt2cmodelserial with the installed text2code,
# and run a small Poisson text2code fixture through the shared-library ABI path.
add_test(NAME sharedlibrary_app
  COMMAND ${CMAKE_COMMAND} -E env
    "EXASIM_BUILD=${CMAKE_BINARY_DIR}"
    "KOKKOS_DIR=${Kokkos_DIR}"
    "EXASIM_ROOT=${EXASIM_DIR}"
    "CC=${CMAKE_C_COMPILER}"
    "CXX=${CMAKE_CXX_COMPILER}"
    "INSTALL_PREFIX=/tmp/exasim_install_sharedlibrary"
    bash "${CMAKE_CURRENT_SOURCE_DIR}/run-sharedlibrary-test.sh")
set_tests_properties(sharedlibrary_app PROPERTIES
  TIMEOUT 2400 SKIP_RETURN_CODE 77)

# SurfaceQuantities on a moving mesh (cylinder Mach 8 mesh-adaptivity app, ibs = [3, 5, 6]):
# the per-save outbousurfgeo record must be the geometry the values were evaluated on after
# the mesh moved. Builds exasimapp like sharedlibrary_app; SKIPs (77) without text2code.
add_test(NAME surfacequantities_moving_mesh
  COMMAND ${CMAKE_COMMAND} -E env
    "EXASIM_BUILD=${CMAKE_BINARY_DIR}"
    "KOKKOS_DIR=${Kokkos_DIR}"
    "CC=${CMAKE_C_COMPILER}"
    "CXX=${CMAKE_CXX_COMPILER}"
    "INSTALL_PREFIX=/tmp/exasim_install_surfq_meshadapt"
    bash "${CMAKE_CURRENT_SOURCE_DIR}/run-surfacequantities-meshadapt.sh")
set_tests_properties(surfacequantities_moving_mesh PROPERTIES
  TIMEOUT 2400 SKIP_RETURN_CODE 77)

# PETSc operator export: PETSc (SNES + KSP=GMRES + MatShell + PCShell) drives the
# exported in-memory HDG operators for steady Poisson, validated against Exasim's
# own GMRES on the identical operators. Installs Exasim to a temp prefix, builds
# tests/petsc/petsc_poisson via find_package(Exasim) + PETSc (pkg-config), and runs
# the self-checking consumer. Gated on PETSc: the wrapper exits 77 (clean ctest
# SKIP) when no PETSc pkg-config module is found, so PETSc-less runners don't fail.
# (The lighter operator-export build/run is already covered by run-consumer-tests.sh
# via tests/consumers/operators, with no PETSc dependency.)
add_test(NAME petsc_operator_export
  COMMAND ${CMAKE_COMMAND} -E env
    "EXASIM_BUILD=${CMAKE_BINARY_DIR}"
    "KOKKOS_DIR=${Kokkos_DIR}"
    "EXASIM_ROOT=${EXASIM_DIR}"
    "CC=${CMAKE_C_COMPILER}"
    "CXX=${CMAKE_CXX_COMPILER}"
    "INSTALL_PREFIX=/tmp/exasim_install_petsc"
    bash "${EXASIM_DIR}/tests/petsc/run-petsc-test.sh")
set_tests_properties(petsc_operator_export PROPERTIES
  TIMEOUT 1800 SKIP_RETURN_CODE 77)

# ---------------------------------------------------------------------------------------
# Unit test: commonstruct default-initialization.
#
# In-tree and cheap (no solve, no mesh), unlike the consumer tests above. It exists because
# fields assigned only on the mpiProcs>1 path of setstructs.cpp are INDETERMINATE on the
# serial path, and one of them (couplingparams.ncie, which sizes four allocations) was
# caught in review rather than by a test. A field added without a default member
# initializer now fails here instead of surfacing later as a wrong allocation size.
add_executable(unit_struct_defaults ${CMAKE_CURRENT_SOURCE_DIR}/unit/struct_defaults.cpp)
target_link_libraries(unit_struct_defaults PRIVATE Kokkos::kokkos)
target_include_directories(unit_struct_defaults PRIVATE ${EXASIM_DIR})
if(EXASIM_MPI)
  find_package(MPI REQUIRED)
  target_link_libraries(unit_struct_defaults PRIVATE MPI::MPI_CXX)
  target_compile_definitions(unit_struct_defaults PRIVATE HAVE_MPI)
endif()
add_test(NAME unit_struct_defaults COMMAND unit_struct_defaults)
set_tests_properties(unit_struct_defaults PROPERTIES TIMEOUT 120)

# [nonmatching] Phase 0: the frozen non-matching interface header compiles and defaults to 0 / nullptr.
add_executable(unit_nonmatchinginterface_header ${CMAKE_CURRENT_SOURCE_DIR}/unit/nonmatchinginterface_header.cpp) # [nonmatching]
target_link_libraries(unit_nonmatchinginterface_header PRIVATE Kokkos::kokkos) # [nonmatching]
target_include_directories(unit_nonmatchinginterface_header PRIVATE ${EXASIM_DIR}) # [nonmatching]
if(EXASIM_MPI) # [nonmatching]
  target_link_libraries(unit_nonmatchinginterface_header PRIVATE MPI::MPI_CXX) # [nonmatching]
  target_compile_definitions(unit_nonmatchinginterface_header PRIVATE HAVE_MPI) # [nonmatching]
endif() # [nonmatching]
add_test(NAME unit_nonmatchinginterface_header COMMAND unit_nonmatchinginterface_header) # [nonmatching]
set_tests_properties(unit_nonmatchinginterface_header PROPERTIES TIMEOUT 120) # [nonmatching]

# [nonmatching] Phase 1: host geometry kernels for non-matching interface setup.
add_executable(unit_nonmatching_geometry ${CMAKE_CURRENT_SOURCE_DIR}/unit/nonmatching_geometry.cpp) # [nonmatching]
target_link_libraries(unit_nonmatching_geometry PRIVATE Kokkos::kokkos ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}) # [nonmatching]
target_include_directories(unit_nonmatching_geometry PRIVATE ${EXASIM_DIR}) # [nonmatching]
if(EXASIM_MPI) # [nonmatching]
  target_link_libraries(unit_nonmatching_geometry PRIVATE MPI::MPI_CXX) # [nonmatching]
  target_compile_definitions(unit_nonmatching_geometry PRIVATE HAVE_MPI) # [nonmatching]
endif() # [nonmatching]
add_test(NAME unit_nonmatching_geometry COMMAND unit_nonmatching_geometry) # [nonmatching]
set_tests_properties(unit_nonmatching_geometry PROPERTIES TIMEOUT 120) # [nonmatching]

# [nonmatching] Phase 2: donor search, 1, 2 and 4 ranks per model (MPI only).
if(EXASIM_MPI) # [nonmatching]
  add_executable(unit_nonmatching_donorsearch ${CMAKE_CURRENT_SOURCE_DIR}/unit/nonmatching_donorsearch.cpp) # [nonmatching]
  target_link_libraries(unit_nonmatching_donorsearch PRIVATE Kokkos::kokkos MPI::MPI_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}) # [nonmatching]
  target_include_directories(unit_nonmatching_donorsearch PRIVATE ${EXASIM_DIR}) # [nonmatching]
  target_compile_definitions(unit_nonmatching_donorsearch PRIVATE HAVE_MPI) # [nonmatching]
  foreach(np 2 4 8) # [nonmatching]
    add_test(NAME unit_nonmatching_donorsearch_np${np} # [nonmatching]
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} # [nonmatching]
              $<TARGET_FILE:unit_nonmatching_donorsearch> ${MPIEXEC_POSTFLAGS}) # [nonmatching]
    set_tests_properties(unit_nonmatching_donorsearch_np${np} PROPERTIES TIMEOUT 300) # [nonmatching]
  endforeach() # [nonmatching]

  # [nonmatching] Phase 3: D6-D7 setup distribution, stored point data, and communication lists.
  add_executable(unit_nonmatching_setup ${CMAKE_CURRENT_SOURCE_DIR}/unit/nonmatching_setup.cpp) # [nonmatching]
  target_link_libraries(unit_nonmatching_setup PRIVATE Kokkos::kokkos MPI::MPI_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}) # [nonmatching]
  target_include_directories(unit_nonmatching_setup PRIVATE ${EXASIM_DIR}) # [nonmatching]
  target_compile_definitions(unit_nonmatching_setup PRIVATE HAVE_MPI) # [nonmatching]
  foreach(np 2 4 8) # [nonmatching]
    add_test(NAME unit_nonmatching_setup_np${np} # [nonmatching]
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} # [nonmatching]
              $<TARGET_FILE:unit_nonmatching_setup> ${MPIEXEC_POSTFLAGS}) # [nonmatching]
    set_tests_properties(unit_nonmatching_setup_np${np} PROPERTIES TIMEOUT 300) # [nonmatching]
  endforeach() # [nonmatching]

  # [nonmatching] Phase 10: mixed triangle/quadrilateral face masters.
  add_executable(unit_nonmatching_mixed_meshes ${CMAKE_CURRENT_SOURCE_DIR}/unit/nonmatching_mixed_meshes.cpp) # [nonmatching]
  target_link_libraries(unit_nonmatching_mixed_meshes PRIVATE Kokkos::kokkos MPI::MPI_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}) # [nonmatching]
  target_include_directories(unit_nonmatching_mixed_meshes PRIVATE ${EXASIM_DIR}) # [nonmatching]
  target_compile_definitions(unit_nonmatching_mixed_meshes PRIVATE HAVE_MPI) # [nonmatching]
  foreach(np 2 4 8) # [nonmatching]
    add_test(NAME unit_nonmatching_mixed_meshes_np${np} # [nonmatching]
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} # [nonmatching]
              $<TARGET_FILE:unit_nonmatching_mixed_meshes> ${MPIEXEC_POSTFLAGS}) # [nonmatching]
    set_tests_properties(unit_nonmatching_mixed_meshes_np${np} PROPERTIES TIMEOUT 600) # [nonmatching]
  endforeach() # [nonmatching]
  foreach(mode porder pgauss npf ngf quadrature) # [nonmatching]
    add_test(NAME unit_nonmatching_mixed_incompatible_${mode} # [nonmatching] expected failure
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} # [nonmatching]
              $<TARGET_FILE:unit_nonmatching_mixed_meshes> ${MPIEXEC_POSTFLAGS} --incompatible ${mode}) # [nonmatching]
    set_tests_properties(unit_nonmatching_mixed_incompatible_${mode} PROPERTIES TIMEOUT 120 # [nonmatching]
      PASS_REGULAR_EXPRESSION "non-matching interface: incompatible face spaces") # [nonmatching]
  endforeach() # [nonmatching]
  add_test(NAME unit_nonmatching_mixed_incompatible_quadrature_local # [nonmatching] expected failure
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} # [nonmatching]
            $<TARGET_FILE:unit_nonmatching_mixed_meshes> ${MPIEXEC_POSTFLAGS} --incompatible quadrature_local) # [nonmatching]
  set_tests_properties(unit_nonmatching_mixed_incompatible_quadrature_local PROPERTIES TIMEOUT 120 # [nonmatching]
    PASS_REGULAR_EXPRESSION "non-matching interface: incompatible face spaces") # [nonmatching]
  add_test(NAME unit_nonmatching_mixed_missing_sidecar # [nonmatching] expected failure
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} # [nonmatching]
            $<TARGET_FILE:unit_nonmatching_mixed_meshes> ${MPIEXEC_POSTFLAGS} --incompatible sidecar) # [nonmatching]
  set_tests_properties(unit_nonmatching_mixed_missing_sidecar PROPERTIES TIMEOUT 120 # [nonmatching]
    PASS_REGULAR_EXPRESSION "mixed element families require the owner-test sidecar") # [nonmatching]

  # [nonmatching] Phase 4: residual/matrix quadrature and compact owner-donor exchange.
  add_executable(unit_nonmatching_coupling ${CMAKE_CURRENT_SOURCE_DIR}/unit/nonmatching_coupling.cpp) # [nonmatching]
  target_link_libraries(unit_nonmatching_coupling PRIVATE Kokkos::kokkos MPI::MPI_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}) # [nonmatching]
  target_include_directories(unit_nonmatching_coupling PRIVATE ${EXASIM_DIR}) # [nonmatching]
  target_compile_definitions(unit_nonmatching_coupling PRIVATE HAVE_MPI) # [nonmatching]
  foreach(np 2 4 8) # [nonmatching]
    add_test(NAME unit_nonmatching_coupling_np${np} # [nonmatching]
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} # [nonmatching]
              $<TARGET_FILE:unit_nonmatching_coupling> ${MPIEXEC_POSTFLAGS}) # [nonmatching]
    set_tests_properties(unit_nonmatching_coupling_np${np} PROPERTIES TIMEOUT 300) # [nonmatching]
  endforeach() # [nonmatching]

  # [nonmatching] Phase 7: distributed donor search on the interface communicator (equivalence with the exhaustive oracle).
  add_executable(unit_nonmatching_search ${CMAKE_CURRENT_SOURCE_DIR}/unit/nonmatching_search.cpp) # [nonmatching]
  target_link_libraries(unit_nonmatching_search PRIVATE Kokkos::kokkos MPI::MPI_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}) # [nonmatching]
  target_include_directories(unit_nonmatching_search PRIVATE ${EXASIM_DIR}) # [nonmatching]
  target_compile_definitions(unit_nonmatching_search PRIVATE HAVE_MPI) # [nonmatching]
  foreach(np 2 4 8) # [nonmatching]
    add_test(NAME unit_nonmatching_search_np${np} # [nonmatching]
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} # [nonmatching]
              $<TARGET_FILE:unit_nonmatching_search> ${MPIEXEC_POSTFLAGS}) # [nonmatching]
    set_tests_properties(unit_nonmatching_search_np${np} PROPERTIES TIMEOUT 600) # [nonmatching]
  endforeach() # [nonmatching]
  # [nonmatching] Phase 7 Revision 2: the same test compiled with -ffast-math, as the production
  # library is; only the strict-IEEE diagnostics are conditioned out (NONMATCHING_SEARCH_FAST_MATH).
  add_executable(unit_nonmatching_search_fastmath ${CMAKE_CURRENT_SOURCE_DIR}/unit/nonmatching_search.cpp) # [nonmatching]
  target_link_libraries(unit_nonmatching_search_fastmath PRIVATE Kokkos::kokkos MPI::MPI_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}) # [nonmatching]
  target_include_directories(unit_nonmatching_search_fastmath PRIVATE ${EXASIM_DIR}) # [nonmatching]
  target_compile_definitions(unit_nonmatching_search_fastmath PRIVATE HAVE_MPI NONMATCHING_SEARCH_FAST_MATH) # [nonmatching]
  target_compile_options(unit_nonmatching_search_fastmath PRIVATE -ffast-math) # [nonmatching]
  foreach(np 2 4 8) # [nonmatching]
    add_test(NAME unit_nonmatching_search_fastmath_np${np} # [nonmatching]
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} # [nonmatching]
              $<TARGET_FILE:unit_nonmatching_search_fastmath> ${MPIEXEC_POSTFLAGS}) # [nonmatching]
    set_tests_properties(unit_nonmatching_search_fastmath_np${np} PROPERTIES TIMEOUT 600) # [nonmatching]
  endforeach() # [nonmatching]

  # [nonmatching] Phase 5: non-matching coupling in hdgMatVec (FD, conformal equivalence, GMRES counts).
  add_executable(unit_nonmatching_matvec ${CMAKE_CURRENT_SOURCE_DIR}/unit/nonmatching_matvec.cpp) # [nonmatching]
  target_link_libraries(unit_nonmatching_matvec PRIVATE Kokkos::kokkos MPI::MPI_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}) # [nonmatching]
  target_include_directories(unit_nonmatching_matvec PRIVATE ${EXASIM_DIR}) # [nonmatching]
  target_compile_definitions(unit_nonmatching_matvec PRIVATE HAVE_MPI) # [nonmatching]
  foreach(np 2 4 8) # [nonmatching]
    add_test(NAME unit_nonmatching_matvec_np${np} # [nonmatching]
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} # [nonmatching]
              $<TARGET_FILE:unit_nonmatching_matvec> ${MPIEXEC_POSTFLAGS}) # [nonmatching]
    set_tests_properties(unit_nonmatching_matvec_np${np} PROPERTIES TIMEOUT 300) # [nonmatching]
  endforeach() # [nonmatching]
  # [nonmatching] Phase 9 B1: transfer consistency of u, q, uhat and the uncondensed residual
  # (p = 1..4, mesh pairs A/S/R, both directions; polynomial case exact). Two ranks.
  add_test(NAME unit_nonmatching_b1_transfer # [nonmatching]
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} # [nonmatching]
            $<TARGET_FILE:unit_nonmatching_matvec> ${MPIEXEC_POSTFLAGS} --b1-transfer) # [nonmatching]
  set_tests_properties(unit_nonmatching_b1_transfer PROPERTIES TIMEOUT 300) # [nonmatching]
  # [nonmatching] Phase 9 A1: opt-in composite owner quadrature on coincident 2D interfaces
  # (T1-T7, coincidence threshold, analytic piecewise integrals of both coupling laws,
  # partition independence at 1+1, 2+2, 4+4) and two expected-failure MPI tests.
  foreach(np 2 4 8) # [nonmatching]
    add_test(NAME unit_nonmatching_quadrature_np${np} # [nonmatching]
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} # [nonmatching]
              $<TARGET_FILE:unit_nonmatching_matvec> ${MPIEXEC_POSTFLAGS} --quadrature) # [nonmatching]
    set_tests_properties(unit_nonmatching_quadrature_np${np} PROPERTIES TIMEOUT 300) # [nonmatching]
  endforeach() # [nonmatching]
  add_test(NAME unit_nonmatching_quadrature_mismatch # [nonmatching] expected failure
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} # [nonmatching]
            $<TARGET_FILE:unit_nonmatching_matvec> ${MPIEXEC_POSTFLAGS} --quadrature-mismatch) # [nonmatching]
  set_tests_properties(unit_nonmatching_quadrature_mismatch PROPERTIES TIMEOUT 120 # [nonmatching]
    PASS_REGULAR_EXPRESSION "composite owner quadrature option differs across ranks") # [nonmatching]
  add_test(NAME unit_nonmatching_quadrature_piece_limit # [nonmatching] expected failure
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} # [nonmatching]
            $<TARGET_FILE:unit_nonmatching_matvec> ${MPIEXEC_POSTFLAGS} --quadrature-piece-limit) # [nonmatching]
  set_tests_properties(unit_nonmatching_quadrature_piece_limit PROPERTIES TIMEOUT 120 # [nonmatching]
    PASS_REGULAR_EXPRESSION "needs more than 2 pieces on one owner face") # [nonmatching]
  # [nonmatching] Phase 9 Revision 2: a forced endpoint candidate failure is a uniform search
  # error with its deterministic tuple, never the noncoincident fallback.
  add_test(NAME unit_nonmatching_quadrature_endpoint_failure # [nonmatching] expected failure
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} # [nonmatching]
            $<TARGET_FILE:unit_nonmatching_matvec> ${MPIEXEC_POSTFLAGS} --quadrature-endpoint-failure) # [nonmatching]
  set_tests_properties(unit_nonmatching_quadrature_endpoint_failure PROPERTIES TIMEOUT 120 # [nonmatching]
    PASS_REGULAR_EXPRESSION "composite owner quadrature endpoint search failed .*endpoint owner rank [0-9]+, direction [01] .*first failing donor rank [0-9]+, global face [0-9]+" # [nonmatching]
    FAIL_REGULAR_EXPRESSION "UNEXPECTED|interface not coincident") # [nonmatching]

  # [nonmatching] Phase 8: persistent backend runtime/workspace lifecycle.
  add_executable(unit_nonmatching_runtime ${CMAKE_CURRENT_SOURCE_DIR}/unit/nonmatching_runtime.cpp) # [nonmatching]
  target_link_libraries(unit_nonmatching_runtime PRIVATE Kokkos::kokkos MPI::MPI_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}) # [nonmatching]
  target_include_directories(unit_nonmatching_runtime PRIVATE ${EXASIM_DIR}) # [nonmatching]
  target_compile_definitions(unit_nonmatching_runtime PRIVATE HAVE_MPI) # [nonmatching]
  foreach(np 2 4 8) # [nonmatching]
    add_test(NAME unit_nonmatching_runtime_np${np} # [nonmatching]
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS} # [nonmatching]
              $<TARGET_FILE:unit_nonmatching_runtime> ${MPIEXEC_POSTFLAGS}) # [nonmatching]
    set_tests_properties(unit_nonmatching_runtime_np${np} PROPERTIES TIMEOUT 300) # [nonmatching]
  endforeach() # [nonmatching]
endif() # [nonmatching]

# Exact device-resident quantile selection used by mesh adaptivity. The reference
# intentionally uses std::sort in the test only; production never gathers values.
add_executable(unit_distributed_radix_quantiles
  ${CMAKE_CURRENT_SOURCE_DIR}/unit/distributed_radix_quantiles.cpp)
target_link_libraries(unit_distributed_radix_quantiles PRIVATE Kokkos::kokkos)
target_include_directories(unit_distributed_radix_quantiles PRIVATE ${EXASIM_DIR})
if(EXASIM_MPI)
  find_package(MPI REQUIRED)
  target_link_libraries(unit_distributed_radix_quantiles PRIVATE MPI::MPI_CXX)
  target_compile_definitions(unit_distributed_radix_quantiles PRIVATE HAVE_MPI)
  add_test(NAME unit_distributed_radix_quantiles
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:unit_distributed_radix_quantiles>
            ${MPIEXEC_POSTFLAGS})
else()
  add_test(NAME unit_distributed_radix_quantiles
    COMMAND unit_distributed_radix_quantiles)
endif()
set_tests_properties(unit_distributed_radix_quantiles PROPERTIES TIMEOUT 120)

# The LDG chemistry STG boundary must construct the same conservative Air-5
# state as the working HDG chemistry path, before HDG subtracts its trace.
add_executable(unit_stg_chem_ldg ${CMAKE_CURRENT_SOURCE_DIR}/unit/stg_chem_ldg.cpp)
target_link_libraries(unit_stg_chem_ldg PRIVATE Kokkos::kokkos)
target_include_directories(unit_stg_chem_ldg PRIVATE ${EXASIM_DIR})
add_test(NAME unit_stg_chem_ldg COMMAND unit_stg_chem_ldg)
set_tests_properties(unit_stg_chem_ldg PROPERTIES TIMEOUT 120)
