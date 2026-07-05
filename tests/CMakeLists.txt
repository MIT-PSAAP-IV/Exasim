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
