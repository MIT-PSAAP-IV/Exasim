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
# EXASIM_BUILD_TESTS=ON, builds, runs ctest). The superseded in-tree consumers
# (install_consumer / multi_tu_consumer) and bespoke runners now live under
# tests/archive/ — see tests/archive/README.md.
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
endforeach()

# Per-user model cache: a second app directory must reuse the cached
# (libfrontend_model, exasimapp) pair with zero compilation.
add_test(NAME frontend_python_modelcache
  COMMAND ${CMAKE_COMMAND} -E env
    "EXASIM_ROOT=${EXASIM_DIR}"
    "EXASIM_INSTALL=${CMAKE_INSTALL_PREFIX}"
    "FRONTEND=python"
    "CACHE_TEST=1"
    "PYTHON3=${Python3_EXECUTABLE}"
    bash "${EXASIM_DIR}/tests/frontends/run-frontend-test.sh")
set_tests_properties(frontend_python_modelcache PROPERTIES
  TIMEOUT 2400 SKIP_RETURN_CODE 77)
