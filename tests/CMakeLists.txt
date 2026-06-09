# Test registration (gated by EXASIM_BUILD_TESTS; wired from install/CMakeLists.txt).
#
# B1-B4: out-of-tree consumer tests. run-consumer-tests.sh installs this build to a
# temp prefix, then for each project under tests/consumers/:
#   B3  asserts the consumer CMakeLists does NOT reference the Exasim source tree;
#   B2  configures it ONLY via find_package(Exasim) and builds it;
#   B4  runs it (if it ships a pdeapp.txt) and checks the QoI gate.
# This catches export/packaging regressions a normal in-tree build would miss.
#
# (The legacy in-tree multi_tu_consumer / install_consumer subprojects remain as
#  standalone consumer examples; their CMakeLists are exercised by the harness as
#  they are migrated under tests/consumers/.)
cmake_minimum_required(VERSION 3.16)

add_test(NAME consumer_out_of_tree
  COMMAND ${CMAKE_COMMAND} -E env
    "EXASIM_BUILD=${CMAKE_BINARY_DIR}"
    "KOKKOS_DIR=${Kokkos_DIR}"
    "EXASIM_ROOT=${EXASIM_DIR}"
    "CC=${CMAKE_C_COMPILER}"
    "CXX=${CMAKE_CXX_COMPILER}"
    bash "${CMAKE_CURRENT_SOURCE_DIR}/run-consumer-tests.sh")
set_tests_properties(consumer_out_of_tree PROPERTIES TIMEOUT 1200)
