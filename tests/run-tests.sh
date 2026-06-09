#!/usr/bin/env bash
# tests/run-tests.sh — single entry point for Exasim's regression suite.
#
# Builds Exasim via the one-command superbuild with the ctest suite enabled,
# then runs ctest. All regression coverage lives behind ctest:
#
#   consumer_out_of_tree  — for each project under tests/consumers/:
#       B3  the consumer CMakeLists does not reach into the Exasim source tree
#       B2  it configures via find_package(Exasim) against an install prefix and builds
#       B4  it runs (if it ships a pdeapp.txt) and meets the QoI gate (QOI_TOL, default 1e-8)
#
# This is what CI runs. Humans can equivalently configure with
# `-DEXASIM_BUILD_TESTS=ON` and run `ctest` in the solver build dir themselves.
#
# Usage:
#   bash tests/run-tests.sh                 # build (reusing vendored deps) + test
#   BUILD=/path bash tests/run-tests.sh     # use a specific superbuild dir
#   CMAKE_ARGS="-DEXASIM_CUDA=ON" bash tests/run-tests.sh   # extra configure args
#
# Toolchain (compiler/MPI) is taken from the environment — source the right
# environment first (e.g. the project conda env on macOS, module loads on a
# cluster), exactly as for a normal build.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${BUILD:-$ROOT/build}"

echo "[run-tests] repo  $ROOT"
echo "[run-tests] build $BUILD"

# 1. Configure + build the superbuild with the ctest suite enabled. The
#    find-or-build steps reuse any already-built vendored deps, so a warm tree
#    only rebuilds what changed.
cmake -S "$ROOT" -B "$BUILD" -DEXASIM_BUILD_TESTS=ON ${CMAKE_ARGS:-}
# Bound parallelism to the core count (override with JOBS=) — a bare `-j` is
# unlimited under Make and can OOM small CI runners on the big template TUs.
cmake --build "$BUILD" -j"${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

# 2. The solver layer (where the ctest is registered) is built by the
#    exasim_build ExternalProject; ctest runs from its binary dir.
INNER="$BUILD/exasim_build-prefix/src/exasim_build-build"
if [ ! -f "$INNER/CTestTestfile.cmake" ]; then
  echo "[run-tests] FAIL: no ctest registered at $INNER (was EXASIM_BUILD_TESTS honored?)" >&2
  exit 1
fi

# 3. Run the suite. Extra ctest flags pass through (e.g. -R name, -V).
ctest --test-dir "$INNER" --output-on-failure "$@"
