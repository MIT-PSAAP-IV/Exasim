#!/usr/bin/env bash
# tests/run-tests.sh — single entry point for Exasim's regression suite.
#
# Builds Exasim via the one-command superbuild with the ctest suite enabled,
# then runs ctest. All regression coverage lives behind ctest:
#
#   consumer_builtin_cpu  — CPU-only variant (EXASIM_MPI=OFF, NP=1):
#       B3  the consumer CMakeLists does not reach into the Exasim source tree
#       B2  configures via find_package(Exasim COMPONENTS cpu) and builds
#       B4  runs the built-in Poisson 2D and meets the QoI gate (QOI_TOL, default 1e-8)
#
#   consumer_builtin_mpi  — CPU+MPI variant (EXASIM_MPI=ON, NP=2):
#       B3/B2/B4 as above but linking Exasim::cpumpiprelib and running with mpirun -np 2
#
# This is what CI runs. Humans can equivalently configure with
# `-DEXASIM_BUILD_TESTS=ON` and run `ctest` in the solver build dir themselves.
#
# Usage:
#   bash tests/run-tests.sh                 # build (reusing vendored deps) + test
#   EXASIM_BUILD_DIR=/path bash tests/run-tests.sh     # use a specific superbuild dir
#   CMAKE_ARGS="-DEXASIM_CUDA=ON" bash tests/run-tests.sh   # extra configure args
#
# Toolchain (compiler/MPI) is taken from the environment — source the right
# environment first (e.g. the project conda env on macOS, module loads on a
# cluster), exactly as for a normal build.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXASIM_BUILD_DIR="${EXASIM_BUILD_DIR:-$ROOT/build}"

echo "[run-tests] repo  $ROOT"
echo "[run-tests] build $EXASIM_BUILD_DIR"

# 1. Configure + build the superbuild with the ctest suite enabled. The
#    find-or-build steps reuse any already-built vendored deps, so a warm tree
#    only rebuilds what changed.
# The superbuild's solver layer runs an install step; default it to a writable,
# build-local prefix so it never needs /usr/local (overridable via CMAKE_ARGS or
# a pre-seeded cache). The out-of-tree consumer test installs to its own temp
# prefix independently.
cmake -S "$ROOT" -B "$EXASIM_BUILD_DIR" -DEXASIM_BUILD_TESTS=ON \
      -DCMAKE_INSTALL_PREFIX="$EXASIM_BUILD_DIR/install" ${CMAKE_ARGS:-}
# Bound parallelism to the core count (override with JOBS=) — a bare `-j` is
# unlimited under Make and can OOM small CI runners on the big template TUs.
cmake --build "$EXASIM_BUILD_DIR" -j"${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

# 2. The solver layer (where the ctest is registered) is built by the
#    exasim_build ExternalProject; ctest runs from its binary dir.
INNER="$EXASIM_BUILD_DIR/exasim_build-prefix/src/exasim_build-build"
if [ ! -f "$INNER/CTestTestfile.cmake" ]; then
  echo "[run-tests] FAIL: no ctest registered at $INNER (was EXASIM_BUILD_TESTS honored?)" >&2
  exit 1
fi

# 3. Run the suite. Extra ctest flags pass through (e.g. -R name, -V).
ctest --test-dir "$INNER" --output-on-failure "$@"
