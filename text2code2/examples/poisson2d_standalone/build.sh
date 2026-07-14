#!/usr/bin/env bash
# Configure + build the standalone header-only app against a petsc-enabled Exasim install.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXASIM_INSTALL="${EXASIM_INSTALL:?set EXASIM_INSTALL to a petsc-enabled Exasim install prefix}"
KOKKOS_DIR="${KOKKOS_DIR:-$EXASIM_INSTALL/../Exasim-build/deps/kokkos/buildserial}"
BUILD="${BUILD:-$HERE/build}"

cmake -S "$HERE" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXASIM_MPI=ON -DEXASIM_GPU=OFF \
  -DCMAKE_PREFIX_PATH="$EXASIM_INSTALL" \
  -DExasim_DIR="$EXASIM_INSTALL/lib/cmake/Exasim" \
  -DKokkos_DIR="$KOKKOS_DIR" \
  -DPKG_CONFIG_USE_CMAKE_PREFIX_PATH=ON

cmake --build "$BUILD" -j 4
echo "built: $BUILD/poisson2d"
