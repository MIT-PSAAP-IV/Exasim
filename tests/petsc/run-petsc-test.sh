#!/usr/bin/env bash
# PETSc operator-export test: PETSc (SNES + KSP=GMRES + MatShell + PCShell) drives
# Exasim's exported in-memory HDG operators for steady Poisson, and the resulting
# trace is checked against Exasim's own GMRES on the IDENTICAL operators (the
# consumer self-checks and returns nonzero on mismatch).
#
# This is gated on PETSc being available: it exits 77 (a clean ctest SKIP) when no
# PETSc pkg-config module is found, so CI runners without PETSc don't hard-fail.
#
# Env (all optional):
#   EXASIM_BUILD     Exasim build dir to install from   (default: $REPO-build)
#   INSTALL_PREFIX   where to install Exasim            (default: /tmp/exasim_install_petsc)
#   KOKKOS_DIR       Kokkos install/build dir           (default: $REPO/deps/kokkos/buildserial)
#   PETSC_PREFIX     PETSc install prefix               (default: $CONDA_PREFIX, the toolchain
#                                                         that builds Exasim; ships PETSc+open-mpi)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXASIM_BUILD="${EXASIM_BUILD:-$REPO-build}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/tmp/exasim_install_petsc}"
KOKKOS_DIR="${KOKKOS_DIR:-$REPO/deps/kokkos/buildserial}"
PETSC_PREFIX="${PETSC_PREFIX:-${CONDA_PREFIX:-}}"

# ---- SKIP if PETSc is unavailable ----
PCP="${PETSC_PREFIX:+$PETSC_PREFIX/lib/pkgconfig}"
if ! PKG_CONFIG_PATH="$PCP:${PKG_CONFIG_PATH:-}" pkg-config --exists petsc 2>/dev/null; then
  echo "[petsc-test] no PETSc pkg-config module found (PETSC_PREFIX='$PETSC_PREFIX') -> SKIP"
  exit 77
fi
echo "[petsc-test] PETSc $(PKG_CONFIG_PATH="$PCP:${PKG_CONFIG_PATH:-}" pkg-config --modversion petsc) at ${PETSC_PREFIX:-<default>}"

# ---- install Exasim from the build dir ----
rm -rf "$INSTALL_PREFIX"
cmake --install "$EXASIM_BUILD" --prefix "$INSTALL_PREFIX" > /tmp/_petsc_install.log 2>&1 \
  || { echo "FAIL: cmake --install (see /tmp/_petsc_install.log)"; exit 1; }

# ---- configure + build the PETSc consumer out-of-tree (find_package only) ----
# Each consumer dir under tests/petsc/ is a self-checking executable (PETSc drives
# Exasim's exported operators and compares to Exasim-native). Build + run each.
fail=0
for dir in "$REPO"/tests/petsc/*/; do
  [ -f "$dir/CMakeLists.txt" ] || continue
  name="$(basename "$dir")"
  echo "=== petsc consumer: $name ==="
  bdir="/tmp/petsc_${name}_build"; rm -rf "$bdir"
  cmake -S "$dir" -B "$bdir" \
        -D "CMAKE_PREFIX_PATH=$INSTALL_PREFIX;$KOKKOS_DIR" \
        ${PETSC_PREFIX:+-DPETSC_PREFIX="$PETSC_PREFIX"} \
        ${CC:+-DCMAKE_C_COMPILER="$CC"} ${CXX:+-DCMAKE_CXX_COMPILER="$CXX"} \
        > "$bdir.cfg.log" 2>&1 || { echo "  FAIL: configure (see $bdir.cfg.log)"; tail -20 "$bdir.cfg.log"; fail=1; continue; }
  cmake --build "$bdir" -j > "$bdir.build.log" 2>&1 \
        || { echo "  FAIL: build (see $bdir.build.log)"; tail -30 "$bdir.build.log"; fail=1; continue; }
  echo "  [build] out-of-tree find_package build ok"

  exe="$(find "$bdir" -maxdepth 1 -type f -perm -u+x ! -name '*.cmake' | head -1)"
  rdir="/tmp/petsc_${name}_run"; rm -rf "$rdir"; mkdir -p "$rdir"
  if ( cd "$rdir" && EXASIM_DATA_DIR="$INSTALL_PREFIX/share/exasim" "$exe" > run.log 2>&1 ); then
    sed 's/^/  | /' "$rdir/run.log"
    echo "  [run] PASS"
  else
    echo "  FAIL: $name returned nonzero (see $rdir/run.log)"
    sed 's/^/  | /' "$rdir/run.log"; fail=1
  fi
done

echo "=========================================="
[ "$fail" -eq 0 ] && { echo "[petsc-test] ALL PASS"; exit 0; } || { echo "[petsc-test] SOME FAILED"; exit 1; }
