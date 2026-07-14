#!/usr/bin/env bash
# MMS reaction-diffusion analytical-solution smoke.
#
# Generates datain from the tracked mesh, builds the solve_steady verify harness against
# the installed Exasim (+ PETSc + Kokkos), runs it, and checks the computed field matches
# the exact manufactured solution u0=sin(pi x)sin(pi y), u1=0.5 sin(pi x)sin(pi y).
#
# Exits 77 (ctest SKIP) when a required dependency is unavailable (this is the PETSc
# emit-app path, so it SKIPs on runners without PETSc — like the other petsc-gated tests).
#
# Env (all optional; discovered/defaulted otherwise):
#   EXASIM_INSTALL  install prefix (has lib/cmake/Exasim + bin/text2code)
#   KOKKOS_DIR      a built Kokkos (has lib/cmake/Kokkos or the serial build dir)
#   CXX             compiler (default: mpicxx then c++)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$(mktemp -d)}"
mkdir -p "$WORK"

skip() { echo "SKIP: $*"; exit 77; }

# ---- locate the install + text2code + petsc (SKIP if any is missing) ----
EXASIM_INSTALL="${EXASIM_INSTALL:-${INSTALL_PREFIX:-}}"
[ -n "${EXASIM_INSTALL}" ] && [ -d "${EXASIM_INSTALL}/lib/cmake/Exasim" ] || skip "no Exasim install (set EXASIM_INSTALL)"
T2C="${EXASIM_INSTALL}/bin/text2code"; [ -x "$T2C" ] || T2C="$(command -v text2code || true)"
[ -x "$T2C" ] || skip "text2code not found"
command -v pkg-config >/dev/null && pkg-config --exists PETSc || skip "PETSc not found (petsc-gated test)"
KOKKOS_DIR="${KOKKOS_DIR:-}"
[ -n "$KOKKOS_DIR" ] || skip "KOKKOS_DIR not set"
# The mesh preprocessing (gendatain) reads master/gauss node data from the SOURCE tree
# (text2code/text2code/*.bin), so exasimpath must be the source, not the install.
EXASIM_ROOT="${EXASIM_ROOT:-$(cd "$HERE/../../.." && pwd)}"
[ -f "${EXASIM_ROOT}/text2code/text2code/masternodes.bin" ] || skip "EXASIM_ROOT source tree not found"

# ---- 1. generate datain from the tracked mesh (gendatain=1, gencode=0) ----
cp "$HERE/pdemodel.txt" "$HERE/grid.bin" "$WORK/"
sed -e "s#exasimpath[ \t]*=[ \t]*\"[^\"]*\"##" "$HERE/pdeapp.txt" > "$WORK/pdeapp.txt"
printf '\nexasimpath = "%s";\ndatapath = "%s";\n' "$EXASIM_ROOT" "$WORK" >> "$WORK/pdeapp.txt"
( cd "$WORK" && "$T2C" pdeapp.txt ) || skip "datain generation failed"
[ -f "$WORK/datain/mesh.bin" ] || skip "datain incomplete"

# ---- 2. configure + build the verify harness ----
cmake -S "$HERE" -B "$WORK/build" \
  -DCMAKE_PREFIX_PATH="${EXASIM_INSTALL}" \
  -DExasim_DIR="${EXASIM_INSTALL}/lib/cmake/Exasim" \
  -DKokkos_DIR="${KOKKOS_DIR}" \
  -DEXASIM_MPI=ON -DPKG_CONFIG_USE_CMAKE_PREFIX_PATH=ON >/dev/null || skip "configure failed (deps?)"
cmake --build "$WORK/build" --parallel 2 || { echo "FAIL: build"; exit 1; }

# ---- 3. run + verify (main.cpp returns non-zero if the field is wrong) ----
mkdir -p "$WORK/dataout"
( cd "$WORK" && "$WORK/build/consumer_mms" datain/ )
rc=$?
[ "$rc" -eq 0 ] && echo "consumer_mms: PASS" || echo "consumer_mms: FAIL (rc=$rc)"
exit "$rc"
