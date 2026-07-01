#!/bin/bash
# App-level golden regression: run every unique app against a chosen Exasim install and compare
# the volume solution (relative L2 of outudg_np*.bin) to the golden baseline in app-baselines/.
#
# The app SOURCE files (mesh grid.bin, generated my_model.hpp, pdeapp.txt) come from a single tree
# (Exasim-master-verify), so only the LIBRARY under test varies -- isolating solver/library changes.
# This exercises the full native (non-PETSc) stack: preprocessing + codegen kernels + HDG/LDG
# assembly + Newton/GMRES + QoI. Baselines are generated once from `main` (gen-baselines.sh).
#
# Usage:  run-app-regression.sh [EXASIM_INSTALL] [TOL]
#   EXASIM_INSTALL  install prefix of the library to test (default: my branch's Exasim-build/install)
#   TOL             relative-L2 tolerance (default 1e-10; a byte-identical solver gives ~0)
set -uo pipefail
P=/Users/teoc/projects/psaap4
set +u; source "$P/.stage/condaenv.sh" 2>/dev/null || true; set -u
SRC=$P/Exasim-master-verify
INSTALL=${1:-$P/Exasim-build/install}
KOKKOS=${EXASIM_KOKKOS:-$P/Exasim-build/deps/kokkos/buildserial}
DATA=$INSTALL/share/exasim
BASE=$P/app-baselines
TOL=${2:-1e-10}
APPS="poisson/poisson2d poisson/poisson3d poisson/periodic poisson/lshape poisson/orion poisson/isoq3d poisson/cone navierstokes/naca0012steady navierstokes/naca0012unsteady navierstokes/nsmach8 navierstokes/sharpb2 navierstokes/isoq"
PASS=0; FAIL=0
for app in $APPS; do
  name=$(echo "$app" | tr '/' '_'); adir=$SRC/apps/$app
  [ -d "$BASE/$name" ] || { echo "SKIP  $app (no baseline)"; continue; }
  bdir=/tmp/cmp_$name
  cmake -S "$adir" -B "$bdir" -DCMAKE_PREFIX_PATH="$INSTALL;$KOKKOS" >/tmp/cmpcfg_$name.log 2>&1 \
    && cmake --build "$bdir" -j6 >/tmp/cmpbld_$name.log 2>&1 || { echo "$app  BUILD_FAIL"; FAIL=$((FAIL+1)); continue; }
  np=$(grep -oE 'mpiprocs *= *[0-9]+' "$adir/pdeapp.txt" | grep -oE '[0-9]+'); np=${np:-1}
  run=/tmp/cmprun_$name; rm -rf "$run"; mkdir -p "$run"; cp "$adir"/*.bin "$adir"/*.txt "$run"/ 2>/dev/null
  (cd "$run" && EXASIM_DATA_DIR=$DATA mpirun -np "$np" "$bdir/exasimapp" pdeapp.txt >/tmp/cmprun_$name.log 2>&1) \
    || { echo "$app  RUN_FAIL (np=$np)"; FAIL=$((FAIL+1)); continue; }
  res=$(python3 "$P/Exasim/tests/compare_app_l2.py" "$BASE/$name" "$run/dataout" "$TOL" 2>&1)
  printf "%-32s %s\n" "$app" "$res"
  echo "$res" | grep -q PASS && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
done
echo "=== app regression: $PASS pass, $FAIL fail (tol $TOL) ==="
[ "$FAIL" -eq 0 ]
