#!/bin/bash
# App-level golden regression (native, non-PETSc solver). For every committed baseline under
# tests/app-baselines/, build the matching app (this repo's own apps/, self-contained) against a
# chosen Exasim install, run it as-shipped, and compare the volume solution (relative L2 of
# dataout/outudg_np*.bin) to the baseline. A byte-identical solver gives rel_L2 = 0.
#
# Exercises the full native stack: preprocessing + codegen kernels + HDG/LDG assembly +
# Newton/GMRES + QoI. Baselines are git-lfs blobs generated from main (tests/gen-app-baselines.sh).
# ParMETIS is deterministic for a fixed (mesh, nprocs), so baseline and new share the partition.
#
# Usage:  run-app-regression.sh [EXASIM_INSTALL] [TOL]
#   EXASIM_INSTALL  install prefix of the library under test (default: ../Exasim-build/install)
#   TOL             relative-L2 tolerance (default 1e-10)
#   EXASIM_KOKKOS   Kokkos build dir for the app cmake (default: ../Exasim-build/deps/kokkos/buildserial)
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)          # Exasim repo root
P=$(cd "$REPO/.." && pwd)                        # psaap4
set +u; source "$P/.stage/condaenv.sh" 2>/dev/null || true; set -u
INSTALL=${1:-$P/Exasim-build/install}
KOKKOS=${EXASIM_KOKKOS:-$P/Exasim-build/deps/kokkos/buildserial}
DATA=$INSTALL/share/exasim
BASE=$REPO/tests/app-baselines
# Physical tol (matches the project's Domain_QoI gate). Branch-vs-main is NOT bit-identical: FP
# reordering (parallel reductions) plus intended correctness fixes (e.g. the init_dispatch stride
# fix, which changes nonzero-IC apps like isoq3d) give ~1e-9 deltas. 1e-8 tolerates those while a
# real regression (a flipped model part changes the solution by O(1)) still fails loudly.
TOL=${2:-1e-8}
[ -d "$BASE" ] || { echo "no baselines at $BASE (git lfs pull?)"; exit 2; }

PASS=0; FAIL=0
for bdir in "$BASE"/*/; do
  name=$(basename "$bdir")
  base="${name%%__*}"                             # strip an optional __<variant> suffix
  app="${base%%_*}/${base#*_}"                    # poisson_poisson2d -> poisson/poisson2d
  adir=$REPO/apps/$app
  [ -f "$adir/grid.bin" ] || { echo "SKIP  $app (app sources missing)"; continue; }
  label=$app; [ "$name" != "$base" ] && label="$app(${name#*__})"
  build=/tmp/reg_$name
  cmake -S "$adir" -B "$build" -DCMAKE_PREFIX_PATH="$INSTALL;$KOKKOS" >/tmp/reg_cfg_$name.log 2>&1 \
    && cmake --build "$build" -j6 >/tmp/reg_bld_$name.log 2>&1 || { echo "$label  BUILD_FAIL"; FAIL=$((FAIL+1)); continue; }
  np=$(grep -oE 'mpiprocs *= *[0-9]+' "$adir/pdeapp.txt" | grep -oE '[0-9]+'); np=${np:-1}
  run=/tmp/reg_run_$name; rm -rf "$run"; mkdir -p "$run"; cp "$adir"/*.bin "$adir"/*.txt "$run"/ 2>/dev/null
  # Variant baselines ship a variant.sed -- the same pdeapp.txt flip used to generate them.
  [ -f "$bdir/variant.sed" ] && sed -i '' -f "$bdir/variant.sed" "$run/pdeapp.txt"
  (cd "$run" && EXASIM_DATA_DIR=$DATA mpirun -np "$np" "$build/exasimapp" pdeapp.txt >/tmp/reg_run_$name.log 2>&1) \
    || { echo "$label  RUN_FAIL (np=$np)"; FAIL=$((FAIL+1)); continue; }
  res=$(python3 "$REPO/tests/compare_app_l2.py" "$bdir" "$run/dataout" "$TOL" 2>&1)
  printf "%-32s %s\n" "$label" "$res"
  echo "$res" | grep -q PASS && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
done
echo "=== app regression: $PASS pass, $FAIL fail (tol $TOL) ==="
[ "$FAIL" -eq 0 ]
