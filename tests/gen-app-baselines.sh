#!/bin/bash
# Generate golden app baselines: build+run each app on a BASELINE Exasim install (e.g. main) and
# save the volume solution (dataout/outudg_np*.bin + outqoi.txt) under tests/app-baselines/<app>.
# Run ONCE on the reference (main); the small outputs are committed via git-lfs and diffed by
# run-app-regression.sh. Big/broken apps are excluded from git (see tests/app-baselines/.gitignore).
#
# Usage:  gen-app-baselines.sh <BASELINE_INSTALL> [BASELINE_KOKKOS]
#   e.g.  gen-app-baselines.sh ~/projects/psaap4/Exasim-master-verify-build/install
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
P=$(cd "$REPO/.." && pwd)
set +u; source "$P/.stage/condaenv.sh" 2>/dev/null || true; set -u
INSTALL=${1:?usage: gen-app-baselines.sh <baseline-install> [kokkos]}
KOKKOS=${2:-$(dirname "$INSTALL")/deps/kokkos/buildserial}
DATA=$INSTALL/share/exasim
BASE=$REPO/tests/app-baselines
# All buildable apps. naca0012unsteady is ~270M (transient); nsmach8/sharpb2/isoq segfault on main
# as-shipped. Those are generated locally but kept out of git (.gitignore) -- regenerate on demand.
APPS="poisson/poisson2d poisson/poisson3d poisson/periodic poisson/lshape poisson/orion poisson/isoq3d poisson/cone navierstokes/naca0012steady navierstokes/naca0012unsteady navierstokes/nsmach8 navierstokes/sharpb2 navierstokes/isoq"
mkdir -p "$BASE"
for app in $APPS; do
  name=$(echo "$app" | tr '/' '_'); adir=$REPO/apps/$app; build=/tmp/gen_$name
  [ -f "$adir/grid.bin" ] || { echo "SKIP $app"; continue; }
  cmake -S "$adir" -B "$build" -DCMAKE_PREFIX_PATH="$INSTALL;$KOKKOS" >/tmp/gen_cfg_$name.log 2>&1 \
    && cmake --build "$build" -j6 >/tmp/gen_bld_$name.log 2>&1 || { echo "BUILD_FAIL $app"; continue; }
  np=$(grep -oE 'mpiprocs *= *[0-9]+' "$adir/pdeapp.txt" | grep -oE '[0-9]+'); np=${np:-1}
  run=/tmp/gen_run_$name; rm -rf "$run"; mkdir -p "$run"; cp "$adir"/*.bin "$adir"/*.txt "$run"/ 2>/dev/null
  (cd "$run" && EXASIM_DATA_DIR=$DATA mpirun -np "$np" "$build/exasimapp" pdeapp.txt >/tmp/gen_run_$name.log 2>&1) \
    || { echo "RUN_FAIL $app (np=$np)"; continue; }
  out=$BASE/$name; rm -rf "$out"; mkdir -p "$out"
  cp "$run"/dataout/outudg_np*.bin "$out"/ 2>/dev/null
  cp "$run"/dataout/outqoi.txt "$out"/ 2>/dev/null
  echo "OK $app  np=$np  size=$(du -sh "$out" | cut -f1)"
done
echo BASELINE_DONE
