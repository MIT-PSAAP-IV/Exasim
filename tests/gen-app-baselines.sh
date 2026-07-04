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
# All buildable apps. naca0012unsteady is ~270M (transient) -- generated locally, kept out of git.
# nsmach8/sharpb2/isoq previously segfaulted (input ncv=0 vs model ncv=2 -> null deref); fixed by
# setting ncv=2 in their pdeapp.txt, so their baselines are now generated + committed like the rest.
# Override with env APPS="..." to (re)baseline only a subset (e.g. when adding output families to a
# few apps without re-running the whole suite / the 270M transient).
APPS=${APPS:-"poisson/poisson2d poisson/poisson3d poisson/periodic poisson/lshape poisson/orion poisson/isoq3d poisson/cone navierstokes/naca0012steady navierstokes/naca0012unsteady navierstokes/nsmach8 navierstokes/sharpb2 navierstokes/isoq"}
# Solver-variant baselines (modest, coverage-guided axis b): pure-runtime input flips on an existing
# app (same mesh, no frontend re-preprocess) that exercise cold native solver/preconditioner code.
# Format: "<app>|<variant-suffix>|<sed-expr applied to pdeapp.txt>". The variant baseline lands in
# <app>__<suffix>/ and carries a variant.sed so run-app-regression.sh reproduces the same flip.
VARIANTS=(
  "poisson/poisson2d|ppdeg4|s/ppdegree = 1;/ppdegree = 4;/"   # native polynomial preconditioner (getpoly/ApplyPoly)
  "poisson/poisson2d|ldg|s/discretization = \"hdg\";/discretization = \"ldg\";/"   # LDG (hybrid=0) path: block-Jacobian + always-write connectivity
)
mkdir -p "$BASE"

# build+run app (with optional sed on pdeapp.txt) and store all output families under $out.
# args: <app> <name> <outdir> <sed-expr-or-empty>
gen_one() {
  local app=$1 name=$2 out=$3 sedexpr=$4 adir=$REPO/apps/$1 build=/tmp/gen_${2%%__*}
  [ -f "$adir/grid.bin" ] || { echo "SKIP $app"; return; }
  cmake -S "$adir" -B "$build" -DCMAKE_PREFIX_PATH="$INSTALL;$KOKKOS" >/tmp/gen_cfg_$name.log 2>&1 \
    && cmake --build "$build" -j6 >/tmp/gen_bld_$name.log 2>&1 || { echo "BUILD_FAIL $app"; return; }
  local np; np=$(grep -oE 'mpiprocs *= *[0-9]+' "$adir/pdeapp.txt" | grep -oE '[0-9]+'); np=${np:-1}
  local run=/tmp/gen_run_$name; rm -rf "$run"; mkdir -p "$run"; cp "$adir"/*.bin "$adir"/*.txt "$run"/ 2>/dev/null
  [ -n "$sedexpr" ] && sed -i '' "$sedexpr" "$run/pdeapp.txt"
  (cd "$run" && EXASIM_DATA_DIR=$DATA mpirun -np "$np" "$build/exasimapp" pdeapp.txt >/tmp/gen_run_$name.log 2>&1) \
    || { echo "RUN_FAIL $name (np=$np)"; return; }
  rm -rf "$out"; mkdir -p "$out"
  cp "$run"/dataout/outudg_np*.bin  "$out"/ 2>/dev/null   # volume solution
  cp "$run"/dataout/outuhat_np*.bin "$out"/ 2>/dev/null   # HDG trace (exercises the trace writer)
  cp "$run"/dataout/outqoi.txt      "$out"/ 2>/dev/null   # QoI
  [ -n "$sedexpr" ] && printf '%s\n' "$sedexpr" > "$out/variant.sed"
  echo "OK $name  np=$np  size=$(du -sh "$out" | cut -f1)"
}

for app in $APPS; do
  gen_one "$app" "$(echo "$app" | tr '/' '_')" "$BASE/$(echo "$app" | tr '/' '_')" ""
done
for v in "${VARIANTS[@]}"; do
  IFS='|' read -r app suffix sedexpr <<< "$v"
  gen_one "$app" "$(echo "$app" | tr '/' '_')__$suffix" "$BASE/$(echo "$app" | tr '/' '_')__$suffix" "$sedexpr"
done
echo BASELINE_DONE
