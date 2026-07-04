#!/usr/bin/env bash
# GPU app-regression -- the CUDA counterpart of tests/run-app-regression.sh.
#
# For every committed baseline under tests/app-baselines/, build the matching app
# against the CUDA Exasim install (EXASIM_GPU=ON -> Kokkos nvcc_wrapper), run it on
# the GPU (platform="gpu"; GPU+MPI when the app's mpiprocs>1), and compare the volume
# solution (rel-L2 of dataout/outudg_np*.bin) to the SAME CPU golden baseline. GPU
# double and CPU double differ only by FP reduction ordering (~1e-10), so the gate is
# a slightly looser physical tol (default 1e-7) than the CPU regression's 1e-8.
#
# This is the GPU coverage that CANNOT run in GitHub CI (no GPU runner) -- it is the
# remote/manual counterpart. It also subsumes "ctest on GPU": rather than run the
# whole CPU-oriented ctest suite on the device, this exercises the meaningful GPU
# path (full native solve: preprocessing + codegen kernels + HDG/LDG assembly +
# Newton/GMRES + QoI) app-by-app on real meshes.
#
# Prereqs (dgx-b), after syncing the branch:
#   bash tests/remote/build-dgx.sh          # builds the CUDA gpu+gpumpi install
# Run (all baselines):
#   ssh dgx-b bash /data/scratch/teoc/exasim-teoc/tests/remote/gpu-app-regression.sh
# Cheap subset / single case:
#   CASES="poisson_poisson2d poisson_poisson3d" ssh dgx-b bash .../gpu-app-regression.sh
#
# Knobs: REPO, INSTALL (CUDA), KOKKOS (buildcuda), DATA, TOL (default 1e-7),
#        CASES (space-separated baseline dir names to include; default: all), JOBS.
set -eo pipefail

REPO=${REPO:-/data/scratch/teoc/exasim-teoc}
INSTALL=${INSTALL:-/data/scratch/teoc/exasim-teoc-install-cuda}
KOKKOS=${KOKKOS:-${REPO}/kokkos/buildcuda}
BASE=$REPO/tests/app-baselines
DATA=${DATA:-$INSTALL/share/exasim}
TOL=${TOL:-1e-7}

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
[ -f "${REPO}/env.sh" ] && source "${REPO}/env.sh"

[ -d "$BASE" ] || { echo "no baselines at $BASE (git lfs pull?)"; exit 2; }

PASS=0; FAIL=0; SKIP=0
for bdir in "$BASE"/*/; do
  name=$(basename "$bdir")
  # optional CASES filter (space-separated baseline dir names)
  if [ -n "${CASES:-}" ] && ! grep -qw "$name" <<<"$CASES"; then continue; fi
  base="${name%%__*}"                             # strip an optional __<variant> suffix
  app="${base%%_*}/${base#*_}"                    # poisson_poisson2d -> poisson/poisson2d
  adir=$REPO/apps/$app
  [ -f "$adir/grid.bin" ] || { echo "SKIP  $app (app sources missing)"; SKIP=$((SKIP+1)); continue; }
  label=$app; [ "$name" != "$base" ] && label="$app(${name#*__})"
  np=$(grep -oE 'mpiprocs *= *[0-9]+' "$adir/pdeapp.txt" | grep -oE '[0-9]+'); np=${np:-1}
  mpi=OFF; [ "$np" -gt 1 ] && mpi=ON

  # Reuse the (expensive nvcc) build across baselines that share an app dir + MPI
  # setting -- e.g. poisson2d, poisson2d(ldg) and poisson2d(ppdeg4) are the SAME app,
  # differing only by a runtime pdeapp.txt flag. Key the build on app+mpi (not the
  # baseline name) and don't wipe it: cmake --build is a near no-op for later variants.
  build=/tmp/gpureg_app_$(echo "${app}_mpi${mpi}" | tr '/ ' '__')
  exe=$(find "$build" -maxdepth 1 -name exasimapp -type f 2>/dev/null | head -1)
  if [ -z "$exe" ]; then
    cmake -S "$adir" -B "$build" -DCMAKE_PREFIX_PATH="$INSTALL;$KOKKOS" \
          -DEXASIM_GPU=ON -DEXASIM_MPI="$mpi" >/tmp/gpureg_cfg_$name.log 2>&1 \
      && cmake --build "$build" -j"${JOBS:-8}" >/tmp/gpureg_bld_$name.log 2>&1 \
      || { echo "$label[gpu]  BUILD_FAIL (see /tmp/gpureg_bld_$name.log)"; FAIL=$((FAIL+1)); continue; }
  else
    echo "  (reusing built exasimapp for $app, mpi=$mpi)"
  fi

  run=/tmp/gpureg_run_$name; rm -rf "$run"; mkdir -p "$run"; cp "$adir"/*.bin "$adir"/*.txt "$run"/ 2>/dev/null
  # Variant baselines ship a variant.sed -- the same pdeapp.txt flip used to generate them.
  [ -f "$bdir/variant.sed" ] && sed -i -f "$bdir/variant.sed" "$run/pdeapp.txt"
  sed -i 's/platform *= *"cpu";/platform = "gpu";/' "$run/pdeapp.txt"

  exe=$(find "$build" -maxdepth 1 -name exasimapp -type f | head -1)
  if [ "$np" = 1 ]; then
    (cd "$run" && EXASIM_DATA_DIR=$DATA "$exe" pdeapp.txt >/tmp/gpureg_run_$name.log 2>&1) \
      || { echo "$label[gpu]  RUN_FAIL (np=1, see /tmp/gpureg_run_$name.log)"; FAIL=$((FAIL+1)); continue; }
  else
    (cd "$run" && EXASIM_DATA_DIR=$DATA mpirun -np "$np" "$exe" pdeapp.txt >/tmp/gpureg_run_$name.log 2>&1) \
      || { echo "$label[gpu]  RUN_FAIL (gpu+mpi np=$np, see /tmp/gpureg_run_$name.log)"; FAIL=$((FAIL+1)); continue; }
  fi

  res=$(python3 "$REPO/tests/compare_app_l2.py" "$bdir" "$run/dataout" "$TOL" 2>&1)
  printf "%-34s %s\n" "$label[gpu np=$np]" "$res"
  echo "$res" | grep -q PASS && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
done
echo "=== GPU app regression: $PASS pass, $FAIL fail, $SKIP skip (tol $TOL vs CPU baselines) ==="
[ "$FAIL" -eq 0 ]
