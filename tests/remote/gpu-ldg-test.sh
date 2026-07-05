#!/usr/bin/env bash
# LDG (hybrid=0) smoke test on GPU and GPU+MPI — the remote counterpart of the
# poisson2d(ldg) app-regression variant that covers CPU + CPU-MPI.
#
# Runs the poisson2d app in LDG mode with platform="gpu", once single-GPU (np=1,
# gpuprelib) and once GPU+MPI (np=2, gpumpiprelib), and checks that both converge
# and reproduce the domain QoI (int u = 4.052847e-01) the CPU/CPU-MPI LDG run
# produces. This exercises the LDG block-Jacobian + always-write-connectivity fix
# (see fix(ldg) commit) on the CUDA backend.
#
# Prereqs (run on dgx-b, after syncing the branch):
#   bash tests/remote/build-dgx.sh          # builds the CUDA gpu+gpumpi install
# Then:
#   ssh dgx-b bash /data/scratch/teoc/exasim-teoc/tests/remote/gpu-ldg-test.sh
#
# Paths match build-dgx.sh's out-of-tree CUDA install.
set -eo pipefail

REPO=${REPO:-/data/scratch/teoc/exasim-teoc}
INSTALL=${INSTALL:-/data/scratch/teoc/exasim-teoc-install-cuda}
KOKKOS=${KOKKOS:-${REPO}/kokkos/buildcuda}
APP=${REPO}/apps/poisson/poisson2d
EXPECT_QOI="4.052847e-01"     # Domain_QoI2 = int u, identical on CPU/CPU-MPI/GPU/GPU+MPI

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source "${REPO}/env.sh"

fail=0
run_case() {  # <label> <mpi:ON|OFF> <np>
  local label=$1 mpi=$2 np=$3
  local B=/tmp/ldggpu_${label}_build R=/tmp/ldggpu_${label}_run
  echo "==== ${label} (EXASIM_GPU=ON, EXASIM_MPI=${mpi}, np=${np}) ===="
  rm -rf "$B" "$R"
  cmake -S "$APP" -B "$B" -DCMAKE_PREFIX_PATH="${INSTALL};${KOKKOS}" \
        -DEXASIM_GPU=ON -DEXASIM_MPI="$mpi" >/tmp/ldggpu_${label}_cfg.log 2>&1 \
    && cmake --build "$B" -j"${JOBS:-8}" >/tmp/ldggpu_${label}_bld.log 2>&1 \
    || { echo "  FAIL: build"; tail -20 /tmp/ldggpu_${label}_bld.log; fail=1; return; }

  mkdir -p "$R"; cp "$APP"/*.bin "$APP"/*.txt "$R"/ 2>/dev/null
  sed -i 's/platform = "cpu";/platform = "gpu";/'             "$R/pdeapp.txt"
  sed -i 's/discretization = "hdg";/discretization = "ldg";/' "$R/pdeapp.txt"
  sed -i "s/mpiprocs = 2;/mpiprocs = ${np};/"                 "$R/pdeapp.txt"

  local exe; exe=$(find "$B" -maxdepth 1 -name exasimapp -type f | head -1)
  local ok=1
  if [ "$np" = 1 ]; then
    ( cd "$R" && EXASIM_DATA_DIR="$INSTALL/share/exasim" "$exe" pdeapp.txt >"$R/run.log" 2>&1 ) || ok=0
  else
    ( cd "$R" && EXASIM_DATA_DIR="$INSTALL/share/exasim" mpirun -np "$np" "$exe" pdeapp.txt >"$R/run.log" 2>&1 ) || ok=0
  fi
  [ "$ok" = 1 ] || { echo "  FAIL: run"; tail -20 "$R/run.log"; fail=1; return; }

  local qoi; qoi=$(sed -n '2p' "$R/dataout/outqoi.txt" 2>/dev/null | awk '{print $3}')
  grep -iE "GMRES converge|Residual Norm" "$R/run.log" | tail -2 | sed 's/^/    /'
  if [ "$qoi" = "$EXPECT_QOI" ]; then
    echo "  PASS: converged, Domain_QoI2 = ${qoi}"
  else
    echo "  FAIL: Domain_QoI2 = ${qoi} (expected ${EXPECT_QOI})"; fail=1
  fi
}

run_case gpu    OFF 1
run_case gpumpi ON  2

if [ "$fail" = 0 ]; then echo "==== GPU-LDG: PASS (single-GPU + GPU+MPI) ===="; else echo "==== GPU-LDG: FAIL ===="; fi
exit $fail
