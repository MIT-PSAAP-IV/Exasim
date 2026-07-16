#!/usr/bin/env bash
# GPU smoke for the header-only emit-app / exasim::petsc::solve_steady path — the CUDA
# counterpart of the CPU consumer_mms_reactdiff ctest (tests/mms-reactdiff).
#
# Two checks on the CUDA backend, both on the small 2-species manufactured-solution model
# u0 = sin(pi x) sin(pi y), u1 = 0.5 sin(pi x) sin(pi y):
#
#   A.  VERIFY harness (single GPU) — build tests/mms-reactdiff with -DEXASIM_GPU=ON and confirm
#       the solution recovered ON THE DEVICE matches the exact field (max|u-u_exact| < 2e-2).
#   A2. VERIFY harness (GPU + MPI)   — same, np=2 across 2 GPUs (partitioned datain); checks the
#       distributed trace solve (owned-element error reduced across ranks).
#   B.  EMITTED scaffold — run `text2code --emit-app` on the same model, build the generated
#       app with EXASIM_GPU=1 ./build.sh, run it, and confirm it converges (SNESConvergedReason=2).
#
# This exercises the PETSc-CUDA device Vec path (the petsc.hpp push_macro/undef HAVE_CUDA
# fix around the PETSc includes), the scaffold's compile-time backend selection (_CUDA -> 2),
# and the GPU+MPI halo exchange (EXASIM_COMM_LOCAL + per-rank device pinning).
#
# Prereqs (run on dgx-b, after syncing the branch):
#   bash tests/remote/build-dgx.sh          # builds the CUDA gpu+gpumpi install
# Then:
#   ssh dgx-b bash /data/scratch/teoc/exasim-teoc/tests/remote/gpu-mms-test.sh
#
# Paths match build-dgx.sh's out-of-tree CUDA install.
set -uo pipefail

REPO=${REPO:-/data/scratch/teoc/exasim-teoc}
INSTALL=${INSTALL:-/data/scratch/teoc/exasim-teoc-install-cuda}
KOKKOS=${KOKKOS:-${REPO}/kokkos/buildcuda}
NVCC_WRAPPER=${NVCC_WRAPPER:-${REPO}/deps/kokkos/bin/nvcc_wrapper}
PETSC_PKGCONFIG=${PETSC_PKGCONFIG:-/data/scratch/teoc/petsc/arch-cuda-mpi/lib/pkgconfig}
GPU_ARCH=${GPU_ARCH:-sm_70}
CUDA_LIBDIR=${CUDA_LIBDIR:-/usr/local/cuda/lib64}

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source "${REPO}/env.sh"
export PKG_CONFIG_PATH="${PETSC_PKGCONFIG}:${PKG_CONFIG_PATH:-}"
export NVCC_WRAPPER_DEFAULT_COMPILER="${NVCC_WRAPPER_DEFAULT_COMPILER:-mpicxx}"
export LD_LIBRARY_PATH="$(dirname "${PETSC_PKGCONFIG}")/../lib:${CUDA_LIBDIR}:${LD_LIBRARY_PATH}"

[ -d "${INSTALL}/lib/cmake/Exasim" ] || { echo "SKIP: no CUDA Exasim install at ${INSTALL} (run build-dgx.sh)"; exit 77; }
[ -x "${NVCC_WRAPPER}" ] || { echo "SKIP: nvcc_wrapper not found at ${NVCC_WRAPPER}"; exit 77; }

fail=0

# ---- A. on-device verify harness --------------------------------------------------------
echo "== A. mms verify harness on GPU (EXASIM_GPU=1) =="
if EXASIM_INSTALL="${INSTALL}" KOKKOS_DIR="${KOKKOS}" EXASIM_GPU=1 \
   NVCC_WRAPPER="${NVCC_WRAPPER}" EXASIM_GPU_ARCH="${GPU_ARCH}" \
   bash "${REPO}/tests/mms-reactdiff/run.sh"; then
  echo "A: PASS"
else
  rc=$?; [ "$rc" = 77 ] && { echo "A: SKIP"; } || { echo "A: FAIL (rc=$rc)"; fail=1; }
fi

# ---- A2. on-device verify harness, GPU + MPI (np=2) --------------------------------------
echo "== A2. mms verify harness on GPU + MPI, np=2 (EXASIM_GPU=1 NP=2) =="
if EXASIM_INSTALL="${INSTALL}" KOKKOS_DIR="${KOKKOS}" EXASIM_GPU=1 NP=2 \
   NVCC_WRAPPER="${NVCC_WRAPPER}" EXASIM_GPU_ARCH="${GPU_ARCH}" \
   bash "${REPO}/tests/mms-reactdiff/run.sh"; then
  echo "A2: PASS"
else
  rc=$?; [ "$rc" = 77 ] && { echo "A2: SKIP"; } || { echo "A2: FAIL (rc=$rc)"; fail=1; }
fi

# ---- B. emitted scaffold build+run on GPU -----------------------------------------------
echo "== B. text2code --emit-app -> GPU build =="
T2C="${INSTALL}/bin/text2code"; [ -x "$T2C" ] || T2C="$(command -v text2code || true)"
if [ ! -x "$T2C" ]; then
  echo "B: SKIP (text2code not found)"
else
  WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
  SRC="${REPO}/tests/mms-reactdiff"
  cp "${SRC}/pdemodel.txt" "${SRC}/grid.bin" "$WORK/"
  sed -e "s#exasimpath[ \t]*=[ \t]*\"[^\"]*\"##" "${SRC}/pdeapp.txt" > "$WORK/pdeapp.txt"
  printf '\nexasimpath = "%s";\ndatapath = "%s";\n' "${REPO}" "$WORK" >> "$WORK/pdeapp.txt"
  ( cd "$WORK" && "$T2C" pdeapp.txt --emit-app "$WORK/app" --app-name mmsgpu --model-id 100 ) \
    && [ -f "$WORK/app/CMakeLists.txt" ] || { echo "B: FAIL (emit-app)"; fail=1; }
  if [ -f "$WORK/app/CMakeLists.txt" ]; then
    # datain lives at $WORK/datain (from the emit run); run the app from there.
    ( cd "$WORK/app" \
      && EXASIM_INSTALL="${INSTALL}" KOKKOS_DIR="${KOKKOS}" EXASIM_GPU=1 \
         NVCC_WRAPPER="${NVCC_WRAPPER}" EXASIM_GPU_ARCH="${GPU_ARCH}" ./build.sh ) \
      && mkdir -p "$WORK/dataout" \
      && out=$( cd "$WORK" && mpirun -np 1 "$WORK/app/build/mmsgpu" "$WORK/datain/" "$WORK/dataout/out" 2>&1 ) \
      && echo "$out" | grep -q "SNESConvergedReason=2" \
      && echo "B: PASS ($(echo "$out" | grep -o 'SNESConvergedReason=[0-9]*'))" \
      || { echo "B: FAIL (build/run)"; echo "${out:-}" | tail -5; fail=1; }
  fi
fi

echo "======================="
[ "$fail" = 0 ] && echo "gpu-mms-test: PASS" || echo "gpu-mms-test: FAIL"
exit "$fail"
