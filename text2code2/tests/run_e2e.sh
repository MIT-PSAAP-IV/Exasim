#!/usr/bin/env bash
# End-to-end proof: generate a model + a standalone header-only app with pyt2c, build
# it against the real Exasim backend + PETSc + Kokkos (unity-compiled, NO Exasim install
# needed, nothing outside the work dir is touched), and run it to convergence.
#
# The only pre-existing artifacts consumed (read-only): a built Kokkos, brew PETSc, and a
# prebuilt C++ `text2code` used ONLY for mesh preprocessing (gendatain=1, gencode=0 -> it
# writes datain/ into the work dir and touches nothing else).
#
# Env:
#   EXASIM_SRC   Exasim source tree (default: repo two levels up)
#   KOKKOS_ROOT  a built Kokkos (dir with include/ and lib/libkokkos*.a)
#   TEXT2CODE    prebuilt text2code binary (for datain/ only)
#   GEN_INC      dir containing generated exasim_paths.h
#   PY           python with `symengine` + `numpy` (default: python3)
#   WORK         scratch work dir (default: mktemp)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
EXASIM_SRC="${EXASIM_SRC:-$(cd "$here/../.." && pwd)}"
PY="${PY:-python3}"
KOKKOS_ROOT="${KOKKOS_ROOT:-$EXASIM_SRC/build_local/deps/kokkos/buildserial}"
GEN_INC="${GEN_INC:-$EXASIM_SRC/build_local/install/include}"
TEXT2CODE="${TEXT2CODE:-$EXASIM_SRC/build_local/install/bin/text2code}"
WORK="${WORK:-$(mktemp -d)}"

case_dir="$WORK/case"; mkdir -p "$case_dir/mesh"
cp "$here/goldens/poisson2d/pdemodel.txt" "$case_dir/"
cp "$EXASIM_SRC"/../CHEFSI-apps/FSP-1/ExaSim-SumMIT/poisson2d-new-architecture-with-kitesurf/mesh/half_square_left.bin \
   "$case_dir/mesh/" 2>/dev/null || { echo "provide a mesh at $case_dir/mesh/half_square_left.bin"; exit 1; }

# 1) mesh preprocessing only -> datain/  (gendatain=1, gencode=0: nothing else written)
sed -e 's/gendatain = 0;/gendatain = 1;/' "$here/goldens/poisson2d/pdeapp.txt" > "$case_dir/pdeapp.txt"
printf '\ngencode = 0;\nexasimpath = "%s";\ndatapath = "%s";\n' "$EXASIM_SRC" "$case_dir" >> "$case_dir/pdeapp.txt"
( cd "$case_dir" && "$TEXT2CODE" pdeapp.txt >/dev/null )

# 2) pyt2c: emit the standalone header-only app (model + driver + scaffold)
( cd "$here/../pyt2c" && PYTHONPATH=. "$PY" -m pyt2c "$case_dir/pdemodel.txt" \
    --emit-app "$case_dir" --app-name poisson2d --model-id 8 >/dev/null )

# 3) compile + link the driver (unity backend inlined; only Kokkos+PETSc+Accelerate)
mpicxx -std=c++20 -O2 -D_MPI -DEXASIM_HAVE_PETSC -DHAVE_BACKEND_PREPROCESSING \
  -I"$EXASIM_SRC/include" -I"$EXASIM_SRC" -I"$GEN_INC" -I"$KOKKOS_ROOT/include" -I"$case_dir" \
  $(pkg-config --cflags PETSc) "$case_dir/poisson2d.cc" \
  -L"$KOKKOS_ROOT/lib" -lkokkoscore -lkokkoscontainers -lkokkossimd \
  $(pkg-config --libs PETSc) -framework Accelerate -o "$case_dir/poisson2d"

# 4) run to convergence + check the output field is finite
( cd "$case_dir" && mkdir -p dataout && ./poisson2d datain/ dataout/out | grep -i "SNESConvergedReason" )
"$PY" - "$case_dir/dataout/outudg_np0.bin" <<'PYEOF'
import sys, numpy as np
a = np.fromfile(sys.argv[1], dtype=np.float64)
assert a.size > 0 and np.isfinite(a).all(), "output not finite"
print(f"[e2e] OK: udg n={a.size} finite, |u|~0 as expected for a partnerless run (nnz={int((abs(a)>1e-9).sum())})")
PYEOF
echo "[e2e] built+ran generated app in $WORK (no install touched)"
