#!/usr/bin/env bash
# App-generation sweep: for every model with a pdemodel.txt, emit a standalone
# header-only app (pyt2c --emit-app) and verify it is CORRECT by compiling the
# emitted driver + generated/my_model.hpp against the real Exasim backend + PETSc +
# Kokkos (full templated instantiation: CSolution<PdeModel> + exasim::petsc::solve_steady).
# A clean compile proves the emitted app is a valid Model in the real solver templates.
#
# Env: PY, EXASIM_SRC, KOKKOS_INC (dir with Kokkos_Core.hpp), GEN_INC (dir with
#      exasim_paths.h), PETSC_CFLAGS (defaults to `pkg-config --cflags PETSc`), MPICXX.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
EXASIM_SRC="${EXASIM_SRC:-$(cd "$here/../.." && pwd)}"
PY="${PY:-python3}"
MPICXX="${MPICXX:-mpicxx}"
KOKKOS_INC="${KOKKOS_INC:?set KOKKOS_INC to a dir containing Kokkos_Core.hpp}"
GEN_INC="${GEN_INC:?set GEN_INC to a dir containing exasim_paths.h}"
PETSC_CFLAGS="${PETSC_CFLAGS:-$(pkg-config --cflags PETSc)}"
work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT

SWEEP_DIRS="${SWEEP_DIRS:-$EXASIM_SRC/examples $EXASIM_SRC/backend/Model/BuiltIn $EXASIM_SRC/apps $EXASIM_SRC/text2code/text2code}"
models=$(find $SWEEP_DIRS -name "pdemodel*.txt" 2>/dev/null | sort)

pass=0; fail=0; i=0
for m in $models; do
    i=$((i+1)); tag="${m#$EXASIM_SRC/}"; app="$work/app$i"
    if ! ( cd "$here/../pyt2c" && PYTHONPATH=. "$PY" -m pyt2c "$m" --emit-app "$app" --app-name a$i --model-id 100 ) >/dev/null 2>"$work/gen$i.err"; then
        echo "FAIL $tag :: emit-app: $(tail -1 "$work/gen$i.err" | head -c 100)"; fail=$((fail+1)); continue
    fi
    if "$MPICXX" -std=c++20 -fsyntax-only -D_MPI -DEXASIM_HAVE_PETSC -DHAVE_BACKEND_PREPROCESSING \
         -I"$EXASIM_SRC/include" -I"$EXASIM_SRC" -I"$GEN_INC" -I"$KOKKOS_INC" -I"$app" $PETSC_CFLAGS \
         "$app/a$i.cc" 2>"$work/cc$i.err"; then
        echo "PASS $tag"; pass=$((pass+1))
    else
        echo "FAIL $tag :: compile:"; grep 'error:' "$work/cc$i.err" | head -3 | sed 's/^/       /'; fail=$((fail+1))
    fi
done
echo "===================="
echo "appgen sweep: pass=$pass fail=$fail"
[ "$fail" -eq 0 ]
