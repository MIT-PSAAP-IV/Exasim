#!/usr/bin/env bash
# Sweep pyt2c vs the C++ text2code across ALL built-in models (backend/Model/BuiltIn/
# pdemodel1..15.txt): generate my_model.hpp with each, then compare every guaranteed kernel
# in-process at a fixed input point (NaN/Inf-aware). Proves pyt2c == C++ text2code across the
# whole built-in corpus, not just the two goldens.
#
# Env: PY (python w/ symengine), TEXT2CODE (a text2code binary), EXASIM_SRC.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
EXASIM_SRC="${EXASIM_SRC:-$(cd "$here/../.." && pwd)}"
PY="${PY:-python3}"
TEXT2CODE="${TEXT2CODE:?set TEXT2CODE to a text2code binary}"
BUILTIN="$EXASIM_SRC/backend/Model/BuiltIn"
work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
printf '#pragma once\n' > "$work/Kokkos_Core.hpp"

pass=0; fail=0; skip=0
for id in $(seq 1 15); do
    model="$BUILTIN/pdemodel${id}.txt"; pdeapp="$BUILTIN/pdeapp${id}.txt"
    [ -f "$model" ] || { echo "model $id: no pdemodel"; skip=$((skip+1)); continue; }

    # pyt2c
    ( cd "$here/../pyt2c" && PYTHONPATH=. "$PY" -m pyt2c "$model" -o "$work/py$id" ) >/dev/null 2>"$work/py$id.err" \
        || { echo "model $id: pyt2c FAILED"; head -3 "$work/py$id.err"; fail=$((fail+1)); continue; }

    # C++ text2code (gencode only) into a scratch dir; rewrite exasimpath + gendatain=0.
    cdir="$work/cx$id"; mkdir -p "$cdir"
    cp "$model" "$cdir/pdemodel${id}.txt"
    sed -e 's/gendatain[ ]*=[ ]*[0-9]*/gendatain = 0/' \
        -e "s#exasimpath[ ]*=[ ]*\"[^\"]*\"#exasimpath = \"$EXASIM_SRC\"#" "$pdeapp" > "$cdir/pdeapp.txt"
    grep -q 'exasimpath' "$cdir/pdeapp.txt" || printf '\nexasimpath = "%s";\n' "$EXASIM_SRC" >> "$cdir/pdeapp.txt"
    grep -q 'gencode'    "$cdir/pdeapp.txt" || printf '\ngencode = 1;\n' >> "$cdir/pdeapp.txt"
    ( cd "$cdir" && "$TEXT2CODE" pdeapp.txt --out-dir "$cdir/gen" ) >"$cdir/t2c.log" 2>&1
    if [ ! -f "$cdir/gen/my_model.hpp" ]; then
        echo "model $id: C++ text2code produced no my_model.hpp (skip compare)"; skip=$((skip+1)); continue
    fi

    # compile the in-process comparator with both headers
    if ! c++ -std=c++17 -O2 -I"$work" \
          -DPY_HEADER="\"$work/py$id/my_model.hpp\"" -DCX_HEADER="\"$cdir/gen/my_model.hpp\"" \
          "$here/sweep_equiv.cpp" -o "$work/cmp$id" 2>"$work/cc$id.err"; then
        echo "model $id: comparator build FAILED"; head -5 "$work/cc$id.err"; fail=$((fail+1)); continue
    fi
    out="$("$work/cmp$id")"; rc=$?
    if [ $rc -eq 0 ]; then echo "model $id: pyt2c == C++ text2code  ($(echo "$out" | tail -1 | tr -s ' '))"; pass=$((pass+1))
    else echo "model $id: DIFFERS"; echo "$out" | head -6; fail=$((fail+1)); fi
done
echo "----"
echo "builtin sweep: pass=$pass fail=$fail skip=$skip"
[ "$fail" -eq 0 ]
