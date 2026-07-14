#!/usr/bin/env bash
# Comprehensive pyt2c vs C++ text2code equivalence sweep over EVERY model with a
# pdemodel.txt + a sibling pdeapp (examples/ + backend/Model/BuiltIn/). For each:
#   1. STRUCTURAL: the set of generated `void <method>(` names must match exactly
#      (catches a missing/extra method — vis_tensors, fint/fext, jac_w, ...).
#   2. NUMERIC: every value method + Jacobian both headers define is compared
#      in-process at a fixed input point (NaN/Inf-aware).
#
# Env: PY (python w/ symengine), TEXT2CODE (a text2code binary), EXASIM_SRC.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
EXASIM_SRC="${EXASIM_SRC:-$(cd "$here/../.." && pwd)}"
PY="${PY:-python3}"
TEXT2CODE="${TEXT2CODE:?set TEXT2CODE to a text2code binary}"
work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
printf '#pragma once\n' > "$work/Kokkos_Core.hpp"

methods() { grep -oE "void [a-z_]+\(" "$1" | sort -u; }
has() { grep -q "void $2(" "$1"; }

# Search examples/, built-ins, and apps/ (the complex real-world NS/reacting models),
# plus the standalone text2code sample. Override with SWEEP_DIRS to narrow.
SWEEP_DIRS="${SWEEP_DIRS:-$EXASIM_SRC/examples $EXASIM_SRC/backend/Model/BuiltIn $EXASIM_SRC/apps $EXASIM_SRC/text2code/text2code}"
models=$(find $SWEEP_DIRS -name "pdemodel*.txt" 2>/dev/null | sort)
pass=0; fail=0; skip=0; i=0
for m in $models; do
    i=$((i+1)); d="$(dirname "$m")"; b="$(basename "$m" .txt)"; n="${b#pdemodel}"
    app="$d/pdeapp$n.txt"; [ -f "$app" ] || app="$d/pdeapp.txt"
    tag="${m#$EXASIM_SRC/}"
    [ -f "$app" ] || { echo "SKIP $tag (no pdeapp)"; skip=$((skip+1)); continue; }

    pyh="$work/py$i.hpp"
    if ! ( cd "$here/../pyt2c" && PYTHONPATH=. "$PY" -m pyt2c "$m" --stdout ) >"$pyh" 2>"$work/py$i.err"; then
        echo "FAIL $tag :: pyt2c: $(tail -1 "$work/py$i.err" | head -c 100)"; fail=$((fail+1)); continue
    fi

    cdir="$work/cx$i"; mkdir -p "$cdir"; cp "$m" "$cdir/$b.txt"
    sed -e 's/gendatain[ ]*=[ ]*[0-9]*/gendatain = 0/' \
        -e "s#exasimpath[ ]*=[ ]*\"[^\"]*\"#exasimpath = \"$EXASIM_SRC\"#" "$app" > "$cdir/pdeapp.txt"
    grep -q 'exasimpath' "$cdir/pdeapp.txt" || printf '\nexasimpath = "%s";\n' "$EXASIM_SRC" >> "$cdir/pdeapp.txt"
    grep -q 'gencode'    "$cdir/pdeapp.txt" || printf '\ngencode = 1;\n' >> "$cdir/pdeapp.txt"
    # Force gendatain=0 (codegen only, no mesh): append when the pdeapp omits it, else
    # the default gendatain=1 makes text2code try to read a grid.bin the sweep has no mesh for.
    grep -q 'gendatain' "$cdir/pdeapp.txt" || printf '\ngendatain = 0;\n' >> "$cdir/pdeapp.txt"
    # point modelfile at the copied model (name may differ from the sibling default)
    sed -i.bak "s#modelfile[ ]*=[ ]*\"[^\"]*\"#modelfile = \"$b.txt\"#" "$cdir/pdeapp.txt" 2>/dev/null || true
    ( cd "$cdir" && "$TEXT2CODE" pdeapp.txt --out-dir "$cdir/gen" ) >"$cdir/t2c.log" 2>&1
    cxh="$cdir/gen/my_model.hpp"
    [ -f "$cxh" ] || { echo "SKIP $tag (C++ text2code produced no header; see log)"; skip=$((skip+1)); continue; }

    # 1. structural: identical method sets
    if ! diff <(methods "$pyh") <(methods "$cxh") >"$work/struct$i.diff"; then
        echo "FAIL $tag :: STRUCTURAL method-set mismatch:"; sed 's/^/       /' "$work/struct$i.diff" | head; fail=$((fail+1)); continue
    fi

    # 2. numeric: guard optional methods by presence
    flags=""
    has "$pyh" initu        && flags="$flags -DHAS_INITU"
    has "$pyh" vis_scalars  && flags="$flags -DHAS_VISSC"
    has "$pyh" vis_vectors  && flags="$flags -DHAS_VISVEC"
    has "$pyh" qoi_volume   && flags="$flags -DHAS_QOIV"
    has "$pyh" qoi_boundary && flags="$flags -DHAS_QOIB"
    has "$pyh" fint         && flags="$flags -DHAS_FINT"
    has "$pyh" fext         && flags="$flags -DHAS_FEXT"
    has "$pyh" flux_jac_w   && flags="$flags -DHAS_FLUXJACW"
    if ! c++ -std=c++17 -O2 -I"$work" $flags \
          -DPY_HEADER="\"$pyh\"" -DCX_HEADER="\"$cxh\"" "$here/sweep_equiv.cpp" -o "$work/cmp$i" 2>"$work/cc$i.err"; then
        echo "FAIL $tag :: comparator build:"; grep 'error:' "$work/cc$i.err" | head -3 | sed 's/^/       /'; fail=$((fail+1)); continue
    fi
    out="$("$work/cmp$i")"; rc=$?
    if [ $rc -eq 0 ]; then
        echo "PASS $tag  (methods: $(methods "$pyh" | wc -l | tr -d ' '); $(echo "$out" | tail -1 | tr -s ' '))"; pass=$((pass+1))
    else
        echo "FAIL $tag :: NUMERIC"; echo "$out" | head -6 | sed 's/^/       /'; fail=$((fail+1))
    fi
done
echo "===================="
echo "model sweep: pass=$pass fail=$fail skip=$skip"
[ "$fail" -eq 0 ]
