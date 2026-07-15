#!/usr/bin/env bash
# Regenerate my_model.hpp with pyt2c for each golden model and prove numeric
# equivalence to the C++ text2code reference by compiling both into
# equiv_harness.cpp and diffing kernel outputs at a fixed input point.
#
# Usage: PY=/path/to/venv/python tests/run_equiv.sh   (PY defaults to python3)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python3}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
printf '#pragma once\n' > "$work/Kokkos_Core.hpp"

pass=0 fail=0
compare() {  # name  golden_hpp  pdemodel  extra_defs
    local name="$1" gold="$2" model="$3" defs="${4:-}"
    ( cd "$here/../pyt2c" && PYTHONPATH=. "$PY" -m pyt2c "$model" --stdout ) > "$work/$name.py.hpp"
    c++ -std=c++17 -O2 -I"$work" $defs -DMODEL_HEADER="\"$gold\"" "$here/equiv_harness.cpp" -o "$work/hg"
    c++ -std=c++17 -O2 -I"$work" $defs -DMODEL_HEADER="\"$work/$name.py.hpp\"" "$here/equiv_harness.cpp" -o "$work/hp"
    "$work/hg" > "$work/$name.g.txt"; "$work/hp" > "$work/$name.p.txt"
    if diff -q "$work/$name.g.txt" "$work/$name.p.txt" >/dev/null; then
        echo "  $name: NUMERICALLY IDENTICAL to C++ text2code golden"
        pass=$((pass+1))
    else
        # `|| true`: under `set -e`, diff returning 1 (differences) must not abort the
        # script before we record the failure and print the summary.
        echo "  $name: DIFFERS"; { diff "$work/$name.g.txt" "$work/$name.p.txt" | head; } || true
        fail=$((fail+1))
    fi
}

echo "pyt2c vs C++ text2code — kernel equivalence:"
compare isoq2d "$here/goldens/isoq2d_model100/my_model.ref.hpp" \
        "$here/goldens/isoq2d_model100/pdemodel.txt" "-DHAS_COUPLING"
compare poisson "$here/goldens/poisson2d/my_model.ref.hpp" \
        "$here/goldens/poisson2d/pdemodel.txt" ""

echo "pass=$pass fail=$fail"
[ "$fail" -eq 0 ]
