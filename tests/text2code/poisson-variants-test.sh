#!/usr/bin/env bash
set -euo pipefail

root=${EXASIM_ROOT:?EXASIM_ROOT not set}
prefix=${EXASIM_INSTALL:?EXASIM_INSTALL not set}
kokkos_dir=${KOKKOS_DIR:-$root/deps/kokkos/buildserial}
app=$root/apps/poisson/poisson2d
if [[ ! -x $prefix/bin/text2code ]] || ! command -v mpirun >/dev/null; then
    echo "SKIP: installed text2code or mpirun unavailable"
    exit 77
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
for variant in pdeapp.txt pdeapp_hdg_refined.txt; do
    name=${variant%.txt}
    mkdir "$work/$name"
    cp "$app/$variant" "$work/$name/pdeapp.txt"
    cp "$app/pdemodel.txt" "$app/grid.bin" "$work/$name/"
    if [[ -f $app/xdg.bin ]]; then cp "$app/xdg.bin" "$work/$name/"; fi
    (
        cd "$work/$name"
        EXASIM_PREFIX="$prefix" "$prefix/bin/text2code" pdeapp.txt > text2code.log 2>&1 || exit 1
        cmake -S "$root/apps/sharedlibrary" -B "$work/$name-build" \
            -D "CMAKE_PREFIX_PATH=$prefix;$kokkos_dir" \
            -DExasim_DIR="$prefix" \
            -DEXASIM_MPI=ON -DEXASIM_CUDA=OFF -DEXASIM_HIP=OFF \
            ${CC:+-DCMAKE_C_COMPILER="$CC"} ${CXX:+-DCMAKE_CXX_COMPILER="$CXX"} \
            > configure.log 2>&1 || exit 1
        cmake --build "$work/$name-build" -j > build.log 2>&1 || exit 1
        np=$(sed -n 's/^mpiprocs *= *\([0-9][0-9]*\);/\1/p' pdeapp.txt)
        EXASIM_PREFIX="$prefix" mpirun -np "$np" "$work/$name-build/exasimapp" pdeapp.txt > solver.log 2>&1 || exit 1
        python3 - <<'PY'
import math
from pathlib import Path

log = Path('solver.log').read_text().lower()
assert 'residual norm: nan' not in log and 'residual norm: inf' not in log, log[-2000:]
assert 'residual norm:' in log, log[-2000:]
rows = Path('dataout/outqoi.txt').read_text().splitlines()
qoi = float(rows[-1].split()[1])
assert math.isfinite(qoi) and qoi < 1e-8, f'Domain_QoI1={qoi}'
print(f'Domain_QoI1={qoi:.6e}')
PY
    ) || { cat "$work/$name/text2code.log" "$work/$name/configure.log" \
                  "$work/$name/build.log" "$work/$name/solver.log" 2>/dev/null; exit 1; }
    echo "$name: PASS"
done
