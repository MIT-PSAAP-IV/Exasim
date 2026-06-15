#!/usr/bin/env bash
# End-to-end shared-library app test:
#   1. install Exasim to a temp prefix
#   2. build apps/sharedlibrary against that install via find_package(Exasim)
#   3. use the installed text2code to generate libt2cmodelserial into the install
#   4. run the shared-library app on the small Poisson external-model fixture
#   5. gate the QoI
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXASIM_BUILD="${EXASIM_BUILD:-$REPO-build}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/tmp/exasim_shared_install}"
KOKKOS_DIR="${KOKKOS_DIR:-$REPO/deps/kokkos/buildserial}"
EXASIM_ROOT="${EXASIM_ROOT:-$REPO}"
QOI_TOL="${QOI_TOL:-1e-8}"

echo "[sharedlibrary-test] repo           $REPO"
echo "[sharedlibrary-test] install prefix $INSTALL_PREFIX"

rm -rf "$INSTALL_PREFIX"
cmake --install "$EXASIM_BUILD" --prefix "$INSTALL_PREFIX" > /tmp/_sharedlibrary_install.log 2>&1 \
  || { echo "FAIL: cmake --install failed (see /tmp/_sharedlibrary_install.log)"; exit 1; }

if [ ! -x "$INSTALL_PREFIX/bin/text2code" ]; then
  echo "SKIP: installed text2code not found at $INSTALL_PREFIX/bin/text2code"
  exit 77
fi

BDIR="/tmp/exasim_sharedlibrary_build"
RDIR="/tmp/exasim_sharedlibrary_run"
rm -rf "$BDIR" "$RDIR"
mkdir -p "$RDIR"

cp "$REPO/tests/consumers/external-model"/pdeapp100.txt \
   "$REPO/tests/consumers/external-model"/pdemodel100.txt \
   "$REPO/tests/consumers/external-model"/*.bin \
   "$RDIR"/

python3 - <<'PY' "$RDIR/pdeapp100.txt" "$INSTALL_PREFIX/include"
from pathlib import Path
import sys
p = Path(sys.argv[1])
exasim_include = sys.argv[2]
text = p.read_text()
text = text.replace('exasimpath = "/placeholder";',
                    f'exasimpath = "{exasim_include}";')
p.write_text(text)
PY

(
  cd "$RDIR"
  EXASIM_PREFIX="$INSTALL_PREFIX" "$INSTALL_PREFIX/bin/text2code" "$RDIR/pdeapp100.txt" > text2code.log 2>&1
) || { echo "FAIL: text2code shared model generation (see $RDIR/text2code.log)"; exit 1; }

cmake -S "$REPO/apps/sharedlibrary" -B "$BDIR" \
  -D "CMAKE_PREFIX_PATH=$INSTALL_PREFIX;$KOKKOS_DIR" \
  -DExasim_DIR="$INSTALL_PREFIX" \
  -DEXASIM_MPI=OFF \
  -DEXASIM_CUDA=OFF \
  -DEXASIM_HIP=OFF \
  ${CC:+-DCMAKE_C_COMPILER="$CC"} ${CXX:+-DCMAKE_CXX_COMPILER="$CXX"} \
  > "$BDIR.cfg.log" 2>&1 || { echo "FAIL: sharedlibrary configure (see $BDIR.cfg.log)"; exit 1; }
cmake --build "$BDIR" -j > "$BDIR.build.log" 2>&1 \
  || { echo "FAIL: sharedlibrary build (see $BDIR.build.log)"; exit 1; }

(
  cd "$RDIR"
  EXASIM_PREFIX="$INSTALL_PREFIX" "$BDIR/exasimapp" pdeapp100.txt > run.log 2>&1
) || {
  echo "FAIL: sharedlibrary run nonzero exit (see $RDIR/run.log)"
  echo "--- $RDIR/run.log ---"
  sed 's/^/| /' "$RDIR/run.log"
  echo "--- end run.log ---"
  exit 1
}

qoi1="$(tail -1 "$RDIR/dataout/outqoi.txt" 2>/dev/null | awk '{print $2}')"
if [ -z "$qoi1" ]; then
  echo "FAIL: no QoI output"
  exit 1
fi
if awk "BEGIN{exit !(($qoi1)+0 < ($QOI_TOL)+0)}"; then
  echo "sharedlibrary_app PASSED (Domain_QoI1 = $qoi1 < $QOI_TOL)"
else
  echo "FAIL: QoI[1]=$qoi1 >= $QOI_TOL"
  exit 1
fi
