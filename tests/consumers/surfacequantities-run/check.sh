#!/usr/bin/env bash
# Extra gate for surfacequantities-run, called by run-consumer-tests.sh after the main run:
#   check.sh <rundir>   with EXE (consumer executable) and NP (MPI ranks) in the environment.
# <rundir> already holds the saveSolBouLoc = 0, ibs = [1, 2] run. This checks it, then runs
# the Gauss-point variant and a single-id baseline in sibling directories.
set -euo pipefail
RDIR="$1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON3:-python3}"

run() {  # run <dir>: solve pdeapp.txt in <dir> with the same launcher as the harness
  if [ "${NP}" -gt 1 ]; then
    ( cd "$1" && mpirun -np "$NP" "$EXE" pdeapp.txt > run.log 2>&1 )
  else
    ( cd "$1" && "$EXE" pdeapp.txt > run.log 2>&1 )
  fi || { echo "FAIL: run in $1 (see $1/run.log)" >&2; tail -20 "$1/run.log" >&2; exit 1; }
}

variant() {  # variant <name> <sed expr>...: copy the fixture from RDIR, edit pdeapp.txt, run
  local d="${RDIR}_$1"; shift
  rm -rf "$d"; mkdir -p "$d"
  cp "$RDIR"/*.txt "$RDIR"/*.bin "$RDIR"/*.hpp "$d"/ 2>/dev/null || true
  for e in "$@"; do sed -i.bak "$e" "$d/pdeapp.txt"; done
  run "$d"
  echo "$d"
}

"$PY" "$HERE/check_surface.py" nodes "$RDIR/dataout/out" "$NP"

G="$(variant gauss 's/^saveSolBouLoc = 0;/saveSolBouLoc = 1;/')"
"$PY" "$HERE/check_surface.py" gauss "$G/dataout/out" "$NP"

B="$(variant single 's/^ibs = \[1, 2\];/ibs = 1;/' 's/^boundaryconditions = \[1, 2, 1, 2\];/boundaryconditions = [1, 1, 1, 1];/')"
"$PY" "$HERE/check_surface.py" union "$RDIR/dataout/out" "$B/dataout/out" "$NP"
