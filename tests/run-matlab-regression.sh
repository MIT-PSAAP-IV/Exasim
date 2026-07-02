#!/bin/bash
# MATLAB-frontend golden regression. For every example in tests/matlab-examples.txt that has a
# committed baseline under tests/matlab-baselines/, run it through THIS repo's MATLAB frontend
# exasim(pde,mesh) and compare the volume solution (relative L2 of dataout/outudg_np*.bin) to the
# baseline generated from main. A frontend that codegens/preprocesses/solves identically gives
# rel_L2 = 0.
#
# This is the frontend-level companion to run-app-regression.sh (which tests the compiled C++ app):
# it additionally exercises the MATLAB codegen + mesh generation + preprocessing path. Baselines are
# git-lfs blobs from tests/gen-matlab-baselines.sh run on main.
#
# MEMORY: exasim() spawns an internal mpirun + compiles C++, one example at a time. Prefer:
#     ~/.claude/bin/runjob --name matlab-reg -- tests/run-matlab-regression.sh
#
# Usage:  run-matlab-regression.sh [TOL]
#   TOL   relative-L2 tolerance (default 1e-10)
#   env MATLAB   matlab binary (default: /Applications/MATLAB_R2026a.app/bin/matlab)
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
P=$(cd "$REPO/.." && pwd)
set +u; source "$P/.stage/condaenv.sh" 2>/dev/null || true; set -u   # numpy for compare + mpicc on PATH
MATLAB=${MATLAB:-/Applications/MATLAB_R2026a.app/bin/matlab}
LIST=$REPO/tests/matlab-examples.txt
BASE=$REPO/tests/matlab-baselines
TOL=${1:-1e-10}
[ -x "$MATLAB" ] || { echo "no matlab at $MATLAB (set MATLAB=)"; exit 2; }
[ -d "$BASE" ]   || { echo "no baselines at $BASE (git lfs pull?)"; exit 2; }
# The frontend rebuilds the backend via cmake, which runs find_dependency(MPI). MATLAB does not
# forward the conda toolchain, so cmake's default C compiler fails MPI_C_WORKS. Point it at the MPI
# wrappers (guarded: no-op where mpicc is absent) so MPI is found.
command -v mpicc  >/dev/null 2>&1 && export CC=${CC:-mpicc}
command -v mpicxx >/dev/null 2>&1 && export CXX=${CXX:-mpicxx}

# Headless: shadow the Exasim plot routine (scaplot) with a no-op so examples never render. The
# solve writes dataout before any plotting, so this changes nothing we compare -- it just avoids
# figures/renderers on -nodisplay. Prepended via addpath.
SHIM=$(mktemp -d); trap 'rm -rf "$SHIM"' EXIT
printf 'function varargout = scaplot(varargin)\nvarargout = cell(1, nargout);\nend\n' > "$SHIM/scaplot.m"

PASS=0; FAIL=0
while read -r line; do
  case "$line" in ''|\#*) continue;; esac
  fam=${line%%/*}; case=${line#*/}; name="${fam}_${case}"
  bdir=$BASE/$name
  [ -d "$bdir" ] || { echo "SKIP  $line (no committed baseline)"; continue; }
  exdir=$REPO/examples/$fam/$case
  pdeapp=$exdir/pdeapp.m
  [ -f "$pdeapp" ] || { echo "SKIP  $line (no pdeapp.m)"; continue; }
  # exasim() writes to pde.datapath/dataout, which resolves to the example dir (not our cwd).
  # Wipe it first so we never compare stale output, then read from there.
  dataout=$exdir/dataout; rm -rf "$dataout"
  "$MATLAB" -nodisplay -nosplash -batch "addpath('$SHIM'); run('$pdeapp')" \
      >/tmp/matreg_$name.log 2>&1 || true
  shopt -s nullglob; bins=("$dataout"/outudg_np*.bin); shopt -u nullglob
  [ ${#bins[@]} -gt 0 ] || { echo "$line  RUN_FAIL (no output -- /tmp/matreg_$name.log)"; FAIL=$((FAIL+1)); continue; }
  res=$(python3 "$REPO/tests/compare_app_l2.py" "$bdir" "$dataout" "$TOL" 2>&1)
  printf "%-32s %s\n" "$line" "$res"
  echo "$res" | grep -q PASS && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
done < "$LIST"
echo "=== matlab-frontend regression: $PASS pass, $FAIL fail (tol $TOL) ==="
[ "$FAIL" -eq 0 ]
