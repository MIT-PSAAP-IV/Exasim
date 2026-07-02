#!/bin/bash
# Generate golden MATLAB-frontend baselines: run each example in tests/matlab-examples.txt through
# the MATLAB frontend exasim(pde,mesh) (codegen + mesh + preprocessing + native solve) and save the
# volume solution (dataout/outudg_np*.bin) under tests/matlab-baselines/<Family>_<Case>/.
#
# Run ONCE on the REFERENCE frontend (main) -- check out main (or point EXASIM_REPO at a main
# checkout) so pdeapp.m's relative `run(.../frontends/Matlab/exasim_setup.m)` picks up main's
# frontend. Small outputs are committed via git-lfs; run-matlab-regression.sh diffs against them.
#
# MEMORY: exasim() spawns an internal mpirun (pde.mpiprocs procs) + compiles C++. Runs ONE example
# at a time. For unattended use, launch the whole script under the memguard governor:
#     ~/.claude/bin/runjob --name matlab-base -- tests/gen-matlab-baselines.sh
#
# Usage:  gen-matlab-baselines.sh [DEST]
#   DEST   output root (default: tests/matlab-baselines)
#   env EXASIM_REPO   repo whose examples/ + frontend to run (default: this repo)
#   env MATLAB        matlab binary (default: /Applications/MATLAB_R2026a.app/bin/matlab)
set -uo pipefail
REPO=${EXASIM_REPO:-$(cd "$(dirname "$0")/.." && pwd)}
P=$(cd "$REPO/.." && pwd)
set +u; source "$P/.stage/condaenv.sh" 2>/dev/null || true; set -u   # mpicc on PATH for the backend build
DEST=${1:-$REPO/tests/matlab-baselines}
MATLAB=${MATLAB:-/Applications/MATLAB_R2026a.app/bin/matlab}
LIST=$REPO/tests/matlab-examples.txt
[ -x "$MATLAB" ] || { echo "no matlab at $MATLAB (set MATLAB=)"; exit 2; }
mkdir -p "$DEST"
# MATLAB does not forward the conda toolchain to the cmake it spawns, so find_dependency(MPI) fails
# MPI_C_WORKS. Point cmake at the MPI wrappers (guarded; no-op where absent).
command -v mpicc  >/dev/null 2>&1 && export CC=${CC:-mpicc}
command -v mpicxx >/dev/null 2>&1 && export CXX=${CXX:-mpicxx}
# Headless: shadow scaplot with a no-op so examples never render (solve writes dataout first).
SHIM=$(mktemp -d); trap 'rm -rf "$SHIM"' EXIT
printf 'function varargout = scaplot(varargin)\nvarargout = cell(1, nargout);\nend\n' > "$SHIM/scaplot.m"
while read -r line; do
  case "$line" in ''|\#*) continue;; esac
  fam=${line%%/*}; case=${line#*/}; name="${fam}_${case}"
  exdir=$REPO/examples/$fam/$case
  pdeapp=$exdir/pdeapp.m
  [ -f "$pdeapp" ] || { echo "SKIP  $line (no pdeapp.m)"; continue; }
  # exasim() writes to pde.datapath/dataout, which resolves to the example dir (not our cwd). Wipe
  # first so stale output can't leak in. Plotting after the solve may warn headless; dataout is
  # already written by then, so ignore rc.
  dataout=$exdir/dataout; rm -rf "$dataout"
  "$MATLAB" -nodisplay -nosplash -batch "addpath('$SHIM'); run('$pdeapp')" \
      >/tmp/matbase_$name.log 2>&1 || echo "  (matlab rc!=0 for $line -- checking for output anyway)"
  shopt -s nullglob; bins=("$dataout"/outudg_np*.bin); shopt -u nullglob
  [ ${#bins[@]} -gt 0 ] || { echo "FAIL  $line (no dataout/outudg_np*.bin -- see /tmp/matbase_$name.log)"; continue; }
  out=$DEST/$name; rm -rf "$out"; mkdir -p "$out"
  cp "${bins[@]}" "$out"/
  cp "$dataout"/outqoi.txt "$out"/ 2>/dev/null
  echo "OK    $line  ranks=${#bins[@]}  size=$(du -sh "$out" | cut -f1)"
done < "$LIST"
echo MATLAB_BASELINE_DONE
