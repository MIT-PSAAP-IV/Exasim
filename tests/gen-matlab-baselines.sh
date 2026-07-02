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
DEST=${1:-$REPO/tests/matlab-baselines}
MATLAB=${MATLAB:-/Applications/MATLAB_R2026a.app/bin/matlab}
LIST=$REPO/tests/matlab-examples.txt
[ -x "$MATLAB" ] || { echo "no matlab at $MATLAB (set MATLAB=)"; exit 2; }
mkdir -p "$DEST"
# Headless: shadow scaplot with a no-op so examples never render (solve writes dataout first).
SHIM=$(mktemp -d); trap 'rm -rf "$SHIM"' EXIT
printf 'function varargout = scaplot(varargin)\nvarargout = cell(1, nargout);\nend\n' > "$SHIM/scaplot.m"
while read -r line; do
  case "$line" in ''|\#*) continue;; esac
  fam=${line%%/*}; case=${line#*/}; name="${fam}_${case}"
  pdeapp=$REPO/examples/$fam/$case/pdeapp.m
  [ -f "$pdeapp" ] || { echo "SKIP  $line (no pdeapp.m)"; continue; }
  run=/tmp/matbase_$name; rm -rf "$run"; mkdir -p "$run"
  # datapath defaults to pwd(); running from $run isolates datain/dataout/build there. pdeapp.m
  # finds the frontend + mesh via mfilename('fullpath'), so an out-of-tree cwd is fine. Plotting
  # after the solve may warn under -nodisplay; dataout is already written by then, so ignore rc.
  ( cd "$run" && "$MATLAB" -nodisplay -nosplash -batch "addpath('$SHIM'); run('$pdeapp')" ) \
      >/tmp/matbase_$name.log 2>&1 || echo "  (matlab rc!=0 for $line -- checking for output anyway)"
  shopt -s nullglob; bins=("$run"/dataout/outudg_np*.bin); shopt -u nullglob
  [ ${#bins[@]} -gt 0 ] || { echo "FAIL  $line (no dataout/outudg_np*.bin -- see /tmp/matbase_$name.log)"; continue; }
  out=$DEST/$name; rm -rf "$out"; mkdir -p "$out"
  cp "${bins[@]}" "$out"/
  cp "$run"/dataout/outqoi.txt "$out"/ 2>/dev/null
  echo "OK    $line  ranks=${#bins[@]}  size=$(du -sh "$out" | cut -f1)"
done < "$LIST"
echo MATLAB_BASELINE_DONE
