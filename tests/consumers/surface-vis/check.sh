#!/usr/bin/env bash
# B6 gate for surface-vis, called by run-consumer-tests.sh after the main run:
#   check.sh <rundir>   with EXE (consumer executable) and NP (MPI ranks) in the environment.
# The fixture runs saveParaview + saveSolBouFreq with ibs = 1, so the solve emits both
# outsurf*.vtu (ParaView) and outbousurf_np*.bin (raw nodal values) from the same
# SurfaceQuantities evaluation. This asserts the VTU output exists, then -- on
# single-rank runs, where the face-block layout is unambiguous -- cross-checks every
# VTU point value against outbousurf (the VTU stores float32, so the tolerance is 1e-6).
set -euo pipefail
RDIR="$1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON3:-python3}"
# The consumer main writes to a modelnumber-suffixed data dir (dataout100);
# locate it via the boundary file instead of hard-coding the name.
BOU="$(find "$RDIR" -name 'outbousurf_np0.bin' | head -1)"
[ -n "$BOU" ] || { echo "FAIL: no outbousurf_np0.bin under $RDIR"; exit 1; }
DATA="$(dirname "$BOU")"

n="$(find "$DATA" \( -name 'outsurf*.vtu' -o -name 'outsurf*.pvtu' \) -size +0c | wc -l | tr -d ' ')"
[ "$n" -gt 0 ] || { echo "FAIL: no nonempty outsurf pieces in $DATA"; exit 1; }
echo "surface vis ok: $n piece file(s)"
# The .pvd series is only written for time-dependent runs, whose pieces carry
# a 6-digit step tag (parallel rank pieces use a 5-digit _RRRRR tag instead).
if [ "$(find "$DATA" -regex '.*/outsurf_[0-9]\{6\}\.\(vtu\|pvtu\)' | wc -l | tr -d ' ')" -gt 0 ]; then
  pvd="$(find "$DATA" -maxdepth 1 -name 'outsurf*.pvd' | head -1)"
  [ -n "$pvd" ] || { echo "FAIL: stepped outsurf pieces but no outsurf.pvd series in $DATA"; exit 1; }
fi

if [ "${NP:-1}" = "1" ]; then
  "$PY" "$HERE/check_surf_vtu.py" "$DATA"
else
  r=0
  while [ "$r" -lt "$NP" ]; do
    "$PY" "$HERE/check_surf_vtu.py" "$DATA" "$r" || exit 1
    r=$((r + 1))
  done
fi
