#!/usr/bin/env bash
# Coverage report: categorize every model in the corpus by feature dimension, so we can
# see the equivalence/app sweeps exercise each type of model (dim, #components, coupling,
# matrix ops, aux w-field, extra outputs, discretization).
set -uo pipefail
EX="${EXASIM_SRC:-$(cd "$(dirname "$0")/../.." && pwd)}"
SWEEP_DIRS="${SWEEP_DIRS:-$EX/examples $EX/backend/Model/BuiltIn $EX/apps $EX/text2code/text2code}"

vs() { grep -oE "$2\([0-9]+\)" "$1" | head -1 | grep -oE "[0-9]+"; }   # vector size
declare -A feat
printf "%-42s %3s %4s %4s %5s %4s %-22s %s\n" "model" "nd" "ncu" "ncw" "coup" "mat" "extra-outputs" "disc"
printf '%.0s-' {1..110}; echo
for m in $(find $SWEEP_DIRS -name "pdemodel*.txt" 2>/dev/null | sort); do
  d=$(dirname "$m"); b=$(basename "$m" .txt); n=${b#pdemodel}
  app="$d/pdeapp$n.txt"; [ -f "$app" ] || app="$d/pdeapp.txt"
  nd=$(vs "$m" x); ncu=$(vs "$m" uhat); ncw=$(vs "$m" w)
  outs=$(grep -oE "^outputs.*" "$m" | sed 's/outputs //')
  coup="no"; echo "$outs" | grep -qE "Fint|Fext" && coup="yes"
  mat="no"; grep -qE "matrix |inv\(|det\(|transpose\(" "$m" && mat="yes"
  extra=""; for f in VisTensors EoS Sourcew Initw Initv QoIvolume QoIboundary; do echo "$outs" | grep -q "$f" && extra="$extra,$f"; done
  extra=${extra#,}
  disc=$(grep -oE 'discretization[ ]*=[ ]*"[a-z]+"' "$app" 2>/dev/null | grep -oE '"[a-z]+"' | tr -d '"'); disc=${disc:-hdg}
  printf "%-42s %3s %4s %4s %5s %4s %-22s %s\n" "${m#$EX/}" "$nd" "$ncu" "$ncw" "$coup" "$mat" "${extra:- -}" "$disc"
  feat[nd$nd]=1; feat[ncu$ncu]=1; feat[coup$coup]=1; feat[mat$mat]=1; feat[disc$disc]=1
  [ "${ncw:-0}" != "0" ] && [ -n "${ncw:-}" ] && feat[wfield]=1
  echo "$extra" | tr ',' '\n' | while read -r e; do [ -n "$e" ] && echo "EXTRA:$e"; done
done | tee /tmp/covrows.txt

echo
echo "=== feature dimensions exercised by the corpus ==="
echo "  dimensionality : $(grep -oE ' [0-9] ' /tmp/covrows.txt >/dev/null 2>&1; echo -n; awk '{print $2}' /tmp/covrows.txt | grep -E '^[0-9]$' | sort -u | tr '\n' ' ')(D)"
echo "  #components ncu: $(awk '{print $3}' /tmp/covrows.txt | grep -E '^[0-9]+$' | sort -un | tr '\n' ' ')"
echo "  discretization : $(awk '{print $NF}' /tmp/covrows.txt | grep -E 'hdg|ldg' | sort | uniq -c | tr '\n' ' ')"
echo "  coupling(Fint/Fext): $(awk '{print $5}' /tmp/covrows.txt | grep -c yes) models"
echo "  matrix ops (inv/det): $(awk '{print $6}' /tmp/covrows.txt | grep -c yes) models"
echo "  aux w-field (ncw>0): $(awk '$4>0{c++} END{print c+0}' /tmp/covrows.txt) models"
echo "  extra outputs seen : $(grep -oE 'EXTRA:[A-Za-z]+' /tmp/covrows.txt | sed 's/EXTRA://' | sort | uniq -c | tr '\n' ' ')"
