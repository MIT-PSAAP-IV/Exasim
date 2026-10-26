#!/usr/bin/env bash
# Phase 2 rank-layout and real graph-guard checks. Reviewer-owned execution.
set -euo pipefail

layout=${DOMAIN_LAYOUT_TEST:?DOMAIN_LAYOUT_TEST not set}
mpiexec=${MPIEXEC:-mpiexec}
root=${EXASIM_ROOT:?EXASIM_ROOT not set}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

check_layout() {
  local counts=$1 ranks=$2 expected=$3
  "$mpiexec" -np "$ranks" "$layout" --emit "$counts" |
    LC_ALL=C sort -t= -k2,2n > "$work/actual"
  printf '%s\n' "$expected" > "$work/expected"
  diff -u "$work/expected" "$work/actual"
}

check_layout 1,1,1 3 \
'rank=0 domain=0 localRank=0 fileoffset=0 filein=datain1/ fileout=dataout1/out modelDefinition=0
rank=1 domain=1 localRank=0 fileoffset=1 filein=datain2/ fileout=dataout2/out modelDefinition=1
rank=2 domain=2 localRank=0 fileoffset=2 filein=datain3/ fileout=dataout3/out modelDefinition=2'

check_layout 1,2,3,1 7 \
'rank=0 domain=0 localRank=0 fileoffset=0 filein=datain1/ fileout=dataout1/out modelDefinition=0
rank=1 domain=1 localRank=0 fileoffset=1 filein=datain2/ fileout=dataout2/out modelDefinition=1
rank=2 domain=1 localRank=1 fileoffset=1 filein=datain2/ fileout=dataout2/out modelDefinition=1
rank=3 domain=2 localRank=0 fileoffset=3 filein=datain3/ fileout=dataout3/out modelDefinition=2
rank=4 domain=2 localRank=1 fileoffset=3 filein=datain3/ fileout=dataout3/out modelDefinition=2
rank=5 domain=2 localRank=2 fileoffset=3 filein=datain3/ fileout=dataout3/out modelDefinition=2
rank=6 domain=3 localRank=0 fileoffset=6 filein=datain4/ fileout=dataout4/out modelDefinition=3'

check_layout 3,1,2,1 7 \
'rank=0 domain=0 localRank=0 fileoffset=0 filein=datain1/ fileout=dataout1/out modelDefinition=0
rank=1 domain=0 localRank=1 fileoffset=0 filein=datain1/ fileout=dataout1/out modelDefinition=0
rank=2 domain=0 localRank=2 fileoffset=0 filein=datain1/ fileout=dataout1/out modelDefinition=0
rank=3 domain=1 localRank=0 fileoffset=3 filein=datain2/ fileout=dataout2/out modelDefinition=1
rank=4 domain=2 localRank=0 fileoffset=4 filein=datain3/ fileout=dataout3/out modelDefinition=2
rank=5 domain=2 localRank=1 fileoffset=4 filein=datain3/ fileout=dataout3/out modelDefinition=2
rank=6 domain=3 localRank=0 fileoffset=6 filein=datain4/ fileout=dataout4/out modelDefinition=3'

if [[ -n ${EXASIMAPP:-} && -n ${EXASIM_PREFIX:-} ]]; then
  fixture=$root/apps/coupledproblems/poisson/nonmatching
  for domain in 1 2 3 4; do
    mkdir -p "$work/d$domain"
    cp "$fixture/pdeapp1.txt" "$work/d$domain/pdeapp.txt"
    cp "$fixture/pdemodel1.txt" "$work/d$domain/pdemodel1.txt"
    cp "$fixture/grid1.bin" "$work/d$domain/grid1.bin"
  done
  python3 - "$work" <<'PY'
from pathlib import Path
import re, sys
root=Path(sys.argv[1])
for d in range(1,5):
    p=root/f'd{d}'/'pdeapp.txt'; s=p.read_text()
    s=re.sub(r'^interfaceconditions\s*=.*$', 'interfaceconditions = [];', s, flags=re.M)
    s=re.sub(r'^interfacefluxmap\s*=.*$', 'interfacefluxmap = [];', s, flags=re.M)
    s=re.sub(r'^boundaryconditions\s*=.*$', 'boundaryconditions = [1, 2, 1, 3];', s, flags=re.M)
    s=re.sub(r'^builtinmodelID\s*=.*$', 'builtinmodelID = 0;', s, flags=re.M)
    p.write_text(s)
    model=root/f'd{d}'/'pdemodel1.txt'; s=model.read_text()
    s=s.replace('output_size(fb) = 2;\n  fb[0] = -uhat[0];\n  fb[1] = tau[0]*(uq[0]-uhat[0])+n[0]*mu[0]*uq[1]+n[1]*mu[0]*uq[2];',
                'output_size(fb) = 3;\n  fb[0] = -uhat[0];\n  fb[1] = tau[0]*(uq[0]-uhat[0])+n[0]*mu[0]*uq[1]+n[1]*mu[0]*uq[2];\n  fb[2] = fb[1];')
    model.write_text(s)
PY
  scratch_prefix=$work/prefix
  cp -R "$EXASIM_PREFIX" "$scratch_prefix"
  cat > "$work/interaction3.txt" <<'EOF'
domains = [1,2];
type = 1;
interface1 = 2;
condition1 = 1;
fluxmap1 = [1];
interface2 = 4;
condition2 = 1;
fluxmap2 = [1];
domains = [2,3];
type = 1;
interface1 = 2;
condition1 = 1;
fluxmap1 = [1];
interface2 = 4;
condition2 = 1;
fluxmap2 = [1];
EOF
  cp "$work/interaction3.txt" "$work/interaction4.txt"
  cat >> "$work/interaction4.txt" <<'EOF'
domains = [3,4];
type = 1;
interface1 = 2;
condition1 = 1;
fluxmap1 = [1];
interface2 = 4;
condition2 = 1;
fluxmap2 = [1];
EOF

  check_guard() {
    local name=$1 ranks=$2
    shift 2
    local args=() domain=1
    for count in "$@"; do
      args+=("$work/d$domain/pdeapp.txt" "$count")
      domain=$((domain + 1))
    done
    set +e
    local interactions="$work/interaction$#.txt"
    EXASIM_PREFIX="$scratch_prefix" "$mpiexec" -np "$ranks" "$EXASIMAPP" --interactions "$interactions" \
      "${args[@]}" >"$work/guard-$name.log" 2>&1
    rc=$?
    set -e
    [[ $rc -ne 0 ]] || { echo "FAIL: $name graph input solved instead of reaching the Phase 1 guard"; exit 1; }
    count=$(grep -c 'generalized non-matching interactions are not active until Phase 4' "$work/guard-$name.log" || true)
    [[ $count -eq $ranks ]] || {
      cat "$work/guard-$name.log"
      echo "FAIL: $name produced $count guard lines for $ranks ranks"
      exit 1
    }
  }
  check_guard three-equal 3 1 1 1
  check_guard four-a 7 1 2 3 1
  check_guard four-b 7 3 1 2 1

  solver_args=(4
    "$work/d1/datain1/" "$work/d1/dataout1/out"
    "$work/d2/datain2/" "$work/d2/dataout2/out"
    "$work/d3/datain3/" "$work/d3/dataout3/out"
    "$work/d4/datain4/" "$work/d4/dataout4/out")
  "$mpiexec" -np 7 "$layout" --parse "${solver_args[@]}" 0 \
    --domain-ranks 3,1,2,1 >"$work/parse.log" 2>&1
  grep -q 'PASS: production ParseInputs' "$work/parse.log"
  "$mpiexec" -np 7 "$layout" --parse-twice "${solver_args[@]}" \
    --domain-ranks 3,1,2,1 >"$work/lifecycle.log" 2>&1
  grep -q 'PASS: production ParseInputs communicator lifecycle' "$work/lifecycle.log"

  expect_parse_failure() {
    local name=$1 expected=$2
    shift 2
    set +e
    "$mpiexec" -np 7 "$layout" --parse "${solver_args[@]}" "$@" \
      >"$work/parse-$name.log" 2>&1
    rc=$?
    set -e
    [[ $rc -ne 0 ]] || { echo "FAIL: production CLI accepted $name"; exit 1; }
    grep -q -- "$expected" "$work/parse-$name.log" || {
      cat "$work/parse-$name.log"
      echo "FAIL: production CLI $name missed diagnostic: $expected"
      exit 1
    }
  }
  expect_parse_failure duplicate 'Duplicate --domain-ranks option' \
    --domain-ranks 3,1,2,1 --domain-ranks 3,1,2,1
  expect_parse_failure missing 'Missing value for --domain-ranks' --domain-ranks
  expect_parse_failure extra 'Unexpected positional argument with --domain-ranks' \
    --domain-ranks 3,1,2,1 extra
  expect_parse_failure parameter-coexist '--parameter-cases requires physicsparamcases.bin' \
    0 --domain-ranks 3,1,2,1 --parameter-cases 1

  encoded_args=(103
    "$work/d1/datain1/" "$work/d1/dataout1/out"
    "$work/d2/datain2/" "$work/d2/dataout2/out")
  set +e
  "$mpiexec" -np 7 "$layout" --parse "${encoded_args[@]}" \
    --domain-ranks 3,4 >"$work/parse-encoded.log" 2>&1
  rc=$?
  set -e
  [[ $rc -ne 0 ]] || { echo 'FAIL: encoded form accepted --domain-ranks'; exit 1; }
  grep -q -- '--domain-ranks cannot be used with the encoded two-domain form' \
    "$work/parse-encoded.log"

  cp "$work/d1/pdeapp.txt" "$work/misuse.txt"
  # Keep both graph keys after another statement on one physical line.  The
  # launcher must parse exact semicolon-delimited keys, not just the first key
  # on a line and not substrings in values or comments.
  printf '\nphase2dummy = 7; interactionfile = "%s"; numdomains = 3;\n' \
    "$work/interaction3.txt" >> "$work/misuse.txt"
  for form in single coupled; do
    set +e
    if [[ $form == single ]]; then
      EXASIM_PREFIX="$scratch_prefix" "$mpiexec" -np 3 "$EXASIMAPP" "$work/misuse.txt" 3 >"$work/$form.log" 2>&1
    else
      EXASIM_PREFIX="$scratch_prefix" "$mpiexec" -np 3 "$EXASIMAPP" "$work/misuse.txt" 1 "$work/d2/pdeapp.txt" 2 >"$work/$form.log" 2>&1
    fi
    rc=$?
    set -e
    [[ $rc -ne 0 ]] || { echo "FAIL: old $form form accepted graph metadata"; exit 1; }
    grep -q 'generalized interactions require RunText2CodeApp --interactions' "$work/$form.log" || {
      cat "$work/$form.log"; echo "FAIL: old $form form missed graph diagnostic"; exit 1; }
  done
else
  echo 'SKIP: set EXASIMAPP and EXASIM_PREFIX for the real graph-guard run'
  exit 77
fi

echo 'PASS: Phase 2 rank layouts and real graph guard'
