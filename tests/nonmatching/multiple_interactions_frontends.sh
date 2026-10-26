#!/usr/bin/env bash
# Phase 1 cross-frontend validator and canonical-table parity test.
set -euo pipefail

root=${EXASIM_ROOT:?EXASIM_ROOT not set}
python=${PYTHON3:-python3}
matlab=${MATLAB:-matlab}
julia=${JULIA:-julia}
for command in "$python" "$matlab" "$julia"; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "SKIP: required frontend interpreter is unavailable: $command"
    exit 77
  fi
done
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
artifact=${PHASE5_ARTIFACT_DIR:-$(mktemp -d /tmp/exasim-phase5-frontends-artifacts.XXXXXX)}
[[ $artifact == /* && $artifact != "$root" && $artifact != "$root/"* ]] || {
  echo "FAIL: PHASE5_ARTIFACT_DIR must be absolute and outside the repository"; exit 1; }
mkdir -p "$artifact"
echo "ARTIFACT_DIR=$artifact"
hygiene_before="$work/source-status.before"
hygiene_after="$work/source-status.after"
git -C "$root" status --porcelain --ignored --untracked-files=all -- \
  examples apps tests frontends | LC_ALL=C sort >"$hygiene_before"

cat > "$work/phase1_model.m" <<'EOF'
function pde=phase1_model
pde.fint=@fint;
end
function f=fint(varargin)
f=0;
end
EOF
cat > "$work/phase1_no_fint.m" <<'EOF'
function pde=phase1_no_fint
pde.flux=@flux;
end
function f=flux(varargin)
f=0;
end
EOF

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root/frontends/Python${PYTHONPATH:+:$PYTHONPATH}" \
  "$python" -B "$root/tests/nonmatching/multiple_interactions_frontends.py" \
  "$root" "$work/python.bin"
"$matlab" -batch "addpath('$work'); addpath('$root/tests/nonmatching'); multiple_interactions_frontends('$root','$work/matlab.bin');"
JULIA_DEPOT_PATH="$work/julia-depot:${JULIA_DEPOT_PATH:-$HOME/.julia}" \
  "$julia" --project="$root/frontends/Julia/Exasim" \
  "$root/tests/nonmatching/multiple_interactions_frontends.jl" "$root" "$work/julia.bin"

cmp "$work/matlab.bin" "$work/python.bin"
cmp "$work/julia.bin" "$work/python.bin"
bytes=$(wc -c < "$work/python.bin" | tr -d ' ')
[[ $bytes -eq 216 ]] || { echo "FAIL: golden table is $bytes bytes, expected 216"; exit 1; }
echo "PASS: MATLAB, Python, and Julia validation rules and 27-value table parity"

# Phase 5 matched-solve measurements are reviewer-owned and deliberately opt-in:
# they require a current disposable install and are not part of the short Phase 1
# validator. They compare structure exactly and print numerical scales without a
# tolerance gate until the maintainer approves the Phase 5 tolerance card.
if [[ ${PHASE5_FRONTEND_SOLVES:-0} == 1 ]]; then
  : "${EXASIM_PREFIX:?EXASIM_PREFIX must name the approved disposable Phase 5 install}"
  result_dirs=()
  for name in multiple_interactions_2d multiple_interactions_3d_hextet; do
    cp -R "$root/examples/CoupledProblems/Poisson/$name" "$work/$name"
    mdir=$(mktemp -d "/tmp/exasim-phase5-$name-matlab.XXXXXX")
    pdir=$(mktemp -d "/tmp/exasim-phase5-$name-python.XXXXXX")
    jdir=$(mktemp -d "/tmp/exasim-phase5-$name-julia.XXXXXX")
    "$matlab" -batch \
      "addpath('$work/$name'); run_multiple_interactions(2,2,[1 1 1 1],'$mdir');" \
      >"$work/$name-matlab.log" 2>&1
    reported=$(sed -n 's/^PHASE5_OUTDIR=//p' "$work/$name-matlab.log" | tail -1)
    [[ $reported == "$mdir" ]] || {
      cat "$work/$name-matlab.log"; echo "FAIL: MATLAB reported the wrong output directory"; exit 1; }
    EXASIM_ROOT="$root" PYTHONDONTWRITEBYTECODE=1 "$python" -B \
      "$work/$name/run_multiple_interactions.py" \
      --porder 2 --N 2 --ranks 1,1,1,1 --outdir "$pdir" \
      >"$work/$name-python.log" 2>&1
    reported=$(sed -n 's/^PHASE5_OUTDIR=//p' "$work/$name-python.log" | tail -1)
    [[ $reported == "$pdir" ]] || {
      cat "$work/$name-python.log"; echo "FAIL: Python reported the wrong output directory"; exit 1; }
    mkdir -p "$jdir/julia-depot"
    EXASIM_ROOT="$root" JULIA_DEPOT_PATH="$jdir/julia-depot:${JULIA_DEPOT_PATH:-$HOME/.julia}" \
      "$julia" --project="$root/frontends/Julia/Exasim" \
      "$work/$name/run_multiple_interactions.jl" 2 2 1,1,1,1 "$jdir" \
      >"$work/$name-julia.log" 2>&1
    reported=$(sed -n 's/^PHASE5_OUTDIR=//p' "$work/$name-julia.log" | tail -1)
    [[ $reported == "$jdir" ]] || {
      cat "$work/$name-julia.log"; echo "FAIL: Julia reported the wrong output directory"; exit 1; }
    for directory in "$mdir" "$pdir" "$jdir"; do
      for file in phase5_manifest.json phase5_solution.bin phase5_coordinates.bin \
                  phase5_element_centroids.bin solver.log; do
        [[ -s $directory/$file ]] || {
          echo "FAIL: missing or empty $directory/$file"; exit 1; }
      done
    done
    result_dirs+=("$mdir" "$pdir" "$jdir")
    for entry in "matlab:$mdir" "python:$pdir" "julia:$jdir"; do
      language=${entry%%:*}; directory=${entry#*:}; destination="$artifact/$name-$language"
      mkdir -p "$destination"
      for file in phase5_manifest.json phase5_solution.bin phase5_coordinates.bin phase5_element_centroids.bin solver.log; do
        [[ -f $directory/$file ]] && cp "$directory/$file" "$destination/$file"
      done
    done
    cp "$work/$name-matlab.log" "$artifact/$name-matlab-driver.log"
    cp "$work/$name-python.log" "$artifact/$name-python-driver.log"
    cp "$work/$name-julia.log" "$artifact/$name-julia-driver.log"
  done
  PYTHONDONTWRITEBYTECODE=1 "$python" -B \
    "$root/tests/nonmatching/compare_multiple_interactions_frontends.py" "${result_dirs[@]}" \
    | tee "$artifact/frontend-comparison.log"
  echo "MEASUREMENTS retained at $artifact"
fi

git -C "$root" status --porcelain --ignored --untracked-files=all -- \
  examples apps tests frontends | LC_ALL=C sort >"$hygiene_after"
if ! cmp -s "$hygiene_before" "$hygiene_after"; then
  diff -u "$hygiene_before" "$hygiene_after" || true
  echo "FAIL: frontend harness changed repository files"
  exit 1
fi
