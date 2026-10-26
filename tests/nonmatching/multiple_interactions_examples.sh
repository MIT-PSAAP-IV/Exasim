#!/usr/bin/env bash
# Reviewer-owned Phase 5 smoke/measurement driver. No numerical tolerance gate.
set -euo pipefail
root=${EXASIM_ROOT:?EXASIM_ROOT not set}
prefix=${EXASIM_PREFIX:?EXASIM_PREFIX must name the approved current-tree install}
matlab=${MATLAB:-matlab}
python=${PYTHON3:-python3}
points=${PHASE5_POINTS:-"1:2 2:2"}
command -v "$matlab" >/dev/null || { echo "SKIP: MATLAB unavailable"; exit 77; }
command -v "$python" >/dev/null || { echo "SKIP: Python unavailable"; exit 77; }
[[ -x $prefix/bin/text2code ]] || { echo "SKIP: current Text2Code unavailable"; exit 77; }

scratch=$(mktemp -d /tmp/exasim-phase5-examples-work.XXXXXX)
artifact=${PHASE5_ARTIFACT_DIR:-$(mktemp -d /tmp/exasim-phase5-examples-artifacts.XXXXXX)}
[[ $artifact == /* && $artifact != "$root" && $artifact != "$root/"* ]] || {
  echo "FAIL: PHASE5_ARTIFACT_DIR must be absolute and outside the repository"; exit 1; }
mkdir -p "$artifact"
trap 'rm -rf "$scratch"' EXIT
echo "ARTIFACT_DIR=$artifact"

hygiene_before="$scratch/source-status.before"
hygiene_after="$scratch/source-status.after"
git -C "$root" status --porcelain --ignored --untracked-files=all -- \
  examples apps tests frontends | LC_ALL=C sort >"$hygiene_before"

check_manifest() {
  local manifest=$1
  PYTHONDONTWRITEBYTECODE=1 "$python" -B - "$manifest" <<'PY'
import array, json, math, pathlib, sys
manifest_path=pathlib.Path(sys.argv[1])
manifest=json.loads(manifest_path.read_text())
required={
    "case","dimension","families","N","porder","pgauss","domain_ranks",
    "interaction_table","object_counts","element_counts","interface_counts",
    "per_rank_interface_counts","per_domain_l2","combined_l2","max_nodal_error",
    "newton_history","final_residual","newton_iterations","gmres_iterations",
    "solver_controls",
}
missing=sorted(required-manifest.keys())
if missing:
    raise SystemExit(f"FAIL: manifest misses {missing}")
golden=[1,4,3,
        11,0,1,1,2,1,1,0,4,2,1,0,
        11,1,2,1,2,2,1,0,4,1,1,0,
        11,2,3,1,2,1,1,0,4,2,1,0]
if manifest.get("interaction_table") != golden:
    raise SystemExit("FAIL: manifest interaction_table is not the canonical 39-value table")
if manifest.get("object_counts") != [1,2,2,1]:
    raise SystemExit("FAIL: manifest object_counts differ from [1,2,2,1]")
if not all(len(manifest[key])==4 for key in (
        "families","domain_ranks","object_counts","element_counts",
        "interface_counts","per_rank_interface_counts","per_domain_l2",
        "newton_history","final_residual","newton_iterations")):
    raise SystemExit("FAIL: manifest has a non-four-domain field")
nltol=float(manifest["solver_controls"]["NLtol"])
for domain,history in enumerate(manifest["newton_history"]):
    if not history or any(len(row)!=4 for row in history):
        raise SystemExit(f"FAIL: domain {domain+1} Newton history is not nonempty Nx4")
    if not all(math.isfinite(float(value)) for row in history for value in row):
        raise SystemExit(f"FAIL: domain {domain+1} Newton history is non-finite")
    if float(manifest["final_residual"][domain]) != float(history[-1][3]):
        raise SystemExit(f"FAIL: domain {domain+1} final residual disagrees with history")
    if int(manifest["newton_iterations"][domain]) != int(history[-1][2]):
        raise SystemExit(f"FAIL: domain {domain+1} Newton count disagrees with history")
    if float(history[-1][3]) > nltol:
        raise SystemExit(f"FAIL: domain {domain+1} did not reach NLtol")
if len(manifest["gmres_iterations"]) != sum(map(int,manifest["newton_iterations"])):
    raise SystemExit("FAIL: GMRES record count differs from Newton linear-solve count")
for key in ("per_domain_l2","combined_l2","max_nodal_error","final_residual"):
    values=manifest[key] if isinstance(manifest[key],list) else [manifest[key]]
    if not all(math.isfinite(float(value)) for value in values):
        raise SystemExit(f"FAIL: manifest {key} contains NaN or Inf")
streams={}
for filename in ("phase5_solution.bin","phase5_coordinates.bin", "phase5_element_centroids.bin"):
    raw=(manifest_path.parent/filename).read_bytes()
    if not raw or len(raw)%8:
        raise SystemExit(f"FAIL: {filename} is empty or not a double stream")
    values=array.array("d"); values.frombytes(raw)
    if not all(math.isfinite(value) for value in values):
        raise SystemExit(f"FAIL: {filename} contains NaN or Inf")
    streams[filename]=len(values)
expected=int(manifest["dimension"])*streams["phase5_solution.bin"]
if streams["phase5_coordinates.bin"] != expected or streams["phase5_element_centroids.bin"] != expected:
    raise SystemExit("FAIL: solution, coordinate, and centroid stream sizes disagree")
for key in ("maximum_conditions", "max_conditions", "maximum_condition_by_domain"):
    if key in manifest and manifest[key] != [1,2,1,2]:
        raise SystemExit(f"FAIL: manifest {key} differs from [1,2,1,2]")
PY
}

cases=(multiple_interactions_2d multiple_interactions_2d_quadtri multiple_interactions_3d multiple_interactions_3d_hextet)
for name in "${cases[@]}"; do
  cp -R "$root/examples/CoupledProblems/Poisson/$name" "$scratch/$name"
  for point in $points; do
    p=${point%%:*}; n=${point##*:}
    [[ $p =~ ^[1-9][0-9]*$ && $n =~ ^[1-9][0-9]*$ ]] || {
      echo "FAIL: invalid PHASE5_POINTS entry $point (expected p:N)"; exit 1; }
    label="$name-p${p}-N${n}"
    outdir=$(mktemp -d "/tmp/exasim-phase5-$label.XXXXXX")
    "$matlab" -batch \
      "addpath('$scratch/$name'); r=run_multiple_interactions($p,$n,[1 1 1 1],'$outdir');" \
      >"$artifact/$label.matlab.log" 2>&1
    resultdir=$(sed -n 's/^PHASE5_OUTDIR=//p' "$artifact/$label.matlab.log" | tail -1)
    [[ $resultdir == "$outdir" ]] || {
      cat "$artifact/$label.matlab.log"; echo "FAIL: $label reported the wrong output directory"; exit 1; }
    [[ -n $resultdir && -s $resultdir/phase5_manifest.json && \
       -s $resultdir/phase5_solution.bin && -s $resultdir/phase5_coordinates.bin && \
       -s $resultdir/phase5_element_centroids.bin && -s $resultdir/solver.log ]] || {
      cat "$artifact/$label.matlab.log"; echo "FAIL: $label produced incomplete artifacts"; exit 1; }
    check_manifest "$resultdir/phase5_manifest.json"
    mixed=()
    if [[ $name == multiple_interactions_3d_hextet ]]; then mixed=(--mixed); fi
    PYTHONDONTWRITEBYTECODE=1 "$python" -B \
      "$root/tests/nonmatching/analyze_multiple_interactions_log.py" \
      "$resultdir/solver.log" "$artifact/$label.donor.json" "${mixed[@]}"

    # Physical edge attribution cannot be recovered from the intact log:
    # edges 0 and 2 have identical face/point signatures and MPI stdout may
    # interleave. Run each edge alone through the diagnostic-only fifth
    # MATLAB argument and validate its two directions by content.
    for edge in 0 1 2; do
      active=(0 0 0); active[$edge]=1
      isolated=$(mktemp -d "/tmp/exasim-phase5-$label-edge$edge.XXXXXX")
      "$matlab" -batch \
        "addpath('$scratch/$name'); run_multiple_interactions($p,$n,[1 1 1 1],'$isolated',[$(printf '%s ' "${active[@]}")]);" \
        >"$artifact/$label.edge$edge.matlab.log" 2>&1
      reported=$(sed -n 's/^PHASE5_OUTDIR=//p' "$artifact/$label.edge$edge.matlab.log" | tail -1)
      [[ $reported == "$isolated" && -s $isolated/solver.log ]] || {
        cat "$artifact/$label.edge$edge.matlab.log"
        echo "FAIL: isolated donor run $label edge $edge is incomplete"; exit 1; }
      PYTHONDONTWRITEBYTECODE=1 "$python" -B \
        "$root/tests/nonmatching/analyze_multiple_interactions_log.py" \
        "$isolated/solver.log" "$artifact/$label.edge$edge.donor.json" \
        --expected-edge "$edge" "${mixed[@]}"
    done
    cp "$resultdir/phase5_manifest.json" "$artifact/$label.manifest.json"
    cp "$resultdir/solver.log" "$artifact/$label.solver.log"
    echo "MEASURED $label manifest=$artifact/$label.manifest.json"
  done
done

# Determinism check on the smallest graph case.  The manifest contains every
# Newton-history value but no timing or rank-ordered diagnostic text, so an
# exact comparison ignores only the two explicitly nondeterministic classes.
repeat_reference=
for repeat in 1 2 3; do
  label="multiple_interactions_2d-repeat-$repeat"
  outdir=$(mktemp -d "/tmp/exasim-phase5-$label.XXXXXX")
  "$matlab" -batch \
    "addpath('$scratch/multiple_interactions_2d'); run_multiple_interactions(1,2,[1 1 1 1],'$outdir');" \
    >"$artifact/$label.matlab.log" 2>&1
  resultdir=$(sed -n 's/^PHASE5_OUTDIR=//p' "$artifact/$label.matlab.log" | tail -1)
  [[ $resultdir == "$outdir" ]] || {
    cat "$artifact/$label.matlab.log"; echo "FAIL: repeat $repeat reported wrong output directory"; exit 1; }
  [[ -n $resultdir && -s $resultdir/phase5_manifest.json && -s $resultdir/phase5_solution.bin ]] || {
    cat "$artifact/$label.matlab.log"; echo "FAIL: deterministic repeat $repeat is incomplete"; exit 1; }
  check_manifest "$resultdir/phase5_manifest.json"
  mkdir -p "$artifact/$label"
  cp "$resultdir/phase5_manifest.json" "$resultdir/phase5_solution.bin" \
     "$resultdir/phase5_coordinates.bin" "$resultdir/phase5_element_centroids.bin" \
     "$resultdir/solver.log" "$artifact/$label/"
  if [[ -z $repeat_reference ]]; then
    repeat_reference="$artifact/$label"
  else
    for file in phase5_manifest.json phase5_solution.bin phase5_coordinates.bin phase5_element_centroids.bin; do
      cmp "$repeat_reference/$file" "$artifact/$label/$file" || {
        echo "FAIL: deterministic repeat differs in $file"; exit 1; }
    done
  fi
done
echo "STRUCTURE PASS: three deterministic MATLAB repeats (timing and rank-output order excluded)"

# Build and run every Text2Code application from a copied source directory.
# These are structural runs: table/file/layout/search completeness is exact;
# numerical comparisons wait for the tolerance card.
for name in "${cases[@]}"; do
  appsrc="$scratch/text2code-$name"
  cp -R "$root/apps/coupledproblems/poisson/$name" "$appsrc"
  build="$scratch/build-$name"
  cmake -S "$appsrc" -B "$build" -DCMAKE_PREFIX_PATH="$prefix" \
    -DEXASIM_TEXT2CODE_EXECUTABLE="$prefix/bin/text2code" \
    >"$artifact/$name.configure.log" 2>&1
  cmake --build "$build" --parallel >"$artifact/$name.build.log" 2>&1
  layout_specs=("equal:1,1,1,1")
  if [[ $name == multiple_interactions_2d || $name == multiple_interactions_2d_quadtri ]]; then
    layout_specs+=("unequal1221:1,2,2,1")
  else
    layout_specs+=("unequal1221:1,2,2,1" "unequal1231:1,2,3,1")
  fi
  for layout_spec in "${layout_specs[@]}"; do
    layout=${layout_spec%%:*}; rank_csv=${layout_spec#*:}
    run="$scratch/run-$name-$layout"
    cp -R "$appsrc" "$run"
    IFS=, read -r -a ranks <<<"$rank_csv"
    np=$((ranks[0]+ranks[1]+ranks[2]+ranks[3]))
    (
      cd "$run"
      mpirun -np "$np" "$build/exasimapp" --interactions "$run/interaction.txt" \
        "$run/pdeapp1.txt" "${ranks[0]}" "$run/pdeapp2.txt" "${ranks[1]}" \
        "$run/pdeapp3.txt" "${ranks[2]}" "$run/pdeapp4.txt" "${ranks[3]}"
    ) >"$artifact/$name.$layout.text2code.log" 2>&1
    for domain in 1 2 3 4; do
      [[ -s $run/datain$domain/app.bin ]] || { echo "FAIL: missing datain$domain/app.bin"; exit 1; }
    done
    [[ -s $run/datain1/mesh1.bin && -s $run/datain1/sol1.bin && \
       -s $run/datain4/mesh1.bin && -s $run/datain4/sol1.bin ]] || {
      echo "FAIL: one-rank graph domains lack numbered mesh1.bin/sol1.bin"; exit 1; }
    PYTHONDONTWRITEBYTECODE=1 "$python" -B - "$run" <<'PY'
import pathlib, struct, sys
root=pathlib.Path(sys.argv[1])
golden=(1,4,3,11,0,1,1,2,1,1,0,4,2,1,0,
        11,1,2,1,2,2,1,0,4,1,1,0,
        11,2,3,1,2,1,1,0,4,2,1,0)
for domain in range(1,5):
    raw=(root/f"datain{domain}"/"app.bin").read_bytes()
    if len(raw)%8: raise SystemExit("FAIL: app.bin is not a double stream")
    values=struct.unpack(f"{len(raw)//8}d",raw)
    lsize=int(values[0]); nsize=[int(x) for x in values[1:1+lsize]]
    if nsize[24]!=len(golden) or values[-len(golden):]!=golden:
        raise SystemExit(f"FAIL: domain {domain} interaction table differs")
PY
    mixed=()
    if [[ $name == multiple_interactions_3d_hextet ]]; then mixed=(--mixed); fi
    PYTHONDONTWRITEBYTECODE=1 "$python" -B \
      "$root/tests/nonmatching/analyze_multiple_interactions_log.py" \
      "$artifact/$name.$layout.text2code.log" "$artifact/$name.$layout.donor.json" \
      "${mixed[@]}"
  done
done

git -C "$root" status --porcelain --ignored --untracked-files=all -- \
  examples apps tests frontends | LC_ALL=C sort >"$hygiene_after"
if ! cmp -s "$hygiene_before" "$hygiene_after"; then
  diff -u "$hygiene_before" "$hygiene_after" || true
  echo "FAIL: Phase 5 examples harness changed repository files"; exit 1
fi

echo "STRUCTURE PASS: four MATLAB and Text2Code graph examples; artifacts retained at $artifact"
