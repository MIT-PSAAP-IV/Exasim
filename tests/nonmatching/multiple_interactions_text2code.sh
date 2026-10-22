#!/usr/bin/env bash
# Phase 1 generalized-interaction Text2Code contract. This generates code only;
# it does not build or run an Exasim solver.
set -euo pipefail

root=${EXASIM_ROOT:?EXASIM_ROOT not set}
prefix=${EXASIM_INSTALL:?EXASIM_INSTALL not set}
text2code=${TEXT2CODE:-$prefix/bin/text2code}
cxx=${CXX:-c++}
artifacts=${EXASIM_PHASE0_ARTIFACTS:-$root/../exasim-artifacts/multiphysics-phase0}
fixture=$root/apps/coupledproblems/poisson/nonmatching
if [[ ! -x $text2code ]]; then
  echo "SKIP: text2code is unavailable at $text2code"
  exit 77
fi
if [[ ! -d $artifacts/text2code-generated/r1/generated1 ]]; then
  echo "SKIP: Phase 0 Text2Code artifacts are unavailable at $artifacts"
  exit 77
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/graph" "$work/legacy" "$work/stub"
cp "$fixture/pdeapp1.txt" "$fixture/pdemodel1.txt" "$fixture/grid1.bin" "$work/graph/"
cp "$fixture/pdeapp1.txt" "$fixture/pdemodel1.txt" "$fixture/grid1.bin" "$work/legacy/"

python3 - "$work/graph/pdeapp1.txt" "$work/graph/pdemodel1.txt" <<'PY'
from pathlib import Path
import re, sys
pdeapp = Path(sys.argv[1]); model = Path(sys.argv[2])
text = pdeapp.read_text()
text = re.sub(r'^modelnumber\s*=.*$', 'modelnumber = 1;', text, flags=re.M)
text = re.sub(r'^builtinmodelID\s*=.*$', 'builtinmodelID = 0;', text, flags=re.M)
text = re.sub(r'^ncw\s*=.*$', 'ncw = 1;', text, flags=re.M)
text = re.sub(r'^interfaceconditions\s*=.*$', 'interfaceconditions = [];', text, flags=re.M)
text = re.sub(r'^interfacefluxmap\s*=.*$', 'interfacefluxmap = [];', text, flags=re.M)
text += '\ngendatain = 0;\ngencode = 1;\nnumdomains = 2;\n'
text += 'interactionfile = "stgib_physicsparam_interfaceconditions.txt";\n'
pdeapp.write_text(text)

text = model.read_text().replace('w(0)', 'w(1)')
text = re.sub(
    r'function Fint\(.*?\nend\n',
    'function Fint(x, uq, v, w, uhat, n, tau, eta, mu, t)\n'
    '  output_size(fi) = 2;\n'
    '  fi[0] = 2*uq[0] + 3*w[0] + 5*uhat[0];\n'
    '  fi[1] = 7*uq[0]*uq[0] + 11*w[0]*w[0] + 13*uhat[0]*uhat[0];\n'
    'end\n', text, flags=re.S)
model.write_text(text)
PY

write_valid_interaction() {
  cat > "$work/graph/stgib_physicsparam_interfaceconditions.txt" <<'EOF'
domains = [1,2];
type = 1;
interface1 = 2;
condition1 = 2;
fluxmap1 = [1];
interface2 = 4;
condition2 = 1;
fluxmap2 = [1];
EOF
}
write_valid_interaction
run_t2c() {
  local directory=$1 log=$2
  shift 2
  set +e
  (cd "$directory" && EXASIM_PREFIX="$prefix" "$text2code" "$@" --gen-only) > "$log" 2>&1
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "FAIL: text2code command exited $rc: $*"
    cat "$log"
    exit "$rc"
  fi
}

# EXASIM_INSTALL is only the template prefix known to text2code. --gen-only
# prevents this harness from compiling or replacing libt2cmodel in that prefix.
run_t2c "$work/graph" "$work/graph/text2code.log" \
  pdeapp1.txt --out-dir "$work/generated"
header=$work/generated/my_model.hpp
[[ -f $header ]] || { cat "$work/graph/text2code.log"; echo "FAIL: my_model.hpp was not generated"; exit 1; }

cat > "$work/stub/Kokkos_Core.hpp" <<'EOF'
#pragma once
#include <cmath>
#define KOKKOS_INLINE_FUNCTION inline
namespace Kokkos {
using std::abs; using std::acos; using std::cos; using std::exp; using std::log;
using std::pow; using std::sin; using std::sqrt; using std::tan; using std::tanh;
}
EOF
"$cxx" -std=c++17 -I"$work/stub" -I"$root" \
  -DGENERATED_MODEL_HEADER=\"$header\" \
  "$root/tests/unit/nonmatching_text2code_conditions.cpp" -o "$work/check_conditions"
"$work/check_conditions"

# Exercise the production writer and backend reader/writer with the same table.
sed -i.bak -e 's/^gendatain = 0;/gendatain = 1;/' -e 's/^gencode = 1;/gencode = 0;/' \
  "$work/graph/pdeapp1.txt" && rm -f "$work/graph/pdeapp1.txt.bak"
run_t2c "$work/graph" "$work/graph/datain.log" \
  pdeapp1.txt --out-dir "$work/datain-only"
appbin=$(find "$work/graph" -path '*/datain*/app.bin' -type f | head -1)
[[ -n $appbin ]] || { cat "$work/graph/datain.log"; echo "FAIL: graph app.bin was not written"; exit 1; }
python3 - "$appbin" <<'PY'
import struct,sys
raw=open(sys.argv[1],'rb').read(); values=struct.unpack(f'{len(raw)//8}d',raw)
lsize=int(values[0]); nsize=[int(v) for v in values[1:1+lsize]]
golden=(1,2,1,11,0,1,1,2,2,1,0,4,1,1,0)
assert nsize[24]==len(golden), (nsize[24],len(golden))
assert values[-len(golden):]==golden, values[-len(golden):]
PY
roundtrip=${EXASIM_BUILD:-}/tests/unit_nonmatching_interaction_table
if [[ -x $roundtrip ]]; then
  "$roundtrip" "$appbin" "$work/app-roundtrip.bin"
else
  echo "SKIP: app read/write executable unavailable (set EXASIM_BUILD)"
  exit 77
fi

# Phase 2: the graph selector union must reproduce the legacy scalar
# partition byte-for-byte for a one-selector graph at one and three ranks.
for ranks in 1 3; do
  scalar_dir="$work/selector-scalar-$ranks"
  graph_dir="$work/selector-graph-$ranks"
  mkdir -p "$scalar_dir" "$graph_dir"
  cp "$fixture/pdeapp1.txt" "$fixture/pdemodel1.txt" "$fixture/grid1.bin" "$scalar_dir/"
  # Both sides use the same untouched physical model.  The earlier
  # two-condition fixture deliberately changes ncw, Fint, and builtinmodelID,
  # so it is not a valid partition-only reference.
  cp "$fixture/pdeapp1.txt" "$fixture/pdemodel1.txt" "$fixture/grid1.bin" "$graph_dir/"
  cat > "$graph_dir/interaction.txt" <<'EOF'
domains = [1,2];
type = 1;
interface1 = 2;
condition1 = 1;
fluxmap1 = [1];
interface2 = 4;
condition2 = 1;
fluxmap2 = [1];
EOF
  python3 - "$scalar_dir/pdeapp1.txt" "$graph_dir/pdeapp1.txt" "$ranks" <<'PY'
from pathlib import Path
import re,sys
for index,name in enumerate(sys.argv[1:3]):
    path=Path(name); text=path.read_text()
    text=re.sub(r'^mpiprocs\s*=.*$', f'mpiprocs = {sys.argv[3]};', text, flags=re.M)
    text=re.sub(r'^gendatain\s*=.*$', 'gendatain = 1;', text, flags=re.M)
    text=re.sub(r'^gencode\s*=.*$', 'gencode = 0;', text, flags=re.M)
    if 'gendatain' not in text: text += '\ngendatain = 1;\n'
    if 'gencode' not in text: text += '\ngencode = 0;\n'
    if index == 1:
        text=re.sub(r'^interfaceconditions\s*=.*$', 'interfaceconditions = [];', text, flags=re.M)
        text=re.sub(r'^interfacefluxmap\s*=.*$', 'interfacefluxmap = [];', text, flags=re.M)
        text += '\nnumdomains = 2;\ninteractionfile = "interaction.txt";\n'
    path.write_text(text)
PY
  run_t2c "$scalar_dir" "$scalar_dir/text2code.log" pdeapp1.txt --out-dir "$work/scalar-generated-$ranks"
  run_t2c "$graph_dir" "$graph_dir/text2code.log" pdeapp1.txt --out-dir "$work/graph-generated-$ranks"
  for rank in $(seq 1 "$ranks"); do
    cmp "$scalar_dir/datain1/mesh$rank.bin" "$graph_dir/datain1/mesh$rank.bin"
    cmp "$scalar_dir/datain1/sol$rank.bin" "$graph_dir/datain1/sol$rank.bin"
  done
done

# Restore generation settings before the grammar cases below.
sed -i.bak -e 's/^gendatain = 1;/gendatain = 0;/' -e 's/^gencode = 0;/gencode = 1;/' \
  "$work/graph/pdeapp1.txt" && rm -f "$work/graph/pdeapp1.txt.bak"

# The filename deliberately contains legacy key names. Exact key dispatch must
# still parse it only as the interactionfile value.
grammar_case() {
  local name=$1 expected=$2
  shift 2
  printf '%s\n' "$@" > "$work/graph/stgib_physicsparam_interfaceconditions.txt"
  set +e
  (cd "$work/graph" && EXASIM_PREFIX="$prefix" "$text2code" pdeapp1.txt \
    --out-dir "$work/grammar-$name" --gen-only > "grammar-$name.log" 2>&1)
  local rc=$?
  set -e
  if [[ $rc -eq 0 ]] || ! grep -q "$expected" "$work/graph/grammar-$name.log"; then
    echo "FAIL: interaction.txt grammar case $name"; cat "$work/graph/grammar-$name.log"; exit 1
  fi
}
grammar_case missing "incomplete interaction record" \
  'domains = [1,2];' 'type = 1;'
grammar_case unknown "unknown, repeated, or out of order" \
  'domains = [1,2];' 'bogus = 1;' 'interface1 = 2;' 'condition1 = 2;' \
  'fluxmap1 = [1];' 'interface2 = 4;' 'condition2 = 1;' 'fluxmap2 = [1];'
grammar_case repeated "unknown, repeated, or out of order" \
  'domains = [1,2];' 'type = 1;' 'type = 1;' 'condition1 = 2;' \
  'fluxmap1 = [1];' 'interface2 = 4;' 'condition2 = 1;' 'fluxmap2 = [1];'
grammar_case order "unknown, repeated, or out of order" \
  'type = 1;' 'domains = [1,2];' 'interface1 = 2;' 'condition1 = 2;' \
  'fluxmap1 = [1];' 'interface2 = 4;' 'condition2 = 1;' 'fluxmap2 = [1];'
grammar_case plus "invalid integer" \
  'domains = [1,2];' 'type = +1;' 'interface1 = 2;' 'condition1 = 2;' \
  'fluxmap1 = [1];' 'interface2 = 4;' 'condition2 = 1;' 'fluxmap2 = [1];'
write_valid_interaction

run_t2c "$work/legacy" "$work/legacy/text2code.log" \
  pdeapp1.txt --out-dir "$work/legacy-generated"
for file in HdgFint.cpp HdgFintonly.cpp my_model.hpp; do
  cmp "$artifacts/text2code-generated/r1/generated1/$file" "$work/legacy-generated/$file" || {
    echo "FAIL: legacy generated $file differs from the Phase 0 artifact"; exit 1; }
done
echo "PASS: interaction grammar, selector partitions, two-condition generation, and legacy source identity"
