#!/usr/bin/env bash
# [nonmatching] Phase 9: text2code parses `interfacetype`, validates it and writes it
# to zero-based app.problem[34]. The exporters (exporttext2code.{m,py,jl}) write the
# key as `interfacetype = <0|1>;`, which is the form exercised here.
#
# Environment: EXASIM_ROOT (repo), EXASIM_INSTALL (install prefix with bin/text2code).
# Exit 0 pass, 1 fail, 77 skip.
set -uo pipefail

root=${EXASIM_ROOT:?EXASIM_ROOT not set}
prefix=${EXASIM_INSTALL:?EXASIM_INSTALL not set}
app=$root/apps/poisson/poisson2d
if [[ ! -x $prefix/bin/text2code ]] || ! command -v python3 >/dev/null; then
    echo "SKIP: installed text2code or python3 unavailable"
    exit 77
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
status=0

# Prints zero-based problem[34] and the problem length of an app.bin.
read_slot() {
    python3 - "$1" <<'PY'
import struct, sys
data = open(sys.argv[1], "rb").read()
v = struct.unpack(f"<{len(data)//8}d", data)
numsize = int(v[0]); nsize = [int(x) for x in v[1:1 + numsize]]
start = 1 + numsize + nsize[0] + nsize[1]
problem = v[start:start + nsize[2]]
print(f"{problem[34] if len(problem) > 34 else 'missing'} {len(problem)}")
PY
}

run_case() {  # name, line appended to pdeapp.txt ("" for none), expected slot value or "reject"
    local name=$1 line=$2 expected=$3 dir=$work/$1
    mkdir -p "$dir"
    cp "$app/pdeapp.txt" "$app/pdemodel.txt" "$app/grid.bin" "$dir/"
    [[ -f $app/xdg.bin ]] && cp "$app/xdg.bin" "$dir/"
    # Write datain/ but skip code generation: only the app.bin serialization is under test.
    sed -i.bak -e '/^gencode *=/d' -e '/^gendatain *=/d' "$dir/pdeapp.txt" && rm -f "$dir/pdeapp.txt.bak"
    printf 'gencode = 0;\ngendatain = 1;\n' >> "$dir/pdeapp.txt"
    [[ -n $line ]] && echo "$line" >> "$dir/pdeapp.txt"
    (cd "$dir" && EXASIM_PREFIX="$prefix" "$prefix/bin/text2code" pdeapp.txt > text2code.log 2>&1)
    local rc=$?
    if [[ $expected == reject ]]; then
        if [[ $rc -ne 0 ]] && grep -q "interfacetype" "$dir/text2code.log"; then
            echo "$name: PASS (rejected)"
        else
            echo "$name: FAIL (expected rejection, rc=$rc)"; tail -5 "$dir/text2code.log"; status=1
        fi
        return
    fi
    local bin
    bin=$(find "$dir" -name app.bin -path '*datain*' | head -1)
    if [[ $rc -ne 0 || -z $bin ]]; then
        echo "$name: FAIL (text2code rc=$rc, app.bin '${bin}')"; tail -20 "$dir/text2code.log"; status=1; return
    fi
    read -r value length < <(read_slot "$bin")
    if [[ $value == "$expected" && $length -eq 35 ]]; then
        echo "$name: PASS (problem[34]=$value, $length problem values)"
    else
        echo "$name: FAIL (problem[34]=$value, $length problem values; expected $expected and 35)"; status=1
    fi
}

run_case omitted "" 0.0
run_case explicit0 "interfacetype = 0;" 0.0
run_case explicit1 "interfacetype = 1;" 1.0
run_case invalid2 "interfacetype = 2;" reject
run_case invalidhalf "interfacetype = 0.5;" reject
[[ $status -eq 0 ]] && echo "PASS: text2code interfacetype" || echo "FAIL: text2code interfacetype"
exit $status
