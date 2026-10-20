#!/usr/bin/env bash
# [nonmatching] Phase 9 Part B: the installed text2code generates datain for a single-rank
# coupled model whose coupled interface is L-shaped (x = 1 and y = 1 of the unit square,
# one boundary expression), so the corner element carries two coupled faces and appears
# twice in mesh.inte. initializeDMD must write elempart as a permutation of 0..ne-1 with
# the unique interface elements last (the corner element once), elempartpts = {ne-n, n, 0}
# and intepartpts = {ne-n, n, 0, 0}. A one-rank coupled model writes the numbered files
# datain/sol1.bin and datain/mesh1.bin (read by the coupled solver, as MATLAB writes them);
# a one-rank uncoupled model keeps datain/sol.bin and datain/mesh.bin (amendment B.5).
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
cp "$app/pdeapp.txt" "$app/pdemodel.txt" "$app/grid.bin" "$work/"
sed -i.bak \
    -e '/^gencode *=/d' -e '/^gendatain *=/d' -e '/^xdgfile *=/d' \
    -e 's/^mpiprocs *=.*/mpiprocs = 1;/' \
    -e 's/^discretization *=.*/discretization = "hdg";/' \
    -e 's/^boundaryconditions *=.*/boundaryconditions = [1, 1, 2];/' \
    -e 's/^boundaryexpressions *=.*/boundaryexpressions = ["abs(y)<1e-8", "abs(x)<1e-8", "max(x,y)>1-1e-8"];/' \
    -e 's/^curvedboundaries *=.*/curvedboundaries = [0, 0, 0];/' \
    -e 's/^curvedboundaryexprs *=.*/curvedboundaryexprs = ["", "", ""];/' \
    -e 's/^interfaceconditions *=.*/interfaceconditions = [0, 0, 1];/' \
    "$work/pdeapp.txt" && rm -f "$work/pdeapp.txt.bak"
grep -q '^interfacefluxmap' "$work/pdeapp.txt" || echo 'interfacefluxmap = [1];' >> "$work/pdeapp.txt"
printf 'gencode = 0;\ngendatain = 1;\n' >> "$work/pdeapp.txt"

(cd "$work" && EXASIM_PREFIX="$prefix" "$prefix/bin/text2code" pdeapp.txt > text2code.log 2>&1)
rc=$?
if [[ $rc -ne 0 ]]; then
    echo "FAIL: text2code rc=$rc"; tail -20 "$work/text2code.log"; exit 1
fi
names_ok=1
for f in sol1.bin mesh1.bin; do
    [[ -f $work/datain/$f ]] || { echo "FAIL coupled one-rank model did not write datain/$f"; names_ok=0; }
done
for f in sol.bin mesh.bin; do
    [[ -e $work/datain/$f ]] && { echo "FAIL coupled one-rank model wrote unnumbered datain/$f"; names_ok=0; }
done
[[ $names_ok -eq 1 ]] && echo "PASS coupled one-rank model writes datain/sol1.bin and datain/mesh1.bin"
mesh=$work/datain/mesh1.bin

# Uncoupled one-rank model: unnumbered names are unchanged.
plain=$work/uncoupled
mkdir -p "$plain"
cp "$app/pdeapp.txt" "$app/pdemodel.txt" "$app/grid.bin" "$plain/"
sed -i.bak -e '/^gencode *=/d' -e '/^gendatain *=/d' -e '/^xdgfile *=/d' \
    -e 's/^mpiprocs *=.*/mpiprocs = 1;/' -e 's/^discretization *=.*/discretization = "hdg";/' \
    "$plain/pdeapp.txt" && rm -f "$plain/pdeapp.txt.bak"
printf 'gencode = 0;\ngendatain = 1;\n' >> "$plain/pdeapp.txt"
(cd "$plain" && EXASIM_PREFIX="$prefix" "$prefix/bin/text2code" pdeapp.txt > text2code.log 2>&1) \
    || { echo "FAIL: text2code (uncoupled) failed"; tail -20 "$plain/text2code.log"; exit 1; }
for f in sol.bin mesh.bin; do
    [[ -f $plain/datain/$f ]] || { echo "FAIL uncoupled one-rank model did not write datain/$f"; names_ok=0; }
done
for f in sol1.bin mesh1.bin; do
    [[ -e $plain/datain/$f ]] && { echo "FAIL uncoupled one-rank model wrote numbered datain/$f"; names_ok=0; }
done
[[ $names_ok -eq 1 ]] && echo "PASS uncoupled one-rank model writes datain/sol.bin and datain/mesh.bin"

python3 - "$work/grid.bin" "$mesh" <<'PY'
import struct, sys
def doubles(path):
    data = open(path, "rb").read()
    return struct.unpack(f"<{len(data)//8}d", data)

# grid.bin: [size(p) size(t) p(:) t(:)], p nd x np, t nve x ne (column-major).
g = doubles(sys.argv[1])
nd, npt, nve, ne = (int(v) for v in g[:4])
p = g[4:4 + nd*npt]
t = [int(v) for v in g[4 + nd*npt:4 + nd*npt + nve*ne]]
base = min(t)  # 0- or 1-based connectivity
on_l = [max(p[nd*i], p[nd*i + 1]) > 1 - 1e-8 for i in range(npt)]
# Quad elements: an element has a coupled face for every edge with both vertices on the L.
counts = []
for e in range(ne):
    v = [t[j + nve*e] - base for j in range(nve)]
    edges = [(v[j], v[(j + 1) % nve]) for j in range(nve)]
    counts.append(sum(1 for a, b in edges if on_l[a] and on_l[b]))
interface = sorted(e for e in range(ne) if counts[e] > 0)
corner = [e for e in range(ne) if counts[e] >= 2]

# mesh file (single rank): [len, nsize(len), ndims(nsize[0]), elempart(nsize[9]),
# elempartpts(nsize[10]), ...] with intepartpts last (nsize[28] values).
m = doubles(sys.argv[2])
ln = int(m[0]); nsize = [int(v) for v in m[1:1 + ln]]
k = 1 + ln + nsize[0]
assert sum(nsize[4:9]) == 0, "single-rank mesh file expected"
elempart = [int(v) for v in m[k:k + nsize[9]]]; k += nsize[9]
elempartpts = [int(v) for v in m[k:k + nsize[10]]]
intepartpts = [int(v) for v in m[len(m) - nsize[28]:]]

n = len(interface)
ok = True
def check(cond, what):
    global ok
    print(("PASS " if cond else "FAIL ") + what)
    ok = ok and cond
check(len(corner) >= 1, f"fixture has an element with two coupled faces ({len(corner)})")
check(sorted(elempart) == list(range(ne)), f"elempart is a permutation of 0..{ne-1}")
check(elempart[ne - n:] == interface, f"the {n} unique interface elements come last, ascending")
check(elempart[:ne - n] == [e for e in range(ne) if e not in set(interface)], "interior elements first, ascending")
check(elempartpts == [ne - n, n, 0], f"elempartpts {elempartpts} == {[ne - n, n, 0]}")
check(intepartpts == [ne - n, n, 0, 0], f"intepartpts {intepartpts} == {[ne - n, n, 0, 0]}")
sys.exit(0 if ok else 1)
PY
status=$?
[[ $names_ok -eq 1 ]] || status=1
[[ $status -eq 0 ]] && echo "PASS text2code_initializedmd_cli" || echo "FAIL text2code_initializedmd_cli"
exit $status
