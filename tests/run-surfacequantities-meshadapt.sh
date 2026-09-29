#!/usr/bin/env bash
# SurfaceQuantities on a MOVING mesh: the cylinder Mach 8 mesh-adaptivity app
# (apps/meshadaptivity/cylindermach8, mesh moved after each of the first nine AV
# continuation solves) with SurfaceQuantities = [rho, x.n] saved on ibs = [3, 5, 6].
#
# Gate: for every saved point, the value x.n computed by the kernel equals x.n from the
# outbousurfgeo record of the SAME save, i.e. the saved geometry is the geometry the values
# were evaluated on. The run also checks that the mesh did move (the start-up outbouxdg no
# longer matches), so the gate is not passing on a static mesh.
#
# Env (like run-sharedlibrary-test.sh): EXASIM_BUILD, INSTALL_PREFIX, KOKKOS_DIR, CXX.
# Exits 77 (ctest SKIP) when the installed text2code is unavailable.
set -euo pipefail

# Print the tail of a log on failure: the run dirs live under /tmp on the test machine,
# which CI users (e.g. Buildbot) cannot reach, so the reason must appear in the test output.
fail() {  # fail <message> <log>
  echo "FAIL: $1 (see $2)"
  # text2code and the compiler recurse deeply on this model's long generated expressions;
  # text2code raises the SOFT stack limit itself, but a low HARD limit cannot be raised.
  echo "stack limit: soft $(ulimit -S -s) KB, hard $(ulimit -H -s) KB"
  echo "--- last 60 lines of $2 ---"; tail -60 "$2" 2>/dev/null | sed 's/^/| /'; echo "--- end ---"
  exit 1
}

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXASIM_BUILD="${EXASIM_BUILD:-$REPO-build}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/tmp/exasim_surfq_meshadapt_install}"
KOKKOS_DIR="${KOKKOS_DIR:-$REPO/deps/kokkos/buildserial}"
PY="${PYTHON3:-python3}"
BDIR="/tmp/exasim_surfq_meshadapt_build"
RDIR="/tmp/exasim_surfq_meshadapt_run"

rm -rf "$INSTALL_PREFIX" "$BDIR" "$RDIR"
cmake --install "$EXASIM_BUILD" --prefix "$INSTALL_PREFIX" > "$RDIR.install.log" 2>&1 \
  || fail "cmake --install failed" "$RDIR.install.log"
if [ ! -x "$INSTALL_PREFIX/bin/text2code" ]; then
  echo "SKIP: installed text2code not found at $INSTALL_PREFIX/bin/text2code"
  exit 77
fi

mkdir -p "$RDIR"
cp "$REPO/apps/meshadaptivity/cylindermach8"/{pdeapp.txt,pdemodel.txt,grid.bin,udg.bin,vdg.bin,xdg.bin} "$RDIR"/
"$PY" - "$RDIR" "$INSTALL_PREFIX" <<'PY'
import re, sys
rdir, prefix = sys.argv[1], sys.argv[2]
m = open(rdir + '/pdemodel.txt').read()
m = m.replace('outputs Flux, Source, Tdfunc, Ubou, Fbou, FbouHdg, Avfield, Initu, VisScalars',
              'outputs Flux, Source, Tdfunc, Ubou, Fbou, FbouHdg, Avfield, Initu, VisScalars, SurfaceQuantities', 1)
m = m.rstrip('\n') + '''

function SurfaceQuantities(x, uq, v, w, uhat, n, tau, eta, mu, t)
  output_size(sq) = 2;
  sq[0] = uq[0];
  sq[1] = x[0]*n[0] + x[1]*n[1];
end
'''
open(rdir + '/pdemodel.txt', 'w').write(m)
a = open(rdir + '/pdeapp.txt').read()
a, k = re.subn(r'(?m)^saveSolBouFreq = 0;', 'saveSolBouFreq = 1;\nibs = [3, 5, 6];', a)
assert k == 1, 'pdeapp.txt: saveSolBouFreq line not found'
a = re.sub(r'(?m)^exasimpath = "[^"]*";\n', '', a)
open(rdir + '/pdeapp.txt', 'w').write('exasimpath = "%s";\n' % prefix + a)
PY

if ! ( cd "$RDIR" && EXASIM_PREFIX="$INSTALL_PREFIX" "$INSTALL_PREFIX/bin/text2code" pdeapp.txt > text2code.log 2>&1 ); then
  # A crash leaves little in the log: re-run under gdb (when available) for a backtrace.
  if command -v gdb > /dev/null 2>&1; then
    ( cd "$RDIR" && EXASIM_PREFIX="$INSTALL_PREFIX" gdb -batch -ex run -ex bt \
        --args "$INSTALL_PREFIX/bin/text2code" pdeapp.txt >> text2code.log 2>&1 ) || true
  fi
  fail "text2code" "$RDIR/text2code.log"
fi
cmake -S "$REPO/apps/sharedlibrary" -B "$BDIR" \
  -D "CMAKE_PREFIX_PATH=$INSTALL_PREFIX;$KOKKOS_DIR" -DExasim_DIR="$INSTALL_PREFIX" \
  -DEXASIM_MPI=OFF -DEXASIM_CUDA=OFF -DEXASIM_HIP=OFF \
  ${CC:+-DCMAKE_C_COMPILER="$CC"} ${CXX:+-DCMAKE_CXX_COMPILER="$CXX"} \
  > "$BDIR.cfg.log" 2>&1 || fail "exasimapp configure" "$BDIR.cfg.log"
cmake --build "$BDIR" -j > "$BDIR.build.log" 2>&1 || fail "exasimapp build" "$BDIR.build.log"
( cd "$RDIR" && EXASIM_PREFIX="$INSTALL_PREFIX" "$BDIR/exasimapp" pdeapp.txt > run.log 2>&1 ) \
  || fail "run" "$RDIR/run.log"

"$PY" - "$RDIR/dataout/out" <<'PY'
import struct, sys
prefix = sys.argv[1]
def rd(name):
    d = open(prefix + name, 'rb').read()
    v = struct.unpack('<%dd' % (len(d) // 8), d)
    return [int(round(t)) for t in v[:3]], list(v[3:])
_, info = rd('bouinfo_np0.bin')
blocks = [(int(info[2*j]), int(info[2*j+1])) for j in range(len(info) // 2)]
(hs, sv) = rd('bousurf_np0.bin'); (hg, gv) = rd('bousurfgeo_np0.bin'); (hx, xv) = rd('bouxdg_np0.bin')
np_, nfbou, nsq = hs; ncx = hx[2]; nd = hg[2] - ncx
assert nsq == 2 and hg[0] == np_ and hg[1] == nfbou and nd == 2, (hs, hg, hx)
srec, grec = np_*nfbou*nsq, np_*nfbou*(ncx+nd)
nsave = len(sv) // srec
assert nsave >= 1 and len(gv) == nsave*grec, 'bousurfgeo must hold one record per save'
worst = moved = 0.0
for t in range(nsave):
    so = t*srec; go = t*grec
    for ib, nf in blocks:
        m = np_*nf
        for i in range(m):
            x0, x1 = gv[go + i], gv[go + m + i]
            n0, n1 = gv[go + 2*m + i], gv[go + 3*m + i]
            worst = max(worst, abs(sv[so + m + i] - (x0*n0 + x1*n1)))
        so += m*nsq; go += m*(ncx+nd)
go = (nsave-1)*grec; xo = 0
for ib, nf in blocks:
    m = np_*nf
    for k in range(ncx):
        for i in range(m):
            moved = max(moved, abs(gv[go + k*m + i] - xv[xo + k*m + i]))
    go += m*(ncx+nd); xo += m*ncx
print('  [surfq meshadapt] %d boundary blocks, %d save(s): |x.n(kernel) - x.n(saved geometry)| = %.1e, '
      'wall motion vs start-up geometry = %.2e' % (len(blocks), nsave, worst, moved))
if moved < 1e-3:
    sys.exit('FAIL: the mesh did not move (%.2e); the test would not exercise mesh motion' % moved)
if worst > 1e-12:
    sys.exit('FAIL: saved geometry does not match the geometry the values were evaluated on (%.2e)' % worst)
PY
echo "PASS: surfacequantities moving-mesh"
