#!/usr/bin/env bash
# Out-of-tree consumer tests (B1-B4): for each project under tests/consumers/,
#   B3: assert the CMakeLists does NOT reach into the Exasim source tree;
#   B2: configure it ONLY via find_package(Exasim) against an install prefix;
#       build it; and, if it ships a pdeapp.txt, run it and check the QoI gate.
#
# This catches export/packaging regressions (a consumer that builds in-tree but
# breaks against the installed package).
#
# Env (all optional):
#   EXASIM_BUILD     Exasim build dir to install from   (default: $REPO/build_conda)
#   INSTALL_PREFIX   where to install Exasim            (default: /tmp/exasim_install)
#   KOKKOS_DIR       Kokkos install/build dir           (default: $REPO/deps/kokkos/buildserial)
#   EXASIM_ROOT      tree holding runtime master data   (default: $REPO) -> fills @EXASIM_ROOT@
#   NP                   MPI ranks for run tests            (default: 2)
#   EXASIM_MPI_VARIANT   ON or OFF; if set, overrides the consumer's EXASIM_MPI
#                        cmake option (default: empty = use consumer default)
#   QOI_TOL              max allowed QoI[1] (L2 error^2)    (default: 1e-8)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXASIM_BUILD="${EXASIM_BUILD:-$REPO-build}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/tmp/exasim_install}"
KOKKOS_DIR="${KOKKOS_DIR:-$REPO/deps/kokkos/buildserial}"
EXASIM_ROOT="${EXASIM_ROOT:-$REPO}"
NP="${NP:-2}"
EXASIM_MPI_VARIANT="${EXASIM_MPI_VARIANT:-}"
QOI_TOL="${QOI_TOL:-1e-8}"

echo "[consumer-tests] repo           $REPO"
echo "[consumer-tests] install prefix $INSTALL_PREFIX"

# Install Exasim from the build dir
rm -rf "$INSTALL_PREFIX"
cmake --install "$EXASIM_BUILD" --prefix "$INSTALL_PREFIX" > /tmp/_consumer_install.log 2>&1 \
  || { echo "FAIL: cmake --install failed (see /tmp/_consumer_install.log)"; exit 1; }

fail=0
for dir in "$REPO"/tests/consumers/*/; do
  [ -f "$dir/CMakeLists.txt" ] || continue
  name="$(basename "$dir")"
  echo "=== consumer: $name ==="

  # B3: forbid source-tree references in the consumer CMakeLists.
  if grep -nE 'EXASIM_DIR|/backend/|/kokkos/build|\.\./\.\./\.\.|CMAKE_SOURCE_DIR.*Exasim' "$dir/CMakeLists.txt" \
       | grep -vE '^\s*#'; then
    echo "  FAIL[B3]: $name CMakeLists references the Exasim source tree"; fail=1; continue
  fi
  echo "  [B3] no source-tree references ok"

  # B2: configure out-of-tree via find_package against the install prefix, then build.
  bdir="/tmp/consumer_${name}_build"
  rm -rf "$bdir"
  cmake -S "$dir" -B "$bdir" \
        -D "CMAKE_PREFIX_PATH=$INSTALL_PREFIX;$KOKKOS_DIR" \
        ${CC:+-DCMAKE_C_COMPILER="$CC"} ${CXX:+-DCMAKE_CXX_COMPILER="$CXX"} \
        ${EXASIM_MPI_VARIANT:+-DEXASIM_MPI="${EXASIM_MPI_VARIANT}"} \
        > "$bdir.cfg.log" 2>&1 || { echo "  FAIL[B2]: configure (see $bdir.cfg.log)"; fail=1; continue; }
  cmake --build "$bdir" -j > "$bdir.build.log" 2>&1 \
        || { echo "  FAIL[B2]: build (see $bdir.build.log)"; fail=1; continue; }
  echo "  [B2] out-of-tree find_package build ok"

  # B4: run + numerical gate, if the consumer ships a pdeapp.txt.
  if [ -f "$dir/pdeapp.txt" ]; then
    rdir="/tmp/consumer_${name}_run"; rm -rf "$rdir"; mkdir -p "$rdir"
    cp "$dir"/*.txt "$dir"/*.bin "$dir"/*.hpp "$rdir"/ 2>/dev/null
    sed -i.bak "s#@EXASIM_ROOT@#$EXASIM_ROOT#" "$rdir/pdeapp.txt"
    exe="$(find "$bdir" -maxdepth 1 -type f -perm -u+x ! -name '*.cmake' | head -1)"
    if [ "${NP}" -gt 1 ]; then
      ( cd "$rdir" && mpirun -np "$NP" "$exe" pdeapp.txt > run.log 2>&1 )
    else
      ( cd "$rdir" && "$exe" pdeapp.txt > run.log 2>&1 )
    fi || { echo "  FAIL[B4]: run nonzero exit (see $rdir/run.log)"; \
            echo "  --- $rdir/run.log ---"; sed 's/^/  | /' "$rdir/run.log"; \
            echo "  --- end run.log ---"; fail=1; continue; }
    qoi1="$(tail -1 "$rdir/dataout/outqoi.txt" 2>/dev/null | awk '{print $2}')"
    if [ -z "$qoi1" ]; then echo "  FAIL[B4]: no QoI output"; fail=1; continue; fi
    if awk "BEGIN{exit !(($qoi1)+0 < ($QOI_TOL)+0)}"; then
      echo "  [B4] run ok, QoI[1]=$qoi1 < $QOI_TOL"
    else
      echo "  FAIL[B4]: QoI[1]=$qoi1 >= $QOI_TOL"; fail=1
    fi

    # B5: visualization gate. When a consumer enables saveParaview (and declares
    # nsca/nvec), the solve must emit ParaView vis. This guards the external-model
    # vis path: ParseInputs must propagate nsca/nvec/nten so savemode>0 (external
    # models do not bake the vis counts into datain). Without it no outvis is written.
    #
    # The backend writes outvis only when (saveParaview != 0) AND (nsca+nvec+nten > 0)
    # — see CVisualization's savemode. Mirror that exact condition here so the gate
    # never demands vis the backend would not produce (e.g. a future consumer with
    # saveParaview=1 but zero vis fields). A serial run emits only <name>.vtu; a
    # parallel run adds <name>.pvtu — accept either. Count with find rather than
    # `ls glob1 glob2`, whose exit status is non-zero when one glob has no match
    # (the serial case, where no .pvtu exists), which would mis-flag a present .vtu.
    visfields=0
    for _k in nsca nvec nten; do
      _v="$(grep -E "^[[:space:]]*${_k}[[:space:]]*=" "$rdir/pdeapp.txt" | head -1 | grep -oE '[0-9]+' | head -1)"
      visfields=$(( visfields + ${_v:-0} ))
    done
    if grep -qE '^[[:space:]]*saveParaview[[:space:]]*=[[:space:]]*[1-9]' "$rdir/pdeapp.txt" \
         && [ "$visfields" -gt 0 ]; then
      nvis="$(find "$rdir/dataout" -maxdepth 1 \( -name 'outvis*.vtu' -o -name 'outvis*.pvtu' \) 2>/dev/null | wc -l | tr -d ' ')"
      if [ "$nvis" -gt 0 ]; then
        echo "  [B5] vis ok: $nvis outvis file(s)"
      else
        echo "  FAIL[B5]: saveParaview enabled (nsca+nvec+nten>0) but no outvis*.vtu/.pvtu written"; fail=1
      fi
    fi
  fi
done

echo "=========================================="
[ "$fail" -eq 0 ] && echo "ALL CONSUMER TESTS PASSED" || echo "SOME CONSUMER TESTS FAILED"
exit "$fail"
