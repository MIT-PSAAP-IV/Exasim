#!/usr/bin/env bash
# Run one language-frontend end-to-end test (FRONTEND=python|julia|matlab).
#
# Each test runs the Poisson 2D pdeapp in tests/frontends/<lang>/ from a
# scratch directory against the installed Exasim package (EXASIM_INSTALL),
# then gates Domain_QoI1 (= integral of (u-uexact)^2) from dataout/outqoi.txt
# against QOI_TOL. Exits 77 (ctest SKIP) when the interpreter or its
# symbolic-math stack is unavailable.
#
# Environment:
#   EXASIM_ROOT     repo root (required)
#   EXASIM_INSTALL  Exasim install prefix (default: $EXASIM_ROOT/build/install)
#   FRONTEND        python | julia | matlab (required)
#   PYTHON3         python interpreter (default: python3)
#   QOI_TOL         QoI gate (default: 1e-8)
set -uo pipefail

ROOT="${EXASIM_ROOT:?set EXASIM_ROOT to the Exasim repo root}"
FE="${FRONTEND:?set FRONTEND to python|julia|matlab}"
INSTALL="${EXASIM_INSTALL:-$ROOT/build/install}"
QOI_TOL="${QOI_TOL:-1e-8}"
SKIP=77

if [ ! -f "$INSTALL/lib/cmake/Exasim/ExasimConfig.cmake" ]; then
  echo "FAIL: no Exasim install at $INSTALL (run tests/run-tests.sh first)"
  exit 1
fi

SRC="$ROOT/tests/frontends/$FE"
case "$FE" in
  python) APP="pdeapp.py" ;;
  julia)  APP="pdeapp.jl" ;;
  matlab) APP="pdeapp.m" ;;
  *) echo "FAIL: unknown FRONTEND=$FE"; exit 1 ;;
esac
if [ ! -f "$SRC/$APP" ]; then
  echo "SKIP: $SRC/$APP does not exist (frontend test not implemented yet)"
  exit $SKIP
fi

# --- dependency probes (exit 77 = clean ctest SKIP) --------------------------
case "$FE" in
  python)
    PY="${PYTHON3:-python3}"
    command -v "$PY" >/dev/null 2>&1 || { echo "SKIP: no python3"; exit $SKIP; }
    "$PY" -c 'import numpy, scipy, sympy' >/dev/null 2>&1 \
      || { echo "SKIP: python lacks numpy/scipy/sympy"; exit $SKIP; }
    ;;
  julia)
    command -v julia >/dev/null 2>&1 || { echo "SKIP: no julia"; exit $SKIP; }
    julia -e 'using SymPy' >/dev/null 2>&1 \
      || { echo "SKIP: julia lacks SymPy.jl"; exit $SKIP; }
    ;;
  matlab)
    command -v matlab >/dev/null 2>&1 || { echo "SKIP: no matlab"; exit $SKIP; }
    matlab -batch "assert(license('test','Symbolic_Toolbox')==1)" >/dev/null 2>&1 \
      || { echo "SKIP: matlab lacks the Symbolic Math Toolbox"; exit $SKIP; }
    ;;
esac

# --- run from a scratch dir (datain/dataout/.exasim land there) --------------
RUN="$(mktemp -d "${TMPDIR:-/tmp}/exasim_frontend_${FE}.XXXXXX")"
trap 'rm -rf "$RUN"' EXIT
cp "$SRC"/* "$RUN"/
cd "$RUN"

export EXASIM_PREFIX="$INSTALL"
status=0
case "$FE" in
  python)
    sitedir="$(echo "$INSTALL"/lib/python*/site-packages)"
    PYTHONPATH="$sitedir${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$APP" || status=$?
    ;;
  julia)
    JULIA_LOAD_PATH="$INSTALL/share/exasim/julia:@:@v#.#:@stdlib" julia "$APP" || status=$?
    ;;
  matlab)
    matlab -batch "run('$INSTALL/share/exasim/matlab/exasim_setup.m'); pdeapp" || status=$?
    ;;
esac
if [ "$status" -ne 0 ]; then
  echo "FAIL: $APP exited with status $status"
  exit 1
fi

# --- QoI gate (in addition to any in-language assert) ------------------------
qoi_file="$RUN/dataout/outqoi.txt"
[ -f "$qoi_file" ] || { echo "FAIL: no $qoi_file produced"; exit 1; }
qoi="$(tail -1 "$qoi_file" | awk '{print $2}')"
[ -n "$qoi" ] || { echo "FAIL: could not read QoI from $qoi_file"; exit 1; }
if awk "BEGIN{exit !( ($qoi)+0 < ($QOI_TOL)+0 )}"; then
  echo "frontend_$FE PASSED (Domain_QoI1 = $qoi < $QOI_TOL)"
else
  echo "FAIL: Domain_QoI1 = $qoi >= $QOI_TOL"
  exit 1
fi
