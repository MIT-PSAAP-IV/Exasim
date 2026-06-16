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
#   EXASIM_INSTALL  Exasim install prefix (default: $EXASIM_PREFIX if set,
#                   else $EXASIM_ROOT-build/install)
#   FRONTEND        python | julia | matlab (required)
#   PYTHON3         python interpreter (default: python3)
#   QOI_TOL         QoI gate (default: 1e-8)
set -uo pipefail

ROOT="${EXASIM_ROOT:?set EXASIM_ROOT to the Exasim repo root}"
FE="${FRONTEND:?set FRONTEND to python|julia|matlab}"
INSTALL="${EXASIM_INSTALL:-${EXASIM_PREFIX:-$ROOT-build/install}}"
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
    # PATH first, then the macOS GUI-app install location.
    MATLAB="$(command -v matlab || true)"
    if [ -z "$MATLAB" ]; then
      MATLAB="$(ls -d /Applications/MATLAB_*.app/bin/matlab 2>/dev/null | sort | tail -1)"
    fi
    [ -n "$MATLAB" ] && [ -x "$MATLAB" ] || { echo "SKIP: no matlab"; exit $SKIP; }
    "$MATLAB" -batch "assert(license('test','Symbolic_Toolbox')==1)" >/dev/null 2>&1 \
      || { echo "SKIP: matlab lacks the Symbolic Math Toolbox"; exit $SKIP; }
    ;;
esac

# --- run from a scratch dir (datain/dataout/.exasim land there) --------------
RUN="$(mktemp -d "${TMPDIR:-/tmp}/exasim_frontend_${FE}.XXXXXX")"
trap 'rm -rf "$RUN"' EXIT
mkdir -p "$RUN/a"
cp "$SRC"/* "$RUN/a/"
cd "$RUN/a"

# --- export-app mode (EXPORT_TEST=1): inject a pde.exportapp assignment just
# before the exasim() call so the run also packages a data-transfer bundle ----
if [ "${EXPORT_TEST:-0}" = "1" ]; then
  case "$FE" in
    python) ins="pde['exportapp'] = os.path.join(os.getcwd(), 'bundle')" ;;
    julia)  ins='pde.exportapp = joinpath(pwd(), "bundle")' ;;
    matlab) ins='pde.exportapp = fullfile(pwd,"bundle");' ;;
  esac
  awk -v ins="$ins" '/exasim\(pde/ && !d {print ins; d=1} {print}' "$APP" > "$APP.tmp" \
    && mv "$APP.tmp" "$APP"
fi

export EXASIM_PREFIX="$INSTALL"
# Hermetic per-user model cache (never touch the real ~/.exasim from tests).
export EXASIM_CACHE_DIR="$RUN/cache"
status=0
case "$FE" in
  python)
    sitedir="$(echo "$INSTALL"/lib/python*/site-packages)"
    PYTHONPATH="$sitedir${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$APP" || status=$?
    ;;
  julia)
    # Leading colon: keep the default environment stack (and active project)
    # first, then append the installed package dir. Putting the package dir
    # first would make it the active project and demand a Manifest.toml.
    JULIA_LOAD_PATH=":$INSTALL/share/exasim/julia" julia "$APP" || status=$?
    ;;
  matlab)
    "$MATLAB" -batch "run('$INSTALL/share/exasim/matlab/exasim_setup.m'); pdeapp" || status=$?
    ;;
esac
if [ "$status" -ne 0 ]; then
  echo "FAIL: $APP exited with status $status"
  exit 1
fi

# --- export-app assertions (EXPORT_TEST=1) ----------------------------------
# The bundle must be present, pristine (no build/, empty dataout/), and
# genuinely relocatable: copy it elsewhere and build+run via run.sh against
# the install, with no frontend involved.
if [ "${EXPORT_TEST:-0}" = "1" ]; then
  B="$(pwd)/bundle"
  [ -d "$B" ] || { echo "FAIL: no bundle at $B"; exit 1; }
  for f in CMakeLists.txt main.cpp run.sh manifest.json; do
    [ -f "$B/$f" ] || { echo "FAIL: bundle missing $f"; exit 1; }
  done
  { [ -d "$B/kernels" ] && [ -d "$B/datain" ]; } \
    || { echo "FAIL: bundle missing kernels/ or datain/"; exit 1; }
  [ -d "$B/build" ] && { echo "FAIL: bundle has build/ (not pristine)"; exit 1; }
  [ -z "$(ls -A "$B/dataout")" ] \
    || { echo "FAIL: bundle dataout/ not empty (not pristine)"; exit 1; }
  REL="$RUN/relocated"; rm -rf "$REL"; cp -R "$B" "$REL"
  ( cd "$REL" && EXASIM_ROOT="$INSTALL" bash run.sh ) > "$RUN/export-run.log" 2>&1 \
    || { cat "$RUN/export-run.log"; echo "FAIL: relocated bundle run.sh failed"; exit 1; }
  [ -f "$REL/dataout/outqoi.txt" ] \
    || { echo "FAIL: relocated run produced no dataout/outqoi.txt"; exit 1; }
  echo "frontend_${FE}_export: bundle pristine + relocatable build+run OK"
fi

# --- model-cache check (CACHE_TEST=1): a second app dir must reuse the
# cached (libfrontend_model, exasimapp) pair with zero compilation ------------
if [ "${CACHE_TEST:-0}" = "1" ]; then
  mkdir -p "$RUN/b"
  cp "$SRC"/* "$RUN/b/"
  cd "$RUN/b"
  case "$FE" in
    python) PYTHONPATH="$sitedir${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$APP" > run-b.log 2>&1 \
              || { cat run-b.log; echo "FAIL: cache-test rerun failed"; exit 1; } ;;
    *) echo "FAIL: CACHE_TEST only implemented for python"; exit 1 ;;
  esac
  grep -q "Model cache hit" run-b.log \
    || { cat run-b.log; echo "FAIL: second app dir did not hit the model cache"; exit 1; }
  if grep -q "Building CXX" run-b.log; then
    cat run-b.log; echo "FAIL: cache hit still compiled something"; exit 1
  fi
  echo "model cache reused from a second app directory (no compilation)"
fi

# --- QoI gate (in addition to any in-language assert) ------------------------
qoi_file="$(pwd)/dataout/outqoi.txt"
[ -f "$qoi_file" ] || { echo "FAIL: no $qoi_file produced"; exit 1; }
qoi="$(tail -1 "$qoi_file" | awk '{print $2}')"
[ -n "$qoi" ] || { echo "FAIL: could not read QoI from $qoi_file"; exit 1; }
if awk "BEGIN{exit !( ($qoi)+0 < ($QOI_TOL)+0 )}"; then
  echo "frontend_$FE PASSED (Domain_QoI1 = $qoi < $QOI_TOL)"
else
  echo "FAIL: Domain_QoI1 = $qoi >= $QOI_TOL"
  exit 1
fi
