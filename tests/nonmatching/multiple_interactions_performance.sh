#!/usr/bin/env bash
# Phase 5 timing-gate precondition.
#
# Do not time `matlab -batch`: that includes interpreter startup, code
# generation and compilation and is not a solver-performance measurement.
# The approved disposable install was not compiled with TIMING, so it does not
# expose the residual/Jacobian/J*v phase values required by Prompt 06.
set -euo pipefail

: "${EXASIM_ROOT:?EXASIM_ROOT not set}"
: "${EXASIM_PREFIX:?EXASIM_PREFIX must name the current-tree install}"
: "${PHASE0_TIMING_PREFIX:?PHASE0_TIMING_PREFIX must name a timing-enabled Phase 0 reference install}"

if [[ ${EXASIM_TIMING_ENABLED:-0} != 1 || ${PHASE0_TIMING_ENABLED:-0} != 1 ]]; then
  echo "SKIP: Phase 5 performance comparison requires matching TIMING-enabled Phase 0 and current-tree builds"
  echo "SKIP: whole MATLAB-process timing is deliberately not accepted"
  exit 77
fi

echo "FAIL: matching timing-enabled builds were declared, but the required test-only phase-timing driver is not implemented"
echo "FAIL: do not substitute whole-process time or a one-edge graph inside four domains"
exit 1
