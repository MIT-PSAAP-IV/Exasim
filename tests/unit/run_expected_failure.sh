#!/usr/bin/env bash
# [nonmatching] Expected-failure wrapper for MPI unit tests (Phase 11 review N2).
#   run_expected_failure.sh <ranks> <regex> <command> [args...]
# Passes only if the command exits nonzero, every "ERROR:" diagnostic matches
# <regex> (one uniform message), and exactly <ranks> such diagnostics appear.
set -uo pipefail
ranks=${1:?ranks}; regex=${2:?regex}; shift 2
output=$("$@" 2>&1)
status=$?
printf '%s\n' "$output"
errors=$(printf '%s\n' "$output" | grep -a 'ERROR:' || true)
count=$(printf '%s\n' "$errors" | grep -a -c -E "$regex" || true)
others=$(printf '%s\n' "$errors" | grep -a -v -E "$regex" | grep -a -c 'ERROR:' || true)
if [[ $status -eq 0 ]]; then
    echo "EXPECTED-FAILURE CHECK FAILED: command exited 0"; exit 1
fi
if [[ $others -ne 0 ]]; then
    echo "EXPECTED-FAILURE CHECK FAILED: $others ERROR line(s) do not match '$regex'"; exit 1
fi
if [[ $count -ne $ranks ]]; then
    echo "EXPECTED-FAILURE CHECK FAILED: $count matching diagnostics, expected $ranks"; exit 1
fi
echo "EXPECTED-FAILURE CHECK PASSED: exit $status, $count uniform diagnostics"
exit 0
