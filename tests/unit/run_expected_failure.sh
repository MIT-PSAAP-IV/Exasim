#!/usr/bin/env bash
# [nonmatching] Expected-failure wrapper for MPI unit tests (Phase 11 review N2).
#   run_expected_failure.sh <ranks> <regex> <command> [args...]
# Passes only if the command exits nonzero, at least one and at most <ranks>
# "ERROR:" diagnostics match <regex>, and every other "ERROR:" line is a
# truncated copy of the matching diagnostic (one uniform message).
#
# The first rank to reach MPI_Abort can kill the other ranks while they are
# still printing, and the launcher may interleave their partial output. A
# diagnostic cut short or broken by another rank's banner ("=====", "-----",
# "[Rank") is therefore accepted when the part before the break is a prefix of
# the matching message; any other ERROR text fails the check. Launcher (PMIx/PRRTE)
# teardown errors are ignored.
set -uo pipefail
ranks=${1:?ranks}; regex=${2:?regex}; shift 2
output=$("$@" 2>&1)
status=$?
printf '%s\n' "$output"
if [[ $status -eq 0 ]]; then
    echo "EXPECTED-FAILURE CHECK FAILED: command exited 0"; exit 1
fi
# The MPI launcher's own teardown messages (e.g. OpenMPI's "PMIX ERROR: PMIX_ERR_UNREACH ..." when ranks die
# during MPI_Abort) are not Exasim diagnostics; leave them out.
errors=$(printf '%s\n' "$output" | grep -a 'ERROR:' | grep -a -v -E 'PMIX ERROR|PRTE ERROR|ORTE ERROR' || true)
count=$(printf '%s\n' "$errors" | grep -a -c -E "$regex" || true)
if [[ $count -lt 1 ]]; then
    echo "EXPECTED-FAILURE CHECK FAILED: 0 matching diagnostics, expected at least 1"; exit 1
fi
if [[ $count -gt $ranks ]]; then
    echo "EXPECTED-FAILURE CHECK FAILED: $count matching diagnostics, expected at most $ranks"; exit 1
fi
reference=$(printf '%s\n' "$errors" | grep -a -E "$regex" | head -n 1 | sed 's/^.*ERROR: //')
others=0
while IFS= read -r line; do
    [[ -z $line ]] && continue
    printf '%s\n' "$line" | grep -a -q -E "$regex" && continue
    text=${line#*ERROR: }
    text=${text%%=====*}; text=${text%%-----*}; text=${text%%\[Rank*}
    text=$(printf '%s' "$text" | sed 's/[[:space:]]*$//')
    if [[ -n $text && $reference != "$text"* ]]; then
        echo "EXPECTED-FAILURE CHECK FAILED: ERROR line does not match '$regex': $line"
        others=$((others+1))
    fi
done <<< "$errors"
if [[ $others -ne 0 ]]; then exit 1; fi
echo "EXPECTED-FAILURE CHECK PASSED: exit $status, $count uniform diagnostics"
exit 0
