#!/usr/bin/env bash
# Repo hygiene check — run in CI before the build to catch accidental binary commits.
#
# Fails if any of the following are found in the git index:
#   1. .DS_Store files
#   2. Prebuilt shared libraries (*.dylib, *.so) tracked anywhere in the repo
#   3. Prebuilt static libraries (*.a) inside backend/ (frontends/Matlab carries
#      legacy MEX archives that predate these checks and are excluded)
#
# Deliberately does NOT sweep the full "tracked files that match .gitignore" set,
# since the repo carries ~230 historical violations from before the rules were added.
# Instead, subsequent PRs can clean those up incrementally.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

fail=0

echo "=== [1] No .DS_Store tracked ==="
ds=$(git ls-files | grep '\.DS_Store' || true)
if [ -n "$ds" ]; then
    printf "FAIL: tracked .DS_Store files (git rm --cached them):\n%s\n" "$ds"
    fail=1
else
    echo "ok"
fi

echo "=== [2] No prebuilt shared libraries (*.dylib, *.so) tracked ==="
dylibs=$(git ls-files | grep -E '\.(dylib|so(\.[0-9]+)*)$' || true)
if [ -n "$dylibs" ]; then
    printf "FAIL: tracked shared library files:\n%s\n" "$dylibs"
    fail=1
else
    echo "ok"
fi

echo "=== [3] No prebuilt static libraries (*.a) in backend/ ==="
afiles=$(git ls-files backend/ | grep '\.a$' || true)
if [ -n "$afiles" ]; then
    printf "FAIL: tracked static libraries in backend/ (git rm --cached them):\n%s\n" "$afiles"
    fail=1
else
    echo "ok"
fi

echo "=== [4] No leaked absolute home paths (/Users/<user>, /home/<user>) ==="
# Catches machine-specific developer paths committed to tracked source — the
# class of leak behind the old /Users/<dev>/... paths. Redacted placeholders
# (/Users/.../, /path/to/...) pass because the match requires an alphanumeric
# first path component; the CI runner workspace (/home/runner) is allowed and
# vendored third-party trees under deps/ are excluded.
leaks=$(git grep -nIE '/(Users|home)/[A-Za-z0-9_][A-Za-z0-9._-]*' \
          -- . ':(exclude)deps/*' ':(exclude)tests/check-hygiene.sh' 2>/dev/null \
        | grep -vE '/home/runner' || true)
if [ -n "$leaks" ]; then
    printf "FAIL: leaked absolute home path(s) — use a relative path, an env var, or /path/to/...:\n%s\n" "$leaks"
    fail=1
else
    echo "ok"
fi

if [ "$fail" -eq 0 ]; then
    echo "=== ALL HYGIENE CHECKS PASSED ==="
else
    echo "=== HYGIENE CHECKS FAILED ==="
    exit 1
fi
