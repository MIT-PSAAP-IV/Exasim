#!/usr/bin/env bash
# Direct Text2Code CLI regression test -- FRONTEND-INDEPENDENT.
#
# Guards the reviewed `--out-dir` USE_CMAKE behavior: `text2code pdeapp.txt --out-dir <dir>`
# runs Code2Cpp to emit the concrete-model header `my_model.hpp` alongside the generated
# symbolic kernel sources. frontend_python_exporttext2code exercises the same path, but only
# via the Python frontend's exported package -- so MATLAB/Julia or a hand-written pdeapp could
# regress this CLI contract while that test stays green. This test invokes the installed
# text2code directly on a checked-in builtin fixture (Poisson / pdemodel1.txt), no frontend.
#
# Exits 77 (ctest SKIP) when the installed text2code or the fixture is unavailable.
set -u
SKIP=77
INSTALL="${EXASIM_INSTALL:?EXASIM_INSTALL not set}"
ROOT="${EXASIM_ROOT:?EXASIM_ROOT not set}"
T2C="${EXASIM_TEXT2CODE:-$INSTALL/bin/text2code}"

[ -x "$T2C" ] || { echo "SKIP: text2code not found at $T2C"; exit "$SKIP"; }
FIX="$ROOT/backend/Model/BuiltIn"
{ [ -f "$FIX/pdeapp1.txt" ] && [ -f "$FIX/pdemodel1.txt" ]; } \
  || { echo "SKIP: builtin pdeapp1/pdemodel1 fixture missing under $FIX"; exit "$SKIP"; }

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
cp "$FIX/pdemodel1.txt" "$WORK/"
# The builtin fixture ships a placeholder exasimpath ("/path/to/Exasim"); point it at the install
# prefix, which carries backend/Model/Text2codeGenerated + the SymEngine wrappers that Code2Cpp
# needs to compile into my_model.hpp.
sed "s|exasimpath = \"[^\"]*\"|exasimpath = \"$INSTALL\"|" "$FIX/pdeapp1.txt" > "$WORK/pdeapp1.txt"

if ! ( cd "$WORK" && EXASIM_PREFIX="$INSTALL" "$T2C" pdeapp1.txt --out-dir generated ) \
       > "$WORK/t2c.log" 2>&1; then
  cat "$WORK/t2c.log"
  echo "FAIL: text2code pdeapp1.txt --out-dir generated exited non-zero"
  exit 1
fi

# The core of the reviewed behavior: --out-dir emits the concrete-model header my_model.hpp
# (the Code2Cpp step) AND the generated symbolic kernel sources.
[ -f "$WORK/generated/my_model.hpp" ] \
  || { echo "FAIL: --out-dir did not emit generated/my_model.hpp"; exit 1; }
[ -f "$WORK/generated/SymbolicFunctions.cpp" ] \
  || { echo "FAIL: --out-dir did not emit the generated symbolic sources"; exit 1; }
# It must be a real generated header, not an empty stub. Its #include + compile is already
# exercised end to end: the run above built libt2cmodel*.{so,dylib}, which includes my_model.hpp.
grep -qE "struct|class|#include" "$WORK/generated/my_model.hpp" \
  || { echo "FAIL: generated/my_model.hpp looks empty/degenerate"; exit 1; }

echo "text2code_cli_out_dir: text2code pdeapp.txt --out-dir -> my_model.hpp + symbolic sources OK"
exit 0
