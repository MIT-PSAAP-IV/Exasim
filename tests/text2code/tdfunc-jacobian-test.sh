#!/usr/bin/env bash
# Fresh generation exercises both the value-only and derivative overloads.
set -eu
ROOT="${EXASIM_ROOT:?EXASIM_ROOT not set}"
INSTALL="${EXASIM_INSTALL:?EXASIM_INSTALL not set}"
T2C="${EXASIM_TEXT2CODE:-$INSTALL/bin/text2code}"
[ -x "$T2C" ] || { echo "SKIP: Text2Code unavailable"; exit 77; }
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cp "$ROOT/tests/text2code/tdfunc-jacobian-model.txt" "$WORK/pdemodel1.txt"
sed "s|exasimpath = \"[^\"]*\"|exasimpath = \"$INSTALL\"|" \
  "$ROOT/backend/Model/BuiltIn/pdeapp1.txt" > "$WORK/pdeapp1.txt"
cp "$ROOT/tests/text2code/tdfunc-jacobian.cmake.txt" "$WORK/CMakeLists.txt"
(cd "$WORK" && EXASIM_PREFIX="$INSTALL" "$T2C" pdeapp1.txt --out-dir generated --gen-only)
# USE_CMAKE builds leave the second (symbolic evaluation) step to callers when
# --gen-only is used; use the regular --out-dir path for those executables.
if [ ! -f "$WORK/generated/my_model.hpp" ]; then
  (cd "$WORK" && EXASIM_PREFIX="$INSTALL" "$T2C" pdeapp1.txt --out-dir generated)
fi
# Quote-includes in the provider must resolve to the freshly generated kernels,
# not to stale checked-in files beside the provider template.
cp "$ROOT/backend/Model/Text2codeGenerated/libt2cmodel.cpp" "$WORK/generated/testprovider.cpp"
cmake -S "$WORK" -B "$WORK/build" -DCMAKE_BUILD_TYPE=Release \
  -DEXASIM_ROOT="$ROOT" -DCMAKE_PREFIX_PATH="$INSTALL/deps/kokkos/buildserial;$INSTALL"
cmake --build "$WORK/build" --parallel 2
"$WORK/build/tdfunc_jacobian"

# A complete HDG provider also needs source/flux partials. Regenerate with the
# ordinary Jacobian declaration, then exercise its actual ABI registration.
sed 's|// Deliberately omit the optional Jacobian list: mass partials are mandatory.|jacobian uq, w, uhat|' \
  "$ROOT/tests/text2code/tdfunc-jacobian-model.txt" > "$WORK/pdemodel1.txt"
(cd "$WORK" && EXASIM_PREFIX="$INSTALL" "$T2C" pdeapp1.txt --out-dir generated --gen-only)
if [ ! -f "$WORK/generated/HdgFlux.cpp" ]; then
  (cd "$WORK" && EXASIM_PREFIX="$INSTALL" "$T2C" pdeapp1.txt --out-dir generated)
fi
cp "$ROOT/backend/Model/Text2codeGenerated/libt2cmodel.cpp" "$WORK/generated/testprovider.cpp"
cmake -S "$WORK" -B "$WORK/build" -DEXASIM_TEST_MASS_PROVIDER=ON
cmake --build "$WORK/build" --parallel 2
"$WORK/build/tdfunc_jacobian"

