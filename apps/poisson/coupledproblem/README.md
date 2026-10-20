# Exasim Text2Code Export

This directory contains the high-level Text2Code inputs exported from an Exasim frontend.

Generated files:

- `pdemodel.txt`: PDE model definition consumed by Text2Code.
- `pdeapp.txt`: application, mesh, solver, output, and runtime configuration.
- `grid.bin`: mesh coordinates and connectivity.
- `xdg.bin`, `udg.bin`, `vdg.bin`, `wdg.bin`: optional field data written only when present.

Generate both models and run the coupled problem with the installed
executables:

```sh
export EXASIM_PREFIX=/path/to/exasim-prefix
"$EXASIM_PREFIX/bin/text2code" pdeapp1.txt
"$EXASIM_PREFIX/bin/text2code" pdeapp2.txt
mpiexec -n 7 "$EXASIM_PREFIX/bin/exasimapp" \
  pdeapp1.txt 3 pdeapp2.txt 4
```

The `vdg.bin` file stores external variables. In backend data structures these are also called `odg`.

Coupled export:

- This package contains 2 coupled model definitions (`pdeapp*.txt`, `pdemodel*.txt`, `grid*.bin`).
- The optional case-local runtime can instead be built with
  `cmake -S . -B build -DExasim_DIR="$EXASIM_PREFIX/lib/cmake/Exasim"` and
  `cmake --build build`.
- Run that local executable with
  `mpiexec -n 7 build/exasimapp pdeapp1.txt 3 pdeapp2.txt 4`.
