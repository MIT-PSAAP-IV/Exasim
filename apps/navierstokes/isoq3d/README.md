# Exasim Text2Code Export

This directory contains the high-level Text2Code inputs exported from an Exasim frontend.

Generated files:

- `pdemodel.txt`: PDE model definition consumed by Text2Code.
- `pdeapp.txt`: application, mesh, solver, output, and runtime configuration.
- `grid.bin`: mesh coordinates and connectivity.
- `xdg.bin`, `udg.bin`, `vdg.bin`, `wdg.bin`: optional field data written only when present.

This directory has no case-local `main.cpp`. Generate the model and run it with
the executables from an installed Exasim prefix:

```sh
export EXASIM_PREFIX=/path/to/exasim-prefix
"$EXASIM_PREFIX/bin/text2code" pdeapp.txt
mpiexec -n 16 "$EXASIM_PREFIX/bin/exasimapp" pdeapp.txt
```

The 16-rank launch matches `mpiprocs = 16` in `pdeapp.txt`. Run from an
isolated copy when existing `datain/` or `dataout/` products must be preserved.

The `vdg.bin` file stores external variables. In backend data structures these are also called `odg`.
