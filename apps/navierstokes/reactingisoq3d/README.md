# Exasim Text2Code Export

This directory contains the high-level Text2Code inputs exported from an Exasim frontend.

Generated files:

- `pdemodel.txt`: PDE model definition consumed by Text2Code.
- `pdeapp.txt`: application, mesh, solver, output, and runtime configuration.
- `grid.bin`: mesh coordinates and connectivity.
- `xdg.bin`, `udg.bin`, `vdg.bin`, `wdg.bin`: optional field data written only when present.

This directory has no case-local `main.cpp`. Generate and run either input with
the executables from an installed Exasim prefix:

```sh
export EXASIM_PREFIX=/path/to/exasim-prefix

"$EXASIM_PREFIX/bin/text2code" pdeapp.txt
mpiexec -n 16 "$EXASIM_PREFIX/bin/exasimapp" pdeapp.txt

"$EXASIM_PREFIX/bin/text2code" pdeapp100.txt
mpiexec -n 16 "$EXASIM_PREFIX/bin/exasimapp" pdeapp100.txt
```

Both inputs declare `mpiprocs = 16`. Use separate working copies when comparing
the variants so their generated `datain/` and `dataout/` products cannot
overwrite one another.

The `vdg.bin` file stores external variables. In backend data structures these are also called `odg`.
