# Exasim Text2Code Export

This directory contains the high-level Text2Code inputs exported from an Exasim frontend.

Generated files:

- `pdemodel.txt`: PDE model definition consumed by Text2Code.
- `pdeapp.txt`: application, mesh, solver, output, and runtime configuration.
- `grid.bin`: mesh coordinates and connectivity.
- `xdg.bin`, `udg.bin`, `vdg.bin`, `wdg.bin`: optional field data written only when present.

Regenerate the application with:

```sh
/path/to/exasim-prefix/bin/text2code pdeapp.txt
```

The `vdg.bin` file stores external variables. In backend data structures these are also called `odg`.

Coupled export:

- This package contains 2 coupled model definitions (`pdeapp*.txt`, `pdemodel*.txt`, `grid*.bin`).
- Configure and build the generated runtime with `cmake -S . -B build -DExasim_DIR=/path/to/exasim-prefix`.
- Run the two-domain coupled solver with `mpirun -np 7 build/exasimapp pdeapp1.txt 3 pdeapp2.txt 4`.

