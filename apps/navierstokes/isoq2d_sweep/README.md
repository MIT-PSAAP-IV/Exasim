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

cd /path/to/Exasim/apps/navierstokes/isoq2d_sweep
/path/to/Exasim/local/bin/text2code pdeapp.txt
cd /path/to/Exasim/apps/sharedlibrary
cmake -B build -DExasim_DIR=/path/to/Exasim/local
cmake --build build -j4
mpirun -np 4 build/exasimapp ../navierstokes/isoq2d_sweep/pdeapp.txt