# Navier-Stokes ISOQ

This Text2Code application solves a steady compressible-flow problem around
the two-dimensional ISOQ geometry. It uses the built-in model `11`, HDG,
polynomial order 2, quadrature order 4, and four MPI ranks. The supplied
`grid.bin`, `xdg.bin`, `udg.bin`, and `vdg.bin` provide the mesh and initial
fields expected by `pdeapp.txt`.

Generate the model and run it with an installed Exasim:

```sh
export EXASIM_PREFIX=/path/to/exasim-prefix
"$EXASIM_PREFIX/bin/text2code" pdeapp.txt
mpiexec -n 4 "$EXASIM_PREFIX/bin/exasimapp" pdeapp.txt
```

Alternatively, build the case-local `main.cpp` against the installed package:

```sh
cmake -S . -B build -DExasim_DIR="$EXASIM_PREFIX/lib/cmake/Exasim"
cmake --build build
mpiexec -n 4 build/exasimapp pdeapp.txt
```

The run generates preprocessing data under `datain/` and writes solver output
under `dataout/`. These directories are generated products and should not be
used as canonical inputs.
