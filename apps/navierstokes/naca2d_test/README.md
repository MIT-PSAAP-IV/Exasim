# NACA 2D refinement test

This compact NACA airfoil application exercises the built-in compressible-flow
model `12` with HDG and two levels of uniform mesh refinement. The configured
polynomial and quadrature orders are 2 and 4, respectively, and the default run
uses one MPI rank. `grid.bin` is the unrefined source mesh.

Run with the installed executables:

```sh
export EXASIM_PREFIX=/path/to/exasim-prefix
"$EXASIM_PREFIX/bin/text2code" pdeapp.txt
"$EXASIM_PREFIX/bin/exasimapp" pdeapp.txt
```

Alternatively, build the case-local `main.cpp`:

```sh
cmake -S . -B build -DExasim_DIR="$EXASIM_PREFIX/lib/cmake/Exasim"
cmake --build build
mpiexec -n 1 build/exasimapp pdeapp.txt
```

The case is intended primarily as a Text2Code and preprocessing regression
fixture; its name does not imply a validated aerodynamic benchmark.
