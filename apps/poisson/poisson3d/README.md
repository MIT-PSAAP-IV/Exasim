# Three-dimensional Poisson problem

This Text2Code application solves a Poisson problem on the unit-cube mesh in
`grid.bin`. It uses built-in model `2`, HDG, polynomial order 3, quadrature
order 6, and four MPI ranks. All six coordinate-plane boundaries use the same
boundary-condition class.

Run with the installed executables:

```sh
export EXASIM_PREFIX=/path/to/exasim-prefix
"$EXASIM_PREFIX/bin/text2code" pdeapp.txt
mpiexec -n 4 "$EXASIM_PREFIX/bin/exasimapp" pdeapp.txt
```

Alternatively, build the case-local `main.cpp`:

```sh
cmake -S . -B build -DExasim_DIR="$EXASIM_PREFIX/lib/cmake/Exasim"
cmake --build build
mpiexec -n 4 build/exasimapp pdeapp.txt
```

The run creates case-local preprocessing and solution output directories.
