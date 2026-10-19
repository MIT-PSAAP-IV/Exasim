# Periodic Poisson problem

This two-dimensional Poisson application uses HDG at polynomial order 3 on
the unit-square mesh in `grid.bin`. The right and left boundaries are paired
periodically through their `y` coordinate. The default run uses two MPI ranks
and a custom `pdemodel.txt`.

Run with the installed executables:

```sh
export EXASIM_PREFIX=/path/to/exasim-prefix
"$EXASIM_PREFIX/bin/text2code" pdeapp.txt
mpiexec -n 2 "$EXASIM_PREFIX/bin/exasimapp" pdeapp.txt
```

Alternatively, build the case-local `main.cpp`:

```sh
cmake -S . -B build -DExasim_DIR="$EXASIM_PREFIX/lib/cmake/Exasim"
cmake --build build
mpiexec -n 2 build/exasimapp pdeapp.txt
```

When changing the mesh, preserve the periodic boundary numbering and matching
coordinate expression defined in `pdeapp.txt`.
