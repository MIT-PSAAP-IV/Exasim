# Poisson cone case

This three-dimensional Text2Code application uses built-in model `15` on the
mesh stored in `grid.bin`. It is configured for HDG, polynomial order 2,
quadrature order 4, and four MPI ranks. Six outer coordinate planes and one
additional surface are classified by the boundary expressions in `pdeapp.txt`.

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

The application writes generated preprocessing data and solution files below
the case directory.
