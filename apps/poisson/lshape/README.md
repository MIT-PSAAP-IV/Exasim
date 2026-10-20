# Poisson L-shaped domain

This Text2Code example solves a Poisson problem on the L-shaped mesh stored in
`grid.bin`. It uses built-in model `1`, HDG, polynomial order 3, and four MPI
ranks. The re-entrant corner makes the case useful for observing reduced
regularity compared with a smooth-domain manufactured solution.

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

No convergence rate is asserted by this README; use a separately defined mesh
sequence and error check for quantitative studies.
