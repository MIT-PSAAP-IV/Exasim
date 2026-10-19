# Poisson Orion geometry

This application solves a Poisson-type problem on the two-dimensional Orion
mesh supplied in `grid.bin`, with geometry nodes in `xdg.bin`. It uses built-in
model `13`, HDG, polynomial order 3, and two MPI ranks. Four boundary classes
are defined in `pdeapp.txt`.

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

Generated input and output directories are local run products.
