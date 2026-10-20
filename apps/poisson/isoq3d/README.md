# Poisson ISOQ 3D

This application exercises built-in Poisson model `14` on a three-dimensional
ISOQ mesh. It uses HDG, polynomial order 2, quadrature order 4, and four MPI
ranks. `grid.bin` contains the mesh and `xdg.bin` supplies the geometry nodes.

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

Boundary identifiers and their coordinate predicates are defined directly in
`pdeapp.txt`; retain their order when modifying the mesh.
