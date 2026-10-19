# Mach-8 cylinder flow

This Text2Code case uses built-in model `7` for a steady Mach-8.03 flow around
a circular body. It is configured for HDG, polynomial order 4, quadrature order
8, and two MPI ranks. The mesh and initial fields are provided by `grid.bin`,
`xdg.bin`, `udg.bin`, and `vdg.bin`.

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

The high-order hypersonic solve can be sensitive to its supplied initial and
artificial-viscosity fields; do not replace those binary inputs with zeros when
using this case as a regression.
