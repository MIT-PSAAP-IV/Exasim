# Unsteady NACA 0012 flow

This Text2Code application advances compressible flow over the NACA 0012
airfoil with the built-in model `12`. It uses HDG, polynomial order 3,
second-order two-stage time integration, a ramped time-step sequence, and eight
MPI ranks. The curved airfoil is defined by the supplied `grid.bin` and the
boundary expression in `pdeapp.txt`.

Run with the installed executables:

```sh
export EXASIM_PREFIX=/path/to/exasim-prefix
"$EXASIM_PREFIX/bin/text2code" pdeapp.txt
mpiexec -n 8 "$EXASIM_PREFIX/bin/exasimapp" pdeapp.txt
```

Alternatively, build the case-local `main.cpp`:

```sh
cmake -S . -B build -DExasim_DIR="$EXASIM_PREFIX/lib/cmake/Exasim"
cmake --build build
mpiexec -n 8 build/exasimapp pdeapp.txt
```

Solutions are saved every two time steps under `dataout/`. This is a
substantial transient run rather than a quick installation smoke test.
