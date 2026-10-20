# Steady NACA 0012 flow

This directory contains steady compressible-flow configurations for the NACA
0012 airfoil. `pdeapp.txt` uses the built-in model `12`, LDG, polynomial order
2, and one rank. `pdeapp_sweep.txt` uses HDG at polynomial order 4 and performs
a warm-started four-case physics-parameter sweep on four ranks. Both inputs use
the supplied curved-airfoil mesh in `grid.bin`.

Generate and run either input with the installed executables:

```sh
export EXASIM_PREFIX=/path/to/exasim-prefix

"$EXASIM_PREFIX/bin/text2code" pdeapp.txt
"$EXASIM_PREFIX/bin/exasimapp" pdeapp.txt

"$EXASIM_PREFIX/bin/text2code" pdeapp_sweep.txt
mpiexec -n 4 "$EXASIM_PREFIX/bin/exasimapp" pdeapp_sweep.txt
```

Alternatively, build the case-local `main.cpp` once and select the input:

```sh
cmake -S . -B build -DExasim_DIR="$EXASIM_PREFIX/lib/cmake/Exasim"
cmake --build build
mpiexec -n 1 build/exasimapp pdeapp.txt
mpiexec -n 4 build/exasimapp pdeapp_sweep.txt
```

Use separate working copies when running both variants concurrently because
they create case-local `datain/`, `dataout/`, and sweep output directories.
