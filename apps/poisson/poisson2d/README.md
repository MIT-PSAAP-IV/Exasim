# Two-dimensional Poisson variants

This directory contains two Text2Code configurations on the unit-square mesh:

- `pdeapp.txt`: custom model `pdemodel.txt`, LDG, polynomial order 3, and two
  MPI ranks;
- `pdeapp_hdg_refined.txt`: built-in model `1`, HDG, polynomial order 3, two
  uniform refinement levels, and four MPI ranks.

Generate and run either variant with the installed executables:

```sh
export EXASIM_PREFIX=/path/to/exasim-prefix

"$EXASIM_PREFIX/bin/text2code" pdeapp.txt
mpiexec -n 2 "$EXASIM_PREFIX/bin/exasimapp" pdeapp.txt

"$EXASIM_PREFIX/bin/text2code" pdeapp_hdg_refined.txt
mpiexec -n 4 "$EXASIM_PREFIX/bin/exasimapp" pdeapp_hdg_refined.txt
```

Alternatively, build the case-local `main.cpp` once and run either variant:

```sh
cmake -S . -B build -DExasim_DIR="$EXASIM_PREFIX/lib/cmake/Exasim"
cmake --build build
mpiexec -n 2 build/exasimapp pdeapp.txt
mpiexec -n 4 build/exasimapp pdeapp_hdg_refined.txt
```

Use isolated working copies when comparing variants so their generated
`datain/` and `dataout/` products cannot overwrite one another.
