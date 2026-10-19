# Orion high-speed flow

This application contains a steady high-speed flow configuration for an Orion
cross-section. `pdeapp.txt` uses built-in model `11`, HDG, polynomial order 3,
quadrature order 6, and four MPI ranks. The canonical mesh and initial fields
are `grid.bin`, `xdg.bin`, `udg.bin`, and `vdg.bin`.

This directory does not contain a standalone CMake project. Use an installed
Text2Code toolchain from an isolated working copy:

```sh
"$EXASIM_PREFIX/bin/text2code" pdeapp.txt
mpiexec -n 4 "$EXASIM_PREFIX/bin/exasimapp" pdeapp.txt
```

Generated `datain/` and `dataout/` directories are run products.
