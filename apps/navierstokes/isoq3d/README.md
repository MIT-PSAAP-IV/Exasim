# Exasim Text2Code Export

This directory contains the high-level Text2Code inputs exported from an Exasim frontend.

Generated files:

- `pdemodel.txt`: PDE model definition consumed by Text2Code.
- `pdeapp.txt`: application, mesh, solver, output, and runtime configuration.
- `grid.bin`: mesh coordinates and connectivity.
- `xdg.bin`, `udg.bin`, `vdg.bin`, `wdg.bin`: optional field data written only when present.

Regenerate the application with:

```sh
/path/to/exasim-prefix/bin/text2code pdeapp.txt
```

The `vdg.bin` file stores external variables. In backend data structures these are also called `odg`.

## Surface heat-flux visualization

`pdemodel.txt` declares a `VisSurfScalars` QoI (wall heat flux plus the
evaluation-point coordinates, `output_size(s) = 4`) and `pdeapp.txt` enables
it with `saveParaview = 1`, `nsurfsca = 4`, `ibvis = 4`. The model is built
as external built-in model ID 108 (see `CMakeLists.txt`,
`pdeapp108.txt`, `pdemodel108.txt`):

```sh
cmake -S apps/navierstokes/isoq3d -B build-isoq3d \
  -DCMAKE_PREFIX_PATH="/path/to/exasim-prefix;/path/to/kokkos-build"
cmake --build build-isoq3d -j
```

A run writes `outsurf*.vtu` (plus `outsurf.pvtu` in parallel) next to the
volume `outvis` files: Surface Field 0 is the wall heat flux,
Fields 1–3 are the evaluation-point coordinates.
Field counts come from the model (`PdeModel::nsurfsca`), not `pdeapp.txt`.

