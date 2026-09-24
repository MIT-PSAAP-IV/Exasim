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

`pdemodel.txt` declares a `VisSurfScalars` QoI (nondimensional wall heat
flux including the HDG penalty term, `output_size(s) = 1`) and `pdeapp.txt`
enables it with `saveParaview = 1`, `nsurfsca = 1`, `ibvis = 3` (the
isothermal-wall tag: the only tag whose HDG boundary chunk prescribes the
wall temperature `TisoW`). The model is built
as external built-in model ID 108 (see `CMakeLists.txt`,
`pdeapp108.txt`, `pdemodel108.txt`):

```sh
cmake -S apps/navierstokes/isoq3d -B build-isoq3d \
  -DCMAKE_PREFIX_PATH="/path/to/exasim-prefix;/path/to/kokkos-build"
cmake --build build-isoq3d -j
```

A run writes `outsurf*.vtu` (plus `outsurf.pvtu` in parallel) next to the
volume `outvis` files: Surface Field 0 is the nondimensional wall heat
flux (HDG numerical flux, penalty included, same convention as the
volume solve).
Field counts come from the model (`PdeModel::nsurfsca`), not `pdeapp.txt`.

