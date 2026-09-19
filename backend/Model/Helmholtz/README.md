# Internal Helmholtz models

This directory contains the scalar HDG models used by the optional frozen-AV
Helmholtz filter.  The model solves

```text
u - div(ell2 grad(u)) = sensor,
n dot (ell2 grad(u)) = 0,
```

using Exasim's convention `q = -grad(u)`.  Its two `odg` components are
`v = (sensor, ell2)`.

The canonical Text2Code inputs are `pdeappN.txt` and `pdemodelN.txt`, where
`N` is the spatial dimension (1, 2, or 3).  When `EXASIM_TEXT2CODE` names an
available generator, CMake regenerates a `my_model.hpp` for each dimension in
the build tree and compiles those headers into the archive matching the active
backend:

- `libHelmholtzModelsserial.a` for the serial CPU backend;
- `libHelmholtzModelscuda.a` for the CUDA backend;
- `libHelmholtzModelship.a` for the HIP backend.

This mirrors `libbuiltinmodel{serial,cuda,hip}.a` and prevents a model archive
compiled for one Kokkos execution space from being linked into a solver built
for another.

`helmholtz_model.hpp` is intentionally retained as a bootstrap fallback for
builds configured without Text2Code.  It implements the same model contract;
the unit test checks the fallback value/Jacobian formulas and the ABI exported
by the compiled static library.

The generator inputs are installed under
`share/exasim/models/Helmholtz`, so an installed Exasim package retains the
mathematical sources used to produce the internal model library.
