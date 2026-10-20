# 3D coupled Poisson with a non-matching interface (tetrahedra)

Two Poisson models share the interface x = 0.5 of the unit cube:

- model 1 (`pdeapp1`): [0, 0.5] x [0, 1] x [0, 1], 2 x 4 x 4 cubes (6 tetrahedra per cube);
- model 2 (`pdeapp2`): [0.5, 1] x [0, 1] x [0, 1], 2 x 2 x 2 cubes.

The exact solution is u = exp(x) sin(pi y) sin(pi z). Its normal flux on x = 0.5
is nonzero, so the coupling has to transfer a real flux. Both models use
p = 2, the same element type and the same quadrature.

The model-1 mesh is twice as fine in y and z as the model-2 mesh, so the
interface faces do not match; every model-1 interface face lies inside one
model-2 face. Both models set

```matlab
pde{m}.interfacetype = 1;   % non-matching coupling
```

During setup the solver prints, for both coupling directions,

```
Non-matching donor search, Gamma^1 ...
Non-matching donor search, Gamma^2 ...
```

with the number of faces and points, the maximum point-to-face distance and
the number of points off their donor face.

Each model builds its mesh with `cubemesh` on the unit cube and maps x onto
its half of the domain. The MATLAB, Python and Julia versions use their own
frontend's `cubemesh`.

## Run

MATLAB, in this directory:

```matlab
pdeapp
```

Python:

```sh
python3 pdeapp.py
```

Julia:

```sh
julia --project=<repo>/frontends/Julia/Exasim pdeapp.jl
```

Each run prints the maximum nodal error against the exact solution.

Python and Julia use `EXASIM_PREFIX` if it is set, and otherwise a sibling
`<repo>/../exasim_install`. Set `pde{m}.mpiprocs` to use more ranks per model.

The Text2Code version is `apps/coupledproblems/poisson/nonmatching3d_tet`.

## Requirements and limits

- MPI build of Exasim; the coupled solver always runs under MPI.
- Both models use the same element family (tetrahedra or hexahedra),
  polynomial order (1 to 4) and quadrature. Tet/hex coupling is not supported.
- CPU only. The GPU path is rejected and not validated.
- See `docs/physics-models/multi-model.md` for the full description.
