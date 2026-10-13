# Coupled Poisson problem with a non-matching interface

Two Poisson models share the interface x = 0.5 of the unit square. Model 1
(`pdeapp1.m`, 8 x 16 quadrilaterals on [0, 0.5] x [0, 1]) and model 2
(`pdeapp2.m`, 12 x 24 quadrilaterals on [0.5, 1] x [0, 1]) are meshed
independently, so their interface faces do not match (h ratio 3:2). Both set

```matlab
pde{m}.interfacetype = 1;   % 0 = conformal (default), 1 = non-matching
```

which is written to the fixed zero-based slot `app.problem[34]`. The exact
solution is u = sin(pi x) sin(pi y).

## Run

From MATLAB, in this directory:

```matlab
pdeapp
```

`pdeapp.m` calls `exasim(pde, mesh)`, which builds the MPI solver variant even
with one rank per model (a coupled world is an MPI application) and runs both
models together. Set `pde{m}.mpiprocs` to use more ranks per model; the two
models may use different counts.

Set `curved = true` in `pdeapp.m` to bend the interface to
x = 0.5 + 0.05 sin(pi y). Each model then interpolates the curve at its own
face nodes, so the discrete interfaces leave small gaps and overlaps.

## What to look for

During setup the solver prints, for both coupling directions,

```
Non-matching donor search, Gamma^1 ...
Non-matching donor search, Gamma^2 ...
```

with the number of faces and points, the maximum point-to-face distance and the
number of points off their donor face. Both lines must appear; with
`interfacetype = 0` neither appears and the interface must match node for node.

## Requirements and limits

- MPI build of Exasim (the coupled solver always runs under MPI).
- Both models use the same element type, polynomial order (1 to 4) and quadrature.
- Two-dimensional problems only; CPU only (the GPU path is not yet validated).
- See `docs/physics-models/multi-model.md` for the full description.
