# External Variables Through v

The `v` variables represent externally supplied or auxiliary nodal fields that
are not solved as the current PDE system's primary unknowns. In backend naming,
these are `odg`; some frontend code also refers to this storage as `vdg`.

`v` is initialized by `initv` / `Initv` when generated, or supplied through
mesh/input data when the application provides it.

## Common Uses

- Prescribed velocity or material fields.
- Electromagnetic fields supplied by another model.
- Atmospheric or database fields.
- Reduced-order model predictions.
- Coupled solver outputs.
- Artificial-viscosity fields stored in `mesh.vdg` / `odg`.

## Where v Can Enter

`v` can be used in:

- Fluxes.
- Sources.
- EOS callbacks.
- Auxiliary `w` equations.
- Boundary conditions.
- Initial conditions.
- QoI and visualization callbacks.

## Simple Pattern

For a scalar transport model with prescribed velocity \(a(x)\), store the
velocity in `v` and write

$$
F(u,v) = v u.
$$

This keeps the transported state `u` small while allowing the model to use a
spatially varying prescribed field.

Do not confuse `v` with `externalparam` (`uinf` or `eta`). `v` is a nodal field;
`externalparam` is a parameter vector.
