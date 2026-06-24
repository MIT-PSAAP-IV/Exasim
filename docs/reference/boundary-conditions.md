# Boundary Conditions Reference

Boundary conditions in Exasim have two layers:

1. `pdeapp.txt` or frontend fields map geometry to integer boundary IDs.
2. Model functions dispatch on the integer boundary ID and compute boundary
   values or fluxes.

## Boundary Geometry

`pdeapp.txt` uses:

```text
boundaryconditions = [1, 2, 3, 4];
boundaryexpressions = [
  "x < 1e-8",
  "x > 1-1e-8",
  "y < 1e-8",
  "y > 1-1e-8"
];
```

Each boundary expression corresponds to one boundary condition code. The
meaning of the code is model-defined.

## Boundary Function Interfaces

### `Ubou`

Defines boundary state/value data.

```text
function Ubou(x, uq, v, w, uhat, n, tau, eta, mu, t)
  output_size(ub) = ncu;
  ...
end
```

### `Fbou`

Defines LDG boundary flux.

```text
function Fbou(x, uq, v, w, uhat, n, tau, eta, mu, t)
  output_size(fb) = ncu;
  ...
end
```

### `FbouHdg`

Defines HDG boundary residual/flux.

```text
function FbouHdg(x, uq, v, w, uhat, n, tau, eta, mu, t)
  output_size(fb) = ncu;
  ...
end
```

Direct C++ functions receive an additional `int ib` boundary ID and dispatch
on it.

## Boundary Types

Exasim does not enforce universal meanings for integer boundary codes. Common
application conventions include:

| Type | Typical implementation |
| --- | --- |
| Dirichlet | `Ubou` returns imposed state; `FbouHdg` penalizes trace/state mismatch. |
| Neumann | Boundary flux is set directly in `Fbou` or `FbouHdg`. |
| Robin | Boundary flux combines state and flux terms. |
| Wall | Model-specific no-penetration/no-slip/thermal condition. |
| Inflow | Prescribed physical state or characteristic inflow. |
| Outflow | Pressure, extrapolation, or characteristic outflow. |
| Farfield | Freestream or characteristic boundary model. |

The exact formulas belong to the PDE model.

## Periodic Boundaries

Periodic boundaries are configured with:

```text
periodicboundaries1 = [1];
periodicexprs1 = ["x"];
periodicboundaries2 = [2];
periodicexprs2 = ["x - 1.0"];
```

The preprocessing layer matches periodic faces; model boundary functions
should not implement periodicity manually.

## Coupled Interfaces

Coupled interfaces use:

- `interfaceconditions`
- `interfacefluxmap`
- `extFhat`
- `extUhat`
- `extStab`

These enable advanced interface flux hooks through provider callbacks such as
`Fint`, `Fext`, `Fhat`, `Uhat`, and `Stab`.

## Common Errors

| Symptom | Likely issue |
| --- | --- |
| Boundary faces unassigned | Expression does not match mesh coordinates within tolerance. |
| Wrong physical boundary behavior | Boundary code ordering does not match model dispatch. |
| HDG Newton failure near boundary | `FbouHdg` Jacobians inconsistent with `FbouHdg`. |
| MPI periodic mismatch | Periodic boundary expressions or node matching inconsistent. |
