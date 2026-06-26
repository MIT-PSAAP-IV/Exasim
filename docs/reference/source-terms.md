# Source Terms Reference

Source terms add volumetric forcing, reactions, body forces, and coupling
contributions to the PDE residual.

## Volume Source

Text2Code:

```text
function Source(x, uq, v, w, eta, mu, t)
  output_size(s) = ncu;
  ...
end
```

C++ model:

```cpp
KOKKOS_INLINE_FUNCTION static
void source(dstype s[], const dstype x[], const dstype uq[],
            const dstype v[], const dstype w[], const dstype mu[],
            const dstype uinf[], dstype t);
```

## Arguments

| Argument | Meaning |
| --- | --- |
| `x` | Coordinates. |
| `uq` | Solution and, for ModelD/ModelW, auxiliary entries. |
| `v` | External/other-DG field. |
| `w` | Auxiliary state. |
| `mu` | Physics parameters. |
| `eta` / `uinf` | Reference/free-stream data. |
| `t` | Time. |

## Source Categories

| Category | Examples |
| --- | --- |
| Body forces | Gravity, Lorentz force, rotating-frame force. |
| Reactions | Chemical kinetics, source/sink terms. |
| Coupling | Energy exchange, multiphysics transfer, forcing from external fields. |
| Manufactured forcing | Source chosen to reproduce an analytical solution. |
| Stabilization support | Model-specific artificial source terms. |

## Auxiliary Source `Sourcew`

`Sourcew` controls the auxiliary `w` equation:

```text
function Sourcew(x, uq, v, w, eta, mu, t)
  output_size(sw) = ncw;
  ...
end
```

See [Auxiliary equations](auxiliary-equations.md).

## Jacobians

For HDG and implicit solves, generated code can provide derivatives with
respect to vectors listed in the `jacobian` declaration. Direct C++ models can
override:

- `source_jac_uq`
- `source_jac_w`

Incorrect source Jacobians can degrade or break Newton convergence.

## Example

```text
function Source(x, uq, v, w, eta, mu, t)
  output_size(s) = 1;
  s[0] = sin(pi*x[0])*sin(pi*x[1]);
end
```
