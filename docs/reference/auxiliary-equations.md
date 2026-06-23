# Auxiliary Equations Reference

Auxiliary variables `w` are stored in `wdg` and configured by `ncw`. They can
represent algebraic closures, ODE states, relaxation variables, turbulence
variables, chemistry states, or other model-specific fields.

## Configuration

| Field | Meaning |
| --- | --- |
| `ncw` | Number of auxiliary `w` components. |
| `Initw` / `initwdg` | Initial value for `w`. |
| `Sourcew` / `sourcew` | Auxiliary equation source/evolution term. |
| `EoS` | Optional closure depending on `w`. |

## Text2Code Interface

```text
function Initw(x, eta, mu)
  output_size(w0) = ncw;
  ...
end

function Sourcew(x, uq, v, w, eta, mu, t)
  output_size(sw) = ncw;
  ...
end
```

## C++ Interface

```cpp
KOKKOS_INLINE_FUNCTION static
void initwdg(dstype w[], const dstype x[],
             const dstype uinf[], const dstype mu[]);

KOKKOS_INLINE_FUNCTION static
void sourcew(dstype sw[], const dstype x[], const dstype uq[],
             const dstype v[], const dstype w[], const dstype mu[],
             const dstype uinf[], dstype t);
```

## Common Auxiliary Forms

| Form | Interpretation |
| --- | --- |
| Algebraic | `w = G(u, q, v, x, t; mu)` through EOS/closure style logic. |
| ODE | `dw/dt = G(u, q, w, v, x, t; mu)`. |
| Relaxation | `tau_w dw/dt + w = G(...)`. |
| ADE | Memory or dispersive auxiliary differential equations. |

The exact time integration semantics are controlled by the backend model and
problem settings; this Reference documents the callback interface.

## Coupling to Other Functions

`w` can appear in:

- `Flux`
- `Source`
- `Tdfunc`
- `EoS`
- `Ubou`
- `Fbou`
- `FbouHdg`
- `VisScalars`, `VisVectors`, `VisTensors`
- `QoIvolume`, `QoIboundary`

## Sweep and Restart Notes

If `sol.bin` or saved solution files contain `wdg`, Exasim trusts the file
contents. If `wdg` is absent and `ncw > 0`, Exasim computes it from `Initw`
using the active `physicsparam`.
