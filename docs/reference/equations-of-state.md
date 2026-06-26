# Equations of State Reference

Equation-of-state callbacks provide closure or derived quantities used by
fluxes, sources, auxiliary equations, and postprocessing.

## Text2Code Function

```text
function EoS(x, uq, v, w, eta, mu, t)
  output_size(e) = N;
  ...
end
```

`EoS` is optional. If omitted, generated code supplies an empty/default
implementation where supported.

## C++ Model Functions

```cpp
KOKKOS_INLINE_FUNCTION static
void eos(dstype e[], const dstype x[], const dstype uq[],
         const dstype v[], const dstype w[], const dstype mu[],
         const dstype uinf[], dstype t);

KOKKOS_INLINE_FUNCTION static
void eos_du(dstype ed[], const dstype x[], const dstype uq[],
            const dstype v[], const dstype w[], const dstype mu[],
            const dstype uinf[], dstype t);

KOKKOS_INLINE_FUNCTION static
void eos_dw(dstype ew[], const dstype x[], const dstype uq[],
            const dstype v[], const dstype w[], const dstype mu[],
            const dstype uinf[], dstype t);
```

## Arguments

| Argument | Meaning |
| --- | --- |
| `x` | Spatial coordinates. |
| `uq` | Packed solution state. |
| `v` | External/other-DG field. |
| `w` | Auxiliary state. |
| `mu` | Physics parameters. |
| `uinf` / `eta` | Reference/free-stream values. |
| `t` | Current time. |

## Typical Outputs

EOS callbacks can compute:

- Pressure.
- Temperature.
- Enthalpy.
- Sound speed.
- Transport coefficients.
- Reaction-rate inputs.
- Material properties.

The exact output size and ordering are model-defined and must match the
generated or hand-written model code that consumes it.

## Derivatives

`eos_du` and `eos_dw` provide derivatives with respect to `uq` and `w`. They
are needed when EOS values affect HDG Jacobian assembly or coupled auxiliary
equations.

## Interactions

| Consumer | EOS role |
| --- | --- |
| Fluxes | Provide pressure, transport coefficients, or closure variables. |
| Sources | Provide reaction rates, thermodynamic state, or material response. |
| Auxiliary equations | Couple `w` evolution to thermodynamic closure. |
| Boundary conditions | Compute characteristic or thermodynamic boundary states. |
| Visualization | Expose derived variables as scalar/vector/tensor fields. |

## Example Pattern

```text
function EoS(x, uq, v, w, eta, mu, t)
  output_size(e) = 2;
  gamma = mu[0];
  rho = uq[0];
  energy = uq[3];
  pressure = (gamma - 1.0)*energy;
  e[0] = pressure;
  e[1] = gamma*pressure/rho;
end
```

Use the actual state ordering for the selected PDE model.
