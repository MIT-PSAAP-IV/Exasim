# Model Functions Reference

This page lists the supported Exasim model functions and their interfaces.
Text2Code uses capitalized names such as `Flux`; direct C++ model defaults use
lower-case names such as `flux`. Provider code maps both forms into the same
runtime ABI.

## Common Arguments

| Argument | Meaning |
| --- | --- |
| `x` | Spatial coordinates. |
| `uq` / `udg` | Packed solution and auxiliary gradient state. |
| `v` / `odg` | External or other-DG field. |
| `w` / `wdg` | Auxiliary state variables. |
| `uhat` / `uhg` | HDG trace state. |
| `n` / `nlg` | Outward normal on a face. |
| `tau` | Stabilization parameters. |
| `mu` / `param` | Physics parameters. |
| `eta` / `uinf` | Extra/reference/free-stream values. |
| `t` / `time` | Current time. |
| `ib` | Boundary-condition index. |

## Volume Functions

| Text2Code | C++ model name | Output size | Purpose |
| --- | --- | --- | --- |
| `Flux` | `flux` | `ncu * nd` | Physical flux tensor. |
| `Source` | `source` | `ncu` | Volume source term. |
| `Tdfunc` | `tdfunc` | `ncu` | Time-derivative coefficient/mass term. |
| `Avfield` | `avfield` | model-dependent | Artificial-viscosity field. |
| `Output` | `output` | `nce` | User-defined DG output field. |
| `Monitor` | `monitor` | model-dependent | Runtime monitor field. |

Text2Code form:

```text
function Flux(x, uq, v, w, eta, mu, t)
  output_size(f) = ncu_times_nd;
  ...
end
```

C++ model form:

```cpp
KOKKOS_INLINE_FUNCTION static
void flux(dstype f[], const dstype x[], const dstype uq[],
          const dstype v[], const dstype w[], const dstype mu[],
          const dstype uinf[], dstype t);
```

## Boundary Functions

| Text2Code | C++ model name | Output size | Purpose |
| --- | --- | --- | --- |
| `Ubou` | `ubou` | `ncu` | Boundary state/value. |
| `Fbou` | `fbou` | `ncu` | LDG boundary flux. |
| `FbouHdg` | `fbou_hdg` | `ncu` | HDG boundary residual/flux. |
| `QoIboundary` | `qoi_boundary` | boundary QoI count | Boundary QoI integrand. |

Text2Code form:

```text
function FbouHdg(x, uq, v, w, uhat, n, tau, eta, mu, t)
  output_size(fb) = ncu;
  ...
end
```

C++ model form:

```cpp
KOKKOS_INLINE_FUNCTION static
void fbou_hdg(dstype fb[], int ib,
              const dstype x[], const dstype uq[],
              const dstype v[], const dstype w[], const dstype uh[],
              const dstype n[], const dstype tau[],
              const dstype mu[], const dstype uinf[], dstype t);
```

## Initial Conditions

| Text2Code | C++ model name | Output size | Purpose |
| --- | --- | --- | --- |
| `Initu` | `initu` | `ncu` | Initial primary state. |
| `Initq` | `initq` | `ncu * nd` | Initial gradient/auxiliary state. |
| `Inituq` | `initudg` | `nc` | Initial packed state. |
| `Initw` | `initwdg` | `ncw` | Initial auxiliary `w`. |
| `Initv` | `initodg` | `nco` | Initial external/other-DG field. |

Initial-condition functions are important for parameter sweeps: if `sol.bin`
does not contain a field, Exasim computes it using the active `physicsparam`
for that run.

## Auxiliary, EOS, and Visualization Functions

| Text2Code | C++ model name | Output size | Purpose |
| --- | --- | --- | --- |
| `Sourcew` | `sourcew` | `ncw` | Auxiliary `w` source/evolution term. |
| `EoS` | `eos` | model-dependent | Closure/derived state. |
| `VisScalars` | `vis_scalars` | `nsca` | Scalar VTK fields. |
| `VisVectors` | `vis_vectors` | vector fields | Vector VTK fields. |
| `VisTensors` | `vis_tensors` | tensor fields | Tensor VTK fields. |
| `QoIvolume` | `qoi_volume` | `nvqoi` | Volume QoI integrand. |

## Coupling and Interface Functions

| Text2Code | C++ model name | Purpose |
| --- | --- | --- |
| `Fint` | interface provider callback | Interior/coupled-interface flux. |
| `Fext` | interface provider callback | External interface flux. |
| Provider `Fhat` | `fhat` | Coupled numerical flux hook. |
| Provider `Uhat` | `uhat` | Coupled trace hook. |
| Provider `Stab` | `stab` | Coupled stabilization hook. |

These are advanced hooks used by coupled and multi-model workflows.

## Jacobian Functions

HDG and derivative-enabled paths require Jacobian callbacks. Generated
Text2Code providers derive these from declarations such as:

```text
jacobian uq, w, uhat
```

Direct C++ models may override functions such as:

- `flux_jac_uq`
- `flux_jac_w`
- `source_jac_uq`
- `source_jac_w`
- `fbou_jac_uq`
- `fbou_jac_uh`
- `fbou_jac_w`
- `ubou_jac_uq`
- `ubou_jac_uh`
- `ubou_jac_w`
- `fbou_hdg_jac_uq`
- `fbou_hdg_jac_uh`
- `fbou_hdg_jac_w`

For HDG, incorrect or zero Jacobians can cause Newton or GMRES failure even if
the value functions are correct.

## ModelC, ModelD, and ModelW

| Model type | Function implications |
| --- | --- |
| ModelC | `Flux` and `Source` depend primarily on `u`, optional `w`, `v`, and `mu`; no `q + grad(u)` equation is introduced. |
| ModelD | `Flux` may depend on `q` entries in `uq`; suitable for diffusion and mixed systems. |
| ModelW | `uq` includes wave auxiliary variables; `Tdfunc` and initialization must be consistent with wave evolution. |

See [Physics Models](../physics-models/index.md) for mathematical details.
