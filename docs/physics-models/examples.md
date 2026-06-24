# Worked Physics-Model Examples

This page maps common PDEs to Exasim model choices and callback functions. The
examples are formulation guides, not complete application listings.

## 1. Linear Advection

$$
\partial_t u + a \cdot \nabla u = 0.
$$

| Item | Mapping |
| --- | --- |
| Model type | `ModelC` |
| `u` | Scalar transported quantity. |
| `q` | Not present. |
| `w` | Usually absent. |
| `v` | Optional prescribed velocity field. |
| Flux | `flux`: \(F = a u\). |
| Source | `source`: zero or forcing. |
| Boundary/IC | `ubou`, `fbou`, `fbouhdg`, `initu`. |
| Examples | `examples/Advection/GaussianRotating`, `examples/Advection/GaussianTransport`. |

## 2. Compressible Euler

| Item | Mapping |
| --- | --- |
| Model type | `ModelC` |
| `u` | Conservative density, momentum, and energy. |
| `q` | Not present. |
| EOS | Pressure, temperature, sound speed from `eos` or helper functions. |
| Flux | Inviscid Euler flux. |
| Source | Body force or reaction source if present. |
| AV | Optional for shocks. |
| Examples | `examples/Euler/EulerVortex`, `examples/Euler/naca0012`, `examples/Euler/mach8cylinder`. |

## 3. Heat Equation

$$
\partial_t u - \nabla \cdot (\kappa \nabla u) = s.
$$

| Item | Mapping |
| --- | --- |
| Model type | `ModelD` |
| `u` | Temperature or scalar field. |
| `q` | Gradient variable with \(q + \nabla u = 0\). |
| Flux | Conductive flux written using `q`, for example \(F = \kappa q\). |
| Source | Heat source. |
| Examples | `examples/HeatEquation/warmup`, `examples/HeatEquation/cooldown`. |

## 4. Advection-Diffusion

| Item | Mapping |
| --- | --- |
| Model type | `ModelD` |
| `u` | Concentration or scalar transported field. |
| `q` | Gradient for diffusive flux. |
| Flux | Convective plus diffusive flux. |
| Source | Reaction or forcing. |
| Examples | `examples/ConvectionDiffusion/1D`, `examples/ConvectionDiffusion/BoundaryLayer`. |

## 5. Compressible Navier-Stokes

| Item | Mapping |
| --- | --- |
| Model type | `ModelD` |
| `u` | Conservative flow state. |
| `q` | Gradient block used for viscous fluxes. |
| EOS | Pressure, temperature, sound speed, transport coefficients. |
| Flux | Inviscid plus viscous flux. |
| Source | Body force, chemistry, or axisymmetric terms if present. |
| AV | Optional for shocks/hypersonic examples. |
| Examples | `apps/navierstokes/naca0012steady`, `examples/NavierStokes/flatplate2d`. |

## 6. Acoustic Wave Equation

| Item | Mapping |
| --- | --- |
| Model type | `ModelW` |
| `u` | Pressure-like or displacement-like state. |
| `q` | Time-evolved wave auxiliary state. |
| Flux/source | First-order wave-system terms. |
| Initial data | `initu`, `initq`, and `initw` in current wave examples. |
| Examples | `examples/WaveEquation/Membrane`, `examples/WaveEquation/Scattering`. |

## 7. Reactive Flow With Auxiliary ODEs

| Item | Mapping |
| --- | --- |
| Model type | `ModelC` or `ModelD`, depending on viscosity/diffusion. |
| `u` | Flow conservative variables. |
| `q` | Present for viscous/diffusive terms. |
| `w` | Chemistry or nonequilibrium auxiliary state. |
| EOS | Depends on `u` and `w`. |
| Source | Flow source plus `sourcew` for auxiliary state. |
| Examples | `examples/Euler/reacting_flow`, `examples/ReactingFlows/reactingcylinder`. |

## 8. Shock-Capturing Compressible Flow With AV

| Item | Mapping |
| --- | --- |
| Model type | `ModelC` for inviscid, `ModelD` for viscous. |
| AV | `AV` flags, `avfield`, and/or AV data in `v`. |
| Flux | Physical flux augmented by AV contributions. |
| Examples | `examples/Euler/mach8cylinder`, `examples/NavierStokes/flaredplate2d`. |

## 9. Multiphysics Example

| Item | Mapping |
| --- | --- |
| Model type | Usually `ModelD` for coupled transport/diffusion, or `ModelC` for first-order systems. |
| `u` | Packed coupled primary state. |
| `w` | Auxiliary chemistry, thermal, or material state. |
| `v` | Prescribed or lagged coupled fields. |
| EOS | Shared closure across fields. |
| Examples | `examples/GSI/*`, `examples/MHD/MagneticVortex`, high-temperature gas modeling code. |

## 10. Multi-Model Example

| Item | Mapping |
| --- | --- |
| Model type | Multiple model definitions or a monolithic packed model. |
| Coupling | Source, boundary/interface, external-field, or auxiliary-state coupling. |
| Data ownership | Each exchanged field should have one owner and documented consumers. |
| Implementation | Use `modelnumber`, multiple model definitions, `v`, `w`, `Fint`/`Fext`, or source terms as appropriate. |
