# ModelC

`ModelC` is intended for convection-type and first-order PDE systems. It does
not introduce a `q` equation.

Canonical strong form:

$$
\frac{\partial u}{\partial t}
+ \nabla \cdot F(u,w,v,x,t;\mu)
= S(u,w,v,x,t;\mu).
$$

In preprocessing, `ModelC` sets `wave = 0`, `nc = ncu`, and `ncq = 0`.
Consequently, `q` is empty in frontend callbacks and `uq` contains only `u` in
Text2Code models.

## Meaning Of The Variables

| Quantity | Meaning |
| --- | --- |
| `u` | Primary transported or conserved variables. |
| `w` | Optional auxiliary state variables. |
| `v` | Optional externally supplied nodal variables. |
| `physicsparam` / `mu` | Physical parameters used by fluxes, sources, EOS, and boundary terms. |
| `q` | Not present for `ModelC`. |

## User-Defined Components

| Component | Frontend callback | Text2Code function |
| --- | --- | --- |
| Fluxes | `flux(u,q,w,v,x,t,mu,eta)` | `Flux(x, uq, v, w, eta, mu, t)` |
| Sources | `source(...)` | `Source(...)` |
| Boundary state | `ubou(...)` | `Ubou(...)` |
| Boundary flux | `fbou(...)`, `fbouhdg(...)` | `Fbou(...)`, `FbouHdg(...)` |
| Initial condition | `initu(x,mu,eta)` | `Initu(x, eta, mu)` |
| EOS closure | `eos(...)` | `EoS(...)` |
| AV field | `avfield(...)` | `Avfield(...)` |

## Representative Examples

| Example | `u` | Fluxes | Sources | EOS / optional state |
| --- | --- | --- | --- | --- |
| Linear advection | Scalar concentration | \(F = a u\) | Usually zero | Optional prescribed velocity through `v`. |
| Hyperbolic conservation law | Conserved state | User-defined conservative flux | Reaction or forcing | Optional `w` for additional state. |
| Compressible Euler | Density, momentum, energy | Inviscid Euler flux | Body forces or reactions | EOS for pressure and sound speed. |
| Reactive transport without gradients | Species or progress variables | Convective flux | Reaction source | `w` may store auxiliary chemistry state. |

Examples in the repository include
`examples/Advection/GaussianRotating`, `examples/Advection/GaussianTransport`,
`examples/Euler/EulerVortex`, `examples/Euler/naca0012`, and
`examples/MHD/MagneticVortex`.

## When To Choose ModelC

Choose `ModelC` when:

- The PDE can be written as a first-order conservation or transport law in `u`.
- You do not need Exasim to represent gradients as `q`.
- Diffusion, viscosity, or gradient-dependent closures are absent or handled
  externally.
- You want the smallest state layout for a convection-dominated problem.

Do not choose `ModelC` if the physical flux naturally depends on \(\nabla u\).
Use [`ModelD`](modeld.md) for diffusion, viscosity, and mixed formulations.
