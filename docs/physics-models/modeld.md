# ModelD

`ModelD` introduces auxiliary gradient variables through

$$
q + \nabla u = 0.
$$

The canonical system is

$$
q + \nabla u = 0,
$$

$$
\frac{\partial u}{\partial t}
+ \nabla \cdot F(u,q,w,v,x,t;\mu)
= S(u,q,w,v,x,t;\mu).
$$

In preprocessing, `ModelD` sets `wave = 0`,
`nc = ncu * (nd + 1)`, and `ncq = ncu * nd`. The packed state is
`uq = [u, q]`.

## Meaning Of q

`q` is the mixed or LDG/HDG auxiliary variable associated with the gradient of
`u`. It is not time evolved as an independent wave state in `ModelD`; it is
introduced so diffusion, viscosity, and higher-order PDEs can be written as a
first-order system.

## User-Defined Components

| Component | Role |
| --- | --- |
| `u` | Primary unknowns. |
| `q` | Gradient block with `ncu * nd` entries. |
| `flux` / `Flux` | Convective and diffusive fluxes, usually depending on `q`. |
| `source` / `Source` | Volume forcing, reaction, or coupling source. |
| `mass` / `Tdfunc` | Required for time-dependent frontend generation and required in Text2Code default output lists. |
| `ubou`, `fbou`, `fbouhdg` | Boundary state and boundary flux definitions. |
| `initu` / `Initu` | Initial state for `u`. |
| `initq` / `Initq` | Optional initial gradient data; not generally required for `ModelD`. |
| `eos`, `sourcew`, `avfield` | Optional closures, auxiliary equations, and stabilization. |

## Representative Examples

| Example | `u` | `q` | Flux/source role |
| --- | --- | --- | --- |
| Poisson | Scalar potential | Gradient of potential | Diffusive flux and forcing. |
| Heat equation | Temperature | Temperature gradient | Conductive heat flux. |
| Advection-diffusion | Concentration | Concentration gradient | Advection plus diffusion. |
| Compressible Navier-Stokes | Conservative flow state | Gradient block used by viscous fluxes | Inviscid + viscous flux, EOS closure. |
| Reactive viscous flow | Flow state plus optional `w` | Gradients for viscous/diffusive terms | Chemistry source and transport closures. |

Examples in the repository include `apps/poisson/poisson2d`,
`examples/Poisson/poisson2d`, `examples/HeatEquation/warmup`,
`examples/ConvectionDiffusion/BoundaryLayer`,
`examples/NavierStokes/flatplate2d`, and
`apps/navierstokes/naca0012steady`.

## When To Choose ModelD

Choose `ModelD` when:

- The PDE contains diffusion, viscosity, conductivity, or gradient-dependent
  physics.
- You are using HDG/LDG mixed formulations.
- A higher-order PDE should be rewritten as a first-order system.
- The flux needs both `u` and `q`.

Use [`ModelW`](modelw.md) instead when the `q` block is physically
time-evolved as part of a wave system.
