# ModelW

`ModelW` is intended for wave-propagation PDE systems. It uses the same packed
state layout as `ModelD`, but the auxiliary block satisfies a time-dependent
equation.

Canonical system:

$$
\frac{\partial q}{\partial t} + \nabla u = 0,
$$

$$
\frac{\partial u}{\partial t}
+ \nabla \cdot F(u,q,w,v,x,t;\mu)
= S(u,q,w,v,x,t;\mu).
$$

In preprocessing, `ModelW` sets `tdep = 1`, `wave = 1`,
`nc = ncu * (nd + 1)`, and `ncq = ncu * nd`.

## Difference From ModelD

`ModelD` uses \(q + \nabla u = 0\) as an algebraic mixed relation. `ModelW`
uses \(\partial q/\partial t + \nabla u = 0\), so the `q` block is part of the
time-evolved wave state.

## Required And Optional Functions

For frontend-generated models, define:

- `initu` for the initial `u` state.
- `initq` for the initial `q` state.
- `initw` for wave examples that use `w`; the current MATLAB/Julia generation
  paths expect `initw` for `ModelW`.
- `mass` to generate `Tdfunc` when time dependence is active.

Text2Code uses the corresponding `Initu`, `Initq` or `Inituq`, `Initw`, and
`Tdfunc` functions.

## Representative Examples

| Example | `u` | `q` | Notes |
| --- | --- | --- | --- |
| Acoustic wave | Pressure-like variable | Velocity/gradient-like wave variable | First-order acoustic system. |
| Elastic wave | Displacement/stress-related variables | Velocity/gradient-like variables | Depends on chosen first-order form. |
| Electromagnetic wave | Field component block | Companion wave block | User supplies Maxwell-type fluxes. |
| General wave system | Primary wave variables | Time-evolved auxiliary block | Requires consistent `initu` and `initq`. |

Repository examples include `examples/WaveEquation/Membrane` and
`examples/WaveEquation/Scattering`.

## When To Choose ModelW

Choose `ModelW` when:

- The model is fundamentally a first-order wave system.
- The `q` block is not just a recovered gradient.
- You need explicit initial conditions for both `u` and `q`.

Do not choose `ModelW` solely because the problem is time dependent. Many
time-dependent diffusion and flow problems are still [`ModelD`](modeld.md) or
[`ModelC`](modelc.md).
