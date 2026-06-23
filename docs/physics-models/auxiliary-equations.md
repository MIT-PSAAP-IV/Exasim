# Auxiliary Equations Through w

The `w` variables are auxiliary state variables independent of `ModelC`,
`ModelD`, and `ModelW`. They can be used with any primary model class.

A useful abstraction is

$$
\alpha \frac{\partial w}{\partial t}
+ \beta w
= \mathcal{G}(u,q,w,v,x,t;\mu).
$$

The exact interpretation is supplied by the model callbacks. Exasim provides
the storage and generated kernels; the model author defines the physics.

## Supported Callback Mapping

| Purpose | Frontend callback | Text2Code function | Generated/contract name |
| --- | --- | --- | --- |
| Initialize `w` | `initw(x,mu,eta)` | `Initw` / `Initwdg` | `initwdg` |
| Source/update for `w` | `sourcew(u,q,w,v,x,t,mu,eta)` | `Sourcew` | `sourcew` |
| EOS dependence on `w` | `eos(...)` | `EoS` | `eos`, `eos_dw` |

## Common Forms

| Form | Equation | Typical use |
| --- | --- | --- |
| Algebraic auxiliary equation | \(\beta w = \mathcal{G}\) | Equilibrium closures or material state. |
| Auxiliary ODE | \(\partial w/\partial t = \mathcal{G}\) | Chemistry, turbulence, internal variables. |
| Relaxation model | \(\tau \partial w/\partial t + w = \mathcal{G}\) | Nonequilibrium or memory models. |
| ADE model | Same callback structure | Dispersive or constitutive auxiliary differential equations. |

## Examples

- Chemical kinetics in `examples/Euler/reacting_flow` and
  `examples/ReactingFlows/reactingcylinder`.
- High-temperature gas modeling under `frontends/Matlab/Modeling/CNS5air`.
- Turbulence-like or internal material variables supplied through `w`.
- Vibrational nonequilibrium and multi-temperature state variables.
- Constitutive relaxation and memory effects.

`w` may enter fluxes, sources, EOS, boundary conditions, initial conditions,
QoI, and visualization callbacks.
