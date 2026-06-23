# Equations Of State

Equation-of-state callbacks provide closures and derived quantities. They are
not a separate primary PDE formulation.

Generic form:

$$
y = \mathcal{E}(u,q,w,v,x,t;\mu).
$$

In frontend models the callback is `eos(...)`; in Text2Code the function is
`EoS`; in the generated model contract the method is `eos`. The generated
backend also supports derivatives with respect to `u` and `w` through
`EoSdu`/`EoSdw` or `eos_du`/`eos_dw` when generated.

## Typical Outputs

- Pressure.
- Temperature.
- Enthalpy.
- Sound speed.
- Transport coefficients.
- Reaction rates.
- Material constitutive quantities.

## Typical EOS Families

- Ideal gas EOS.
- Thermally perfect gas EOS.
- Real-gas EOS.
- Plasma EOS.
- Material constitutive laws.

## Interaction With Other Model Terms

EOS outputs may be used by:

- Flux functions.
- Source functions.
- Auxiliary `w` equations.
- Boundary conditions.
- Artificial-viscosity sensors or coefficients.
- Visualization and QoI functions.

Repository examples include reacting-flow and high-temperature-gas models such
as `examples/Euler/reacting_flow`, `examples/ReactingFlows/reactingcylinder`,
and `frontends/Matlab/Modeling/CNS5air`.
