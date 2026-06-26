# Nonlinear Solvers

Exasim solves nonlinear algebraic systems with Newton iteration. Nonlinearity
can come from fluxes, sources, EOS closures, boundary conditions, artificial
viscosity, auxiliary equations, or implicit time integration.

## Newton Formulation

Let \(U\) denote the unknown vector for the active solver path:

- LDG: element state unknowns;
- HDG: trace unknowns after static condensation.

The nonlinear residual is

\[
R(U)=0.
\]

At Newton iteration \(k\), the correction \(\Delta U^k\) satisfies

\[
J(U^k)\Delta U^k = -R(U^k),
\]

where

\[
J(U^k) = \frac{\partial R}{\partial U}(U^k).
\]

The state is updated as

\[
U^{k+1} = U^k + \alpha \Delta U^k,
\]

where \(\alpha\) may be reduced by the solver when damping is required.

## Sources Of Nonlinearity

| Source | Example |
| --- | --- |
| Convective fluxes | Euler and Navier-Stokes equations. |
| Diffusive fluxes | Temperature- or state-dependent viscosity. |
| Source terms | Chemistry, relaxation, body forces. |
| EOS and material laws | Pressure, temperature, transport coefficients. |
| Boundary fluxes | Wall, inflow, outflow, and Riemann boundary states. |
| DIRK stages | Implicit stage residuals for time-dependent simulations. |

## LDG Versus HDG Newton Systems

| Solver path | Newton unknown | Jacobian action |
| --- | --- | --- |
| LDG | Element state vector | Matrix-free residual-based matrix-vector products. |
| HDG | Trace vector after condensation | Matrix-based trace operator assembled from element Jacobians. |

The nonlinear convergence criteria are controlled by fields such as
`NewtonIter`, `NewtonTol`, `NLiter`, and `NLtol`, depending on frontend or text
input path.

## Convergence Diagnostics

Solver output typically reports the solution norm, residual norm, GMRES
iterations, and Newton update quality. Use these together:

- residual norm decreasing but GMRES stagnating suggests a linear solver or
  preconditioner issue;
- GMRES convergence with poor Newton progress suggests nonlinear stiffness,
  bad initial conditions, or inconsistent model Jacobians;
- sudden residual growth often indicates invalid physics states, boundary
  condition errors, or excessive time step size.

## Practical Guidance

- Start with conservative nonlinear tolerances when validating a new model.
- Verify boundary conditions and flux Jacobians before tuning GMRES.
- For stiff parameter sweeps, use warm starts only when neighboring parameter
  points are physically close.
- For time-dependent runs, distinguish a failed DIRK stage from a failed
  steady-state nonlinear solve; the remedies may differ.
