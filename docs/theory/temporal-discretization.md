# Temporal Discretization

Exasim supports steady-state and time-dependent workflows. The runtime data
structure contains a `temporalScheme` selector with values for DIRK, BDF, and
ERK, while the most common implicit time-dependent workflow is based on
diagonally implicit Runge-Kutta (DIRK) stages.

## Steady Problems

For a steady problem, Exasim solves the spatial residual

\[
R(U)=0.
\]

Newton-GMRES is applied directly to the nonlinear algebraic system produced by
the LDG or HDG spatial discretization.

## Method Of Lines

For time-dependent PDEs, spatial discretization produces an ODE-like system

\[
M \frac{dU}{dt} = G(U,t),
\]

where \(M\) is the DG mass operator and \(G\) includes volume fluxes, face
fluxes, sources, auxiliary equations, and boundary terms.

## DIRK Methods

A DIRK method advances from \(U^n\) to \(U^{n+1}\) through stages. With Butcher
coefficients \(a_{ij}\), \(b_i\), and \(c_i\), each stage solves an implicit
problem of the form

\[
U_i =
U^n + \Delta t \sum_{j=1}^{i} a_{ij} G(U_j,t^n+c_j\Delta t).
\]

Because \(a_{ii}\neq 0\), the current stage \(U_i\) appears implicitly and
requires a nonlinear solve. The stage residual can be written schematically as

\[
R_i(U_i) =
M\left(U_i - U^n - \Delta t \sum_{j<i} a_{ij}G(U_j)\right)
- \Delta t\,a_{ii}G(U_i)=0.
\]

Exasim stores DIRK coefficients in runtime arrays such as `DIRKcoeff_c`,
`DIRKcoeff_d`, and `DIRKcoeff_t`, and loops over stages through
`currentstage`.

## Why DIRK Is Used

DIRK methods are attractive for Exasim applications because they:

- handle stiff source terms and diffusion more robustly than explicit schemes;
- require only one implicit stage solve at a time;
- reuse the same Newton-GMRES machinery used for steady problems;
- avoid fully coupled all-stage systems;
- support high-order temporal accuracy through `torder` and `nstage`.

## BDF And ERK Selectors

The backend runtime carries selector values for BDF and ERK in
`temporalScheme`. Documentation and examples should be checked for the specific
application before relying on a non-DIRK path, because DIRK is the primary
implicit workflow described in the current user documentation.

## User Controls

| Field | Meaning |
| --- | --- |
| `tdep` | `0` for steady, `1` for time-dependent execution. |
| `temporalscheme` | Time integration selector. |
| `torder` | Temporal order requested by the app. |
| `nstage` | Number of DIRK stages. |
| `dt` | Time-step values. |
| `saveSolFreq` | Output frequency for time-dependent results. |
| `timestepOffset` | Offset used when continuing or postprocessing time histories. |

## Practical Guidance

- Use steady mode for residual convergence to a stationary solution.
- Use DIRK for stiff time-dependent flows, diffusion, reacting systems, and
  multiphysics problems.
- Reduce `dt` or increase nonlinear tolerance strictness when Newton fails in a
  stage.
- Use saved residual histories and postprocessing to verify temporal behavior,
  not only final-state convergence.
