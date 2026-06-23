# Preconditioning

Preconditioning is critical for large Exasim simulations. High-order DG
operators, stiff source terms, diffusion, implicit time stepping, and
multiphysics coupling can all make the Newton linear systems ill-conditioned.

A preconditioner \(P^{-1}\) approximates \(J^{-1}\) cheaply. GMRES then works
with the transformed system rather than the raw Jacobian action.

## Implemented Families

The implemented preconditioner options exposed through `preconditioner` are:

| Value | Name | Notes |
| --- | --- | --- |
| `0` | Block Jacobi | Local block approximation with no inter-block coupling. |
| `1` | Elemental additive Schwarz | Element-level overlapping or local subproblem preconditioner. |
| `2` | Superelement additive Schwarz with block ILU(0) | Larger subdomains/superelements with block ILU-style local solves. |

The code also exposes:

- `ppdegree` for polynomial preconditioning;
- `RBdim` for reduced-basis solver/preconditioner storage.

Do not assume ILU, multigrid, or algebraic multigrid support beyond the
implemented block ILU(0)-style superelement Schwarz path unless it is present in
the target build.

## Block Jacobi

Block Jacobi approximates the inverse using independent local blocks:

\[
P^{-1} \approx \text{blockdiag}(J_1^{-1},J_2^{-1},\ldots).
\]

It is simple, parallel, and low communication, but it ignores long-range
coupling. It is often useful as a baseline or for problems where local stiffness
dominates.

## Elemental Additive Schwarz

Additive Schwarz combines local subdomain solves:

\[
P^{-1}r = \sum_s R_s^T A_s^{-1} R_s r,
\]

where \(R_s\) restricts to subdomain \(s\). In the elemental variant,
subdomains are associated with element-level or face-neighbor blocks. This
captures more coupling than block Jacobi while preserving locality.

## Superelement Additive Schwarz

The superelement option groups elements into larger blocks and applies a block
ILU(0)-style local solve. It can be more robust for difficult HDG trace systems,
but it requires more memory and setup time. It is most useful when the extra
local work reduces GMRES iterations enough to offset the cost.

## Polynomial Preconditioning

Polynomial preconditioning approximates inverse behavior by applying a
polynomial in the preconditioned operator:

\[
p_m(P^{-1}J) \approx (P^{-1}J)^{-1}.
\]

The backend constructs polynomial data from Krylov information and Leja-ordered
eigenvalue approximations. Increasing `ppdegree` may reduce GMRES iterations but
adds extra matrix-vector/preconditioner applications.

## Reduced-Basis Controls

The reduced-basis dimension `RBdim` controls how many recent basis vectors are
retained by the solver/preconditioner machinery. This can improve repeated
Newton or time-stage solves when the linear systems change gradually.

## Selection Guidance

| Situation | Candidate |
| --- | --- |
| First validation run | `preconditioner = 0` or the example default |
| HDG trace system with moderate stiffness | `preconditioner = 1` |
| Strongly coupled or poorly conditioned trace system | `preconditioner = 2` |
| GMRES iteration count remains high with stable base preconditioner | Try `ppdegree > 1` |
| Slowly varying sequence of solves | Tune `RBdim` |

## Risks

Preconditioners can change convergence behavior without changing the underlying
discretization. Always verify:

- nonlinear residual convergence;
- conservation and boundary behavior;
- sensitivity to polynomial order and mesh refinement;
- CPU/GPU consistency if changing backend or preconditioner.
