# Linear Solvers

Newton's method requires solving a linear system at each nonlinear iteration:

\[
J\Delta U = -R.
\]

Exasim primarily uses GMRES for these nonsymmetric systems. DG discretizations,
implicit time integration, boundary conditions, and stabilization terms usually
produce nonsymmetric Jacobians, so GMRES is a natural Krylov method.

## GMRES

GMRES builds a Krylov subspace

\[
\mathcal{K}_m(J,r_0) =
\text{span}\{r_0, Jr_0, J^2r_0,\ldots,J^{m-1}r_0\}
\]

and chooses the correction that minimizes the residual over this subspace.
Restarted GMRES limits memory and orthogonalization cost by restarting after
`GMRESrestart` basis vectors.

## Exasim Controls

| Field | Role |
| --- | --- |
| `GMRESiter` | Maximum total Krylov iterations. |
| `GMRESrestart` | Restart length. |
| `GMREStol` | Linear solver tolerance. |
| `GMRESortho` | Orthogonalization selector. |
| `linearsolver` | Linear solver selector. |
| `linearsolveriter` | Frontend-facing maximum iteration count. |
| `linearsolvertol` | Frontend-facing tolerance. |

## Matrix-Free LDG

For LDG, Exasim uses residual evaluations to form matrix-vector products:

\[
Jv \approx \frac{R(U+\epsilon v)-R(U)}{\epsilon}.
\]

The perturbation size and approximation order are controlled by fields such as
`matvectol` and `matvecorder`. This design avoids storing the full LDG Jacobian,
but GMRES convergence depends strongly on preconditioning.

## Matrix-Based HDG

For HDG, local element Jacobian blocks are assembled, statically condensed, and
used to build a trace-space operator. GMRES then operates on trace unknowns.
This costs more memory per element than the matrix-free LDG path, but the global
Krylov vector dimension is reduced to face trace unknowns.

## Polynomial Preconditioning

When `ppdegree > 1`, the GMRES path can apply a polynomial preconditioner. The
backend constructs polynomial data from Krylov/Hessenberg information and uses
the resulting polynomial action together with the selected base preconditioner.

## Reduced-Basis Acceleration

The `RBdim` field controls storage for a reduced basis used by the solver and
preconditioner path. It is intended to reuse recent correction information to
improve the initial guess and preconditioner action. Its effectiveness is
problem-dependent and should be validated with residual histories.

## Practical Guidance

- Increase `GMRESrestart` only when memory allows; orthogonalization cost grows
  with restart length.
- Prefer improving the preconditioner before allowing very large GMRES
  iteration counts.
- For LDG, matrix-free products reduce storage but can increase residual
  evaluations.
- For HDG, trace-system assembly costs must be balanced against fewer global
  unknowns.
