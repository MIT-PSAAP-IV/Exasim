# Discontinuous Galerkin Methods

Discontinuous Galerkin (DG) methods approximate the solution by independent
polynomials on each mesh element. The approximation space is broken:

\[
V_h = \{ v : v|_K \in \mathcal{P}_p(K) \quad \text{for each element } K \}.
\]

For a state \(u\), Exasim stores element-local coefficients \(U_K\) such that

\[
u_h(x)|_K = \sum_j U_{Kj}\phi_j(x).
\]

Neighboring elements do not share nodal degrees of freedom. Coupling occurs
through numerical fluxes on faces.

## Element-Local Weak Form

For a conservation law

\[
\frac{\partial u}{\partial t} + \nabla \cdot F(u) = S(u),
\]

the DG weak form on an element \(K\) is obtained by multiplying by a test
function \(\phi\), integrating by parts, and replacing the physical face flux
with a numerical flux \(F^\ast\):

\[
\int_K \phi \frac{\partial u_h}{\partial t}\,dx
- \int_K \nabla \phi \cdot F(u_h)\,dx
+ \int_{\partial K} \phi F^\ast \cdot n\,ds
= \int_K \phi S(u_h)\,dx.
\]

The numerical flux is responsible for stability, conservation across element
interfaces, and boundary-condition enforcement.

## Why DG Fits Exasim

DG methods match Exasim's code-generation and HPC design:

| Property | Practical effect |
| --- | --- |
| Element-local polynomial data | Most kernels operate on compact element batches. |
| Local residual evaluation | Flux, source, EOS, AV, and visualization callbacks are pointwise or element-local. |
| High arithmetic intensity | High-order elements perform substantial work per loaded degree of freedom. |
| Flexible meshes | Curved, unstructured, and boundary-fitted meshes fit naturally. |
| Conservative face coupling | Interface fluxes preserve conservation when model fluxes are consistent. |
| GPU suitability | Element and quadrature loops map to many independent GPU work items. |

## LDG And HDG

Exasim uses two DG solver paths:

- [LDG](ldg.md) keeps element solution unknowns as the primary global state and
  uses matrix-free Newton-GMRES.
- [HDG](hdg.md) introduces trace unknowns on the mesh skeleton, statically
  condenses element unknowns, and solves a matrix-based global trace problem.

Both paths use the same model callbacks for fluxes, sources, boundary terms,
initial conditions, EOS, artificial viscosity, QoI, and visualization fields.

## Conservation And Consistency

DG conservation depends on using compatible interior and boundary fluxes. For an
interior face shared by elements \(K^-\) and \(K^+\), the numerical flux must be
single-valued on the face. Conservation follows because the same face flux enters
the two neighboring residuals with opposite normal directions.

Consistency means that when the traces from both sides match the exact
solution, \(F^\ast\) reduces to the physical flux. In Exasim, this consistency
is encoded in the user/model boundary and interface callbacks.

## Practical Guidance

Choose DG/LDG/HDG settings together with the PDE model:

| Need | Recommended direction |
| --- | --- |
| Lowest setup complexity and matrix-free iterations | LDG |
| Diffusion-dominated HDG formulations with fewer global unknowns | HDG |
| Large implicit simulations where trace systems are cheaper than full state systems | HDG |
| Problems where matrix-free residual products are preferred over assembled matrices | LDG |
| GPU-heavy runs with high-order local kernels | Either LDG or HDG; benchmark with the target preconditioner |

See also [Physics Models](../physics-models/index.md) for choosing `ModelC`,
`ModelD`, or `ModelW`.
