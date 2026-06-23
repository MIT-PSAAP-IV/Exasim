# Hybridizable Discontinuous Galerkin

Hybridizable Discontinuous Galerkin (HDG) introduces an independent trace
unknown \(\hat{u}\) on the mesh skeleton. Element unknowns are solved locally in
terms of \(\hat{u}\), and the global problem enforces flux continuity across
faces.

For a first-order diffusion-type system,

\[
q + \nabla u = 0,
\]

\[
\frac{\partial u}{\partial t}
+ \nabla \cdot F(u,q,w,v,x,t;\mu)
= S(u,q,w,v,x,t;\mu),
\]

HDG seeks element fields \((q_h,u_h)\) and a trace \(\hat{u}_h\) such that the
element equations and face flux-continuity equations are satisfied.

## Static Condensation

The core HDG idea is local elimination. For each element \(K\), the local
linearized system can be written schematically as

\[
\begin{bmatrix}
A_K & B_K \\
C_K & D_K
\end{bmatrix}
\begin{bmatrix}
\Delta y_K \\
\Delta \hat{u}_{\partial K}
\end{bmatrix}
=
\begin{bmatrix}
r_K \\
g_K
\end{bmatrix},
\]

where \(y_K\) contains element-local unknowns such as \(q_K\) and \(u_K\). The
element correction is condensed as

\[
\Delta y_K = A_K^{-1}(r_K - B_K \Delta\hat{u}_{\partial K}),
\]

and the global system is assembled only for trace corrections:

\[
H \Delta \hat{u} = b.
\]

After the trace solve, element states are recovered locally.

## Exasim HDG Implementation

In Exasim, HDG is the matrix-based path:

- element residuals and Jacobian blocks are evaluated from generated model
  callbacks;
- local matrices are condensed into face/trace contributions;
- GMRES solves the global trace system;
- preconditioners operate on trace-space blocks or Schwarz subproblems;
- local element states are updated after the trace correction.

The model contract for HDG requires boundary and flux Jacobians in addition to
the residual callbacks. See [Model contract](../reference/model-contract.md).

## Trace Unknowns

The trace \(\hat{u}\) lives on faces rather than inside elements. This reduces
the number of globally coupled unknowns for high-order elements because only
face degrees of freedom participate in the global system.

| Quantity | Location | Role |
| --- | --- | --- |
| \(u_h\) | Element interior | Primary solution reconstructed locally. |
| \(q_h\) | Element interior | Auxiliary gradient or flux variable. |
| \(\hat{u}_h\) | Mesh skeleton | Globally coupled trace unknown. |
| Numerical flux | Element faces | Enforces interface conservation and boundary conditions. |

## LDG And HDG Comparison

| Feature | LDG | HDG |
| --- | --- | --- |
| Global unknowns | Element state unknowns | Trace unknowns on faces |
| Auxiliary variable \(q\) | Local recovery inside residual path | Local element unknown condensed against trace |
| Matrix assembly | Matrix-free GMRES path | Matrix-based trace-system path |
| Matrix-vector product | Residual differencing or derivative path | Assembled/condensed trace operator |
| Memory footprint | Lower matrix storage, Krylov vectors over state DOFs | Stores element/trace blocks, Krylov vectors over trace DOFs |
| Preconditioning | Local and Schwarz-style state-space preconditioning | Block Jacobi and additive Schwarz trace-space preconditioning |
| Scalability tradeoff | More global DOFs but less matrix storage | Fewer global DOFs but more local assembly/factorization work |

## When To Choose HDG

Choose HDG when:

- diffusion or mixed formulations dominate the model;
- reducing global unknowns is more important than avoiding matrix assembly;
- trace-space preconditioning is effective for the target problem;
- high polynomial orders make element-local static condensation attractive;
- you need the solver path used by many production HDG examples.

Benchmark LDG and HDG for new applications. The best choice depends on
polynomial order, mesh size, physics stiffness, preconditioner quality, and
backend memory bandwidth.
