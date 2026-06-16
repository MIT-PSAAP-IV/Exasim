# Theory

Mathematical background for the discretization Exasim implements. Read this
before the [model contract](../reference/model-contract.md): the contract methods
are the pointwise terms of the formulations below, so the math is what each
`flux` / `source` / boundary / Jacobian method actually supplies.

| Page | Covers | Maps to (contract) |
|---|---|---|
| [LDG formulation](ldg-formulation.md) | first-order rewrite, weak $q$ and $u$ equations, volume/face terms, numerical fluxes | `flux`, `source`, `fbou`, `ubou`, `initu` |
| [Block-diagonal Jacobian](block-diagonal-jacobian.md) | exact element-block Jacobian of the LDG residual, chain rule through the pipeline | `flux_jac_*`, `source_jac_*`, `fbou_hdg_jac_*` |

## DG, HDG, LDG

Exasim discretizes in space with discontinuous Galerkin (DG) methods on a
first-order form of the PDE. Two variants share most of the machinery:

- **LDG** (local DG) — the auxiliary gradient variable $q$ is recovered
  *locally* per element, then the $u$ residual is assembled from volume and face
  terms. This is the path derived in detail on the
  [LDG formulation](ldg-formulation.md) page.
- **HDG** (hybridized DG) — introduces a trace unknown $\hat{u}$ on the mesh
  skeleton and a global trace system, with element-local static condensation.
  The pointwise Jacobians the HDG solver needs are exactly the blocks derived on
  the [block-diagonal Jacobian](block-diagonal-jacobian.md) page.

Time integration uses diagonally implicit Runge–Kutta (DIRK); the nonlinear
system at each stage is solved with Newton + GMRES. See
[Internals → Architecture](../internals/architecture.md) for how the formulation
flows through preprocessing, assembly, and the solver.
