# Validating AI-Generated Code

AI-generated Exasim code must be validated like any other scientific-computing
code. Compilation is necessary but not sufficient. The output must be
physically and numerically correct.

## Validation Checklist

| Check | Purpose |
| --- | --- |
| Compile from a clean build directory | Detect stale cache and missing dependencies. |
| Run a small serial CPU case | Establish a simple baseline. |
| Check residual convergence | Verify nonlinear and linear solver behavior. |
| Inspect mesh and boundary tags | Catch geometry and boundary-condition errors. |
| Compare to an analytical solution | Validate PDE terms when possible. |
| Use a manufactured solution | Validate flux/source/boundary consistency. |
| Compare to a benchmark example | Check practical behavior for complex physics. |
| Run MPI if supported | Verify rank-local files and communication. |
| Run GPU if supported | Check host/device data updates and backend-specific builds. |
| Review output files | Confirm VTK, QoI, residuals, and parameter metadata are correct. |

## Manufactured Solutions

For new PDE terms, ask AI to help create a manufactured solution:

```text
For this Exasim ModelD PDE, propose a manufactured solution u(x,y) and derive
the source term needed to make it exact. Then update pdemodel.m and pdeapp.m
to run a convergence test over three mesh resolutions.
```

Review the derivation manually. AI can make algebra mistakes.

## Mesh Validation

Before solving:

- plot the mesh;
- check boundary tags;
- run low order first;
- verify no inverted elements;
- confirm curved geometry if used.

## Solver Validation

Track:

- Newton residual norm;
- GMRES iterations;
- preconditioner behavior;
- time-step stability;
- conservation diagnostics when available;
- physical bounds such as density, pressure, temperature, or displacement.

## GPU And MPI Validation

For CPU/GPU agreement:

```text
Run the same small Exasim case on CPU and GPU. Compare residual histories,
solution norms, and key output files within an appropriate tolerance.
Investigate any difference larger than expected roundoff/backend variation.
```

For MPI agreement:

```text
Run with mpiprocs = 1 and mpiprocs = 2 on the same small mesh.
Compare final solution norms and output fields.
Check rank-indexed dataout files and VTK wrapper files.
```

## Production Rule

Do not use AI-generated model, mesh, backend, or GPU code for production until
it passes a documented validation path appropriate for the target physics.
