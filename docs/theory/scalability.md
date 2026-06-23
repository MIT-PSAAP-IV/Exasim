# Scalability

Scalability measures how Exasim performance changes as the problem size,
processor count, GPU count, polynomial order, or solver difficulty changes.

## Strong And Weak Scaling

| Scaling type | Definition | Main bottleneck |
| --- | --- | --- |
| Strong scaling | Fixed global problem size, more ranks/devices | Communication and synchronization dominate as local work shrinks. |
| Weak scaling | Problem size grows with rank/device count | Partition quality, communication surface area, and solver iteration growth. |

## Complexity Drivers

The dominant costs depend on discretization and physics:

- number of elements;
- polynomial order and quadrature order;
- number of state, gradient, auxiliary, and trace variables;
- nonlinear iterations;
- GMRES iterations and restart length;
- preconditioner setup/application cost;
- output frequency and filesystem load.

## LDG Scalability

LDG avoids storing a full global Jacobian, which helps memory scaling. The cost
is repeated residual evaluation for matrix-free Krylov products. LDG scales best
when:

- element-local residual work dominates communication;
- GMRES iteration count is controlled by preconditioning;
- matrix-vector products remain backend-resident;
- output frequency is modest.

## HDG Scalability

HDG reduces globally coupled unknowns to trace unknowns. This often improves
scalability for high-order diffusion and mixed systems, but local matrix
assembly, static condensation, and preconditioner storage can be significant.
HDG scales best when:

- trace DOF reduction offsets local matrix work;
- preconditioners reduce GMRES iterations robustly;
- subdomain partitions minimize inter-rank trace faces;
- local dense algebra is efficient on the target CPU/GPU.

## Memory Footprint

Memory is consumed by:

- solution and residual arrays;
- Krylov vectors, roughly proportional to `GMRESrestart + 1`;
- HDG local matrices and condensed trace blocks;
- preconditioner storage;
- temporary buffers for fluxes, sources, visualization, and QoI;
- MPI halo/trace communication buffers.

On GPUs, memory capacity can be the limiting constraint before raw FLOP/s.

## Practical Scaling Workflow

1. Validate the model on one rank/device.
2. Run a small MPI case and compare residuals and outputs.
3. Increase polynomial order and mesh size separately to identify the main cost.
4. Measure GMRES iterations before changing hardware scale.
5. Tune preconditioner and `GMRESrestart`.
6. Scale ranks/devices only after the single-rank algorithm is numerically
   stable.
7. Reduce output frequency when filesystem time becomes visible.

## Performance Metrics To Track

- time per Newton iteration;
- time per GMRES solve;
- number of GMRES iterations;
- preconditioner setup and application time;
- matrix-vector product time;
- residual assembly time;
- communication time if available;
- output time and file count.
