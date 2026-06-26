# Parallel Computing

Exasim supports distributed-memory execution through MPI. The frontend or
Text2Code workflow partitions the mesh, writes per-rank input data, and the C++
backend runs each MPI rank on its assigned subdomain.

## Domain Decomposition

The `dmd` data structure contains the domain decomposition information used by
the backend:

- element ownership by rank;
- local-to-global connectivity;
- interface and halo information;
- per-rank file offsets and output naming;
- communication maps for shared faces or trace data.

ParMETIS/Metis may be used by installed builds for graph partitioning.

## Communication Patterns

DG methods are local inside elements but communicate through faces.

| Solver path | Main communication |
| --- | --- |
| LDG | Neighbor face states and residual contributions for element-state iterations. |
| HDG | Trace unknowns and trace residuals on inter-rank faces. |

HDG often reduces the globally coupled unknowns but still requires trace
communication and preconditioner communication when subdomains cross rank
interfaces.

## MPI Outputs

Each rank writes rank-indexed binary output files. Visualization output may be
written as per-rank `.vtu` files plus a parallel `.pvtu` wrapper. Parameter
sweeps and postprocessing preserve the same per-rank naming convention inside
case-specific output directories.

## Scalability Considerations

Parallel efficiency depends on:

- element partition quality;
- surface-to-volume ratio of each rank's subdomain;
- polynomial order and element work per communication event;
- preconditioner communication;
- filesystem behavior for rank-indexed output;
- CPU/GPU binding and one-rank-per-device mapping.

## Practical Guidance

- Use enough elements per rank to keep local work larger than communication.
- Prefer partitionings that minimize inter-rank faces.
- For GPU runs, avoid assigning many MPI ranks to one GPU unless the machine
  launch model explicitly supports that workflow.
- Validate serial and MPI results on a small problem before scaling out.
- Check residual histories, not only the existence of output files.
