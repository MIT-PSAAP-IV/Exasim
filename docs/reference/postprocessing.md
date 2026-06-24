# Postprocessing Reference

Postprocessing reads solution data and writes derived outputs such as VTK
files, QoI files, visualization fields, and CG output fields.

## Configuration Fields

| Field/key | Scope | Default | Meaning |
| --- | --- | --- | --- |
| `saveParaview` | Runtime | `0` | Enable VTK/ParaView output when visualization fields are defined. |
| `nsca` | Runtime | `0` | Number of scalar visualization fields. |
| `nvec` | Runtime | `0` | Number of vector visualization fields. |
| `nten` | Runtime | `0` | Number of tensor visualization fields. |
| `nsurf` | Runtime | `0` | Number of surface output components. |
| `nvqoi` | Runtime | `0` | Number of volume QoI components. |
| `saveSolFreq` | Runtime | `1` | Saved solution frequency. |
| `saveSolOpt` | Runtime | `1` | Saved solution storage option. |
| `executionmode` | Frontend-only | `0` | `1` launches standalone postprocess mode from frontends. |

## Model Functions Used by Postprocessing

| Function | Output |
| --- | --- |
| `VisScalars` | Scalar fields for VTK output. |
| `VisVectors` | Vector fields for VTK output. |
| `VisTensors` | Tensor fields for VTK output. |
| `QoIvolume` | Volume QoI integrands. |
| `QoIboundary` | Boundary QoI integrands. |
| `Output` | DG output field. |

## Solve-Time Output

Solve mode writes saved solution files and optionally postprocessing outputs
depending on the runtime flags. `saveParaview = 1` is required for VTK output.

## Standalone Postprocess Mode

Generated apps and app-mode executables can run in postprocess mode. In this
mode the executable reads saved solution records and writes derived output
without rerunning the nonlinear solver.

Frontend `executionmode = 1` is a helper that makes `runcode` launch the
generated executable with postprocess semantics.

## Saved Solution Replay

Postprocessing can read:

- Final solution files through `ReadSolutions`.
- Specific saved records through `GetSolutions(record_index)`.

For compact saved solution formats, Exasim reconstructs missing quantities such
as `q` when required for visualization or QoI.

## Output Files

Common outputs include:

| Output | Meaning |
| --- | --- |
| `outudg_np*.bin` | Rank-local solution data. |
| `outwdg_np*.bin` | Rank-local auxiliary `w` data, when present. |
| `outuhat_np*.bin` | HDG trace data, when present. |
| `out_residualnorms*.bin` | Residual history when enabled. |
| `outqoi.txt` | Quantities of interest. |
| `*.vtu`, `*.pvtu`, `*.pvd` | ParaView visualization files. |

## MPI Behavior

MPI runs write rank-local binary outputs and parallel VTK metadata where
enabled. QoI and visualization routines must avoid double-counting ghost
elements.

## Related Pages

- [Application-mode postprocessing](../usage-modes/postprocessing.md)
- [Model functions](model-functions.md)
- [Frontend postprocessing](../frontends/postprocessing.md)
