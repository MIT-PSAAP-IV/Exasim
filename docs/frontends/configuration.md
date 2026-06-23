# Frontend Configuration

Frontend configuration is centered on `pde` and `mesh`. This page groups the
most important options by purpose. For exhaustive text-input field definitions,
see [pdeapp.txt fields](../reference/pdeapp.md).

## Physics Models

Use `pde.model` and `pde.modelfile` to select the model-generation path.

```matlab
pde.model = "ModelD";
pde.modelfile = "pdemodel";
pde.physicsparam = [1.4, 0.72, 1000.0];
```

The model file defines callbacks such as:

- `initu`, `initq`, `initw`, `initodg`
- `flux`, `source`, `tdfunc`
- `ubou`, `fbou`, `fbouhdg`, `fhat`, `uhat`, `stab`
- `visscalars`, `visvectors`, `vistensors`
- `qoivolume`, `qoiboundary`

The symbolic arguments include state variables, coordinates, time,
`physicsparam` as `param`, and freestream values as `uinf`. The callback
contract is documented in [pdemodel abstraction](pdemodel.md) and
[Model contract](../reference/model-contract.md).

## Physical Parameters

`pde.physicsparam` is the concrete parameter vector used by one solve. It is
serialized into backend input data and passed to model callbacks at runtime.

```python
pde["physicsparam"] = [1.4, 0.72, 2000.0]  # gamma, Prandtl, Reynolds, for example
```

Use `pde.physicsparamsweep` when you want several parameter vectors. Each row is
one case. The frontend sweep and exported standalone app use the same ordering.

```matlab
pde.physicsparam = [1.4, 0.72, 500.0];
pde.physicsparamsweep = [
    1.4 0.72  500.0
    1.4 0.72 1000.0
    1.4 0.72 1500.0
];
pde.physicsparamwarmstart = 1;  % use previous converged case as next initial state
```

See [Parameter sweeps](../usage-modes/parameter-sweeps.md) for output layout,
manifest files, standalone behavior, and warm-start details.

## Dimensions And Discretization

The dimensions must match the PDE model:

| Field | Meaning |
| --- | --- |
| `nd` | Spatial dimension. |
| `ncu` | Number of conserved/primary unknowns. |
| `ncq` | Number of gradient components. |
| `nc` | Total `udg` components, commonly `ncu + ncq`. |
| `nco` | Number of `odg` auxiliary components. |
| `ncw` | Number of `wdg` auxiliary components. |
| `nch` | Number of trace components for HDG. |
| `porder` | Polynomial order. |
| `pgauss` | Quadrature order. |
| `hybrid` | Discretization path; examples use this to select LDG/HDG behavior. |

Example:

```julia
pde.nd = 2
pde.ncu = 4
pde.ncq = pde.ncu * pde.nd
pde.nc = pde.ncu + pde.ncq
pde.porder = 3
pde.pgauss = 2 * pde.porder
```

## Boundary Conditions

Boundary setup is split between `mesh` and model callbacks:

1. `mesh.boundaryexpr` identifies geometric boundary groups.
2. `mesh.boundarycondition` maps those groups to integer boundary IDs.
3. `ubou`, `fbou`, or `fbouhdg` implements the boundary behavior for each ID.

Example:

```matlab
mesh.boundarycondition = [1; 2; 2; 1];
% Boundary callback uses ib/iboundary to distinguish wall, inlet, outlet, etc.
```

Keep boundary IDs deterministic. A mismatch between mesh tags and model
boundary cases is a common source of incorrect solutions.

## Initial Conditions

Initial data can come from:

- frontend model callbacks such as `initu`, `initq`, `initw`, and `initodg`;
- restart or saved solution files;
- parameter-sweep warm-start state from the previous case.

For parameter sweeps, the standard initialization path is used for the first
case. If `physicsparamwarmstart` is enabled, later cases use the previous
converged solution as the initial state.

## Time Integration

Use `pde.tdep = 0` for steady problems and `pde.tdep = 1` for time-dependent
problems.

```python
pde["tdep"] = 1
pde["time"] = 0.0
pde["dt"] = [1.0e-3] * 100
pde["saveSolFreq"] = 10
```

Time-dependent output frequency is controlled by `saveSolFreq`. Restarted or
postprocessed runs may use `timestepOffset` to keep output numbering consistent.

## Solver Controls

Important nonlinear and linear solver fields:

| Field | Purpose |
| --- | --- |
| `NLiter` | Maximum nonlinear iterations. |
| `NLtol` | Nonlinear convergence tolerance. |
| `linearsolveriter` | Maximum linear iterations. |
| `linearsolvertol` | Linear convergence tolerance. |
| `GMRESrestart` | GMRES restart length. |
| `preconditioner` | Preconditioner selection. |
| `saveResNorm` | Write residual history files. |

Example:

```matlab
pde.NLiter = 30;
pde.NLtol = 1e-8;
pde.linearsolveriter = 200;
pde.linearsolvertol = 1e-6;
pde.preconditioner = 1;
pde.saveResNorm = 1;
```

## Postprocessing Flags

`pde.saveParaview = 1` enables backend VTK output during solve mode.
Visualization fields are controlled by `nsca`, `nvec`, and `nten` and by model
callbacks `visscalars`, `visvectors`, and `vistensors`.

`pde.executionmode = 1` makes frontend `runcode(...)` launch the generated
executable in postprocess mode against existing data. The default
`pde.executionmode = 0` preserves normal solve behavior.

```julia
pde.saveParaview = 1
pde.nsca = 1
pde.executionmode = 0
```

See [Postprocessing and visualization](postprocessing.md).

## See Also

- [Execution workflow](execution.md)
- [Preprocessing](preprocessing.md)
- [pdemodel abstraction](pdemodel.md)
- [Parameter sweeps](../usage-modes/parameter-sweeps.md)
- [Postprocessing](../usage-modes/postprocessing.md)
