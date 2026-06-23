# Postprocessing

Exasim postprocessing converts saved DG/HDG solution data into user-facing
outputs: visualization files, integrated quantities of interest, continuous-grid
derived fields, residual histories, and optional boundary extracts. It can run
as part of a normal solve or as a separate pass over existing `datain/` and
`dataout/` files.

Postprocessing callbacks operate on the same model variables described in
[Physics Models](../physics-models/index.md): `u`, optional `q`, auxiliary
`w`, external `v`, EOS-derived fields, and AV fields when present.

This page documents the current implementation used by generated frontend apps,
`text2code` apps, and C++ applications that drive `ExasimSolver`.

## Workflow overview

```text
pdeapp / frontend / pdeapp.txt
        |
        v
preprocessing writes datain/
        |
        v
solve mode: exasimapp [solve] <nmodels> <datain/> <dataout/out>
        |
        +--> saved solution binaries: outudg_np*.bin, outuhat_np*.bin, ...
        +--> solve-time postprocessing: outvis*.vtu/.pvtu/.pvd, outqoi.txt, ...
        |
        v
optional replay:
exasimapp postprocess <nmodels> <datain/> <dataout/out> [restart] [postmode]
```

There are two common workflows:

| Workflow | When to use it | What it does |
|---|---|---|
| Solve-time postprocessing | Normal runs where output should be written during the solve. | `RunSolveProblemOrPostprocess()` calls `SaveQoI`, `SaveParaview`, `SaveOutputCG`, and solution/boundary writers at configured save points. |
| Explicit postprocess replay | Existing `dataout/` already contains saved solution records and you want to regenerate derived outputs. | The executable runs in `Postprocess` mode, reads saved `outudg`/`outwdg`/`outuhat` data, recomputes QoI/visualization/output fields, and writes derived outputs. |

!!! note "Postprocess mode treats saved solution files as inputs"
    Explicit `postprocess` mode must not truncate or rewrite the saved solution
    files before reading them. It reads `outudg_np*.bin`, `outwdg_np*.bin`, and
    `outuhat_np*.bin` when present, then writes derived outputs such as
    `outvis*` and `outqoi.txt`.

## Frontend `pde.executionmode`

The MATLAB, Python, and Julia frontends provide a runtime-only field named
`executionmode`. It controls the command used by `runcode(...)` after
preprocessing and compilation.

| Value | Frontend command behavior | Backend behavior |
|---:|---|---|
| `0` | Run the executable with the historical solve CLI: `exasimapp <nmodels> <datain/> <dataout/out>`. | `ExasimSolver::ParseInputs()` sets `ExasimExecutionMode::Solve`; the solver advances the PDE and writes configured solution/postprocessing outputs. |
| `1` | Insert the `postprocess` subcommand: `exasimapp postprocess <nmodels> <datain/> <dataout/out>`. | `ExasimSolver::ParsePostprocessInputs()` sets `ExasimExecutionMode::Postprocess`; the backend reads existing saved solutions and writes derived outputs. |

Defaults:

=== "MATLAB"

    ```matlab
    pde.executionmode = 0;  % default solve mode
    ```

=== "Python"

    ```python
    pde['executionmode'] = 0  # default solve mode
    ```

=== "Julia"

    ```julia
    pde.executionmode = 0  # default solve mode
    ```

Use frontend postprocess mode when the compatible `datain/` and saved
`dataout/out*` files already exist:

=== "MATLAB"

    ```matlab
    [pde, mesh] = initializeexasim();
    % ... configure the same model, mesh, and output callbacks ...
    pde.saveParaview = 1;
    pde.executionmode = 1;
    [sol,pde,mesh,master,dmd] = exasim(pde,mesh);
    ```

=== "Python"

    ```python
    pde, mesh = exasim.initializeexasim()
    # ... configure the same model, mesh, and output callbacks ...
    pde['saveParaview'] = 1
    pde['executionmode'] = 1
    sol, pde, mesh = exasim.exasim(pde, mesh)[0:3]
    ```

=== "Julia"

    ```julia
    pde, mesh = Exasim.initializeexasim()
    # ... configure the same model, mesh, and output callbacks ...
    pde.saveParaview = 1
    pde.executionmode = 1
    sol, pde, mesh, master, dmd, compilerstr, runstr = Exasim.exasim(pde, mesh)
    ```

Important constraints:

- `executionmode` is not serialized into `app.bin`; it only changes how the
  frontend launches the executable.
- `executionmode` is not a `pdeapp.txt` key. Text/C++ workflows should use the
  explicit `postprocess` command-line form shown below.
- Frontend `executionmode = 1` currently uses the default postprocess replay
  arguments (`postmode = 0`). Use the explicit command line when you need
  `restart`, `postmode`, or `nsca`/`nvec`/`nten`/`nsurf`/`nvqoi` overrides.
- Postprocess mode requires saved solution files from a previous compatible
  solve. It does not generate a new physical solution.
- Postprocess mode uses the same executable backend as the solve. A CPU build
  postprocesses on CPU; a CUDA/HIP build evaluates derived fields on that GPU
  backend and writes files from host-side output buffers.
- For MPI, the frontend still launches with `mpirun -np pde.mpiprocs`; use the
  same rank count and partitioning used to create the saved files.

Internally, postprocess mode constructs lighter solver objects: solve-only
preconditioner, GMRES, and HDG matrix storage are skipped where possible, while
geometry, solution state, `q` reconstruction, QoI, visualization, and output
field data needed for postprocessing remain available.

## Output types

| Output | Files | Controlled by | Description |
|---|---|---|---|
| Saved solution | `outudg_np<rank>.bin` | `saveSolFreq`, `saveSolOpt` | DG state records. With `saveSolOpt=0`, only the conservative/state part is saved; with `saveSolOpt=1`, full `udg` is saved. |
| Saved `w` field | `outwdg_np<rank>.bin` | `ncw > 0` | Auxiliary `wdg` state records. |
| Saved HDG trace | `outuhat_np<rank>.bin` | HDG / `spatialScheme == 1` | HDG trace unknown records. |
| Restart snapshots | `outudg_t<step>_np<rank>.bin`, `outwdg_t<step>_np<rank>.bin`, `_uhat_t<step>_np<rank>.bin` | internal restart-save interval | Time-dependent restart-style snapshots. |
| ParaView visualization | `outvis.vtu`, `outvis.pvtu`, `outvis_000001.vtu`, `outvis.pvd` | `saveParaview`, `nsca`, `nvec`, `nten`, `saveSolFreq` | VTK unstructured-grid files with scalar/vector/tensor point fields. |
| Quantities of interest | `outqoi.txt` | `nvqoi`, `nsurf` | Rank-0 text file containing time followed by integrated volume and surface QoIs. |
| CG output field | `_outputCG_np<rank>.bin`, `_outputCG_t<step>_np<rank>.bin` | `Output` model hook / output component count | Model-defined output field converted from DG to CG-style storage. |
| Boundary extracts | `outbouxdg_np<rank>.bin`, `outboundg_np<rank>.bin`, `outbouudg_np<rank>.bin`, `outbouuhat_np<rank>.bin`, `outbouwdg_np<rank>.bin` | `saveSolBouFreq`, `ibs` | Boundary geometry, normals, solution, trace, and optional `wdg` data for selected boundary blocks. |
| Residual history | `out_residualnorms<rank>.bin` | `saveResNorm` | Binary nonlinear/linear residual history used by frontend helpers. |

All per-rank binary files are written under the path prefix passed as
`fileout`, usually `dataout/out`.

## Visualization fields

Visualization fields come from model callbacks:

| Callback | Output count | VTK field type |
|---|---:|---|
| `visscalars(u, q, w, v, x, t, mu, eta)` / `VisScalars` | `nsca` | one-component scalar point fields |
| `visvectors(u, q, w, v, x, t, mu, eta)` / `VisVectors` | `nvec` | 3-component vector point fields; 2D data are padded with zero z-component |
| `vistensors(u, q, w, v, x, t, mu, eta)` / `VisTensors` | `nten` | tensor point fields with `nd*nd` components |

The frontends infer `nsca`, `nvec`, and `nten` from the callback return sizes.
For `pdeapp.txt`/text2code workflows, the corresponding `VisScalars`,
`VisVectors`, and `VisTensors` functions must be listed in the model outputs,
and the app must provide consistent `nsca`, `nvec`, and `nten` values when they
cannot be inferred from a built-in model.

Visualization is enabled only when:

```text
saveParaview != 0
and
nsca + nvec + nten > 0
```

The backend evaluates these fields on DG nodes, maps them to the visualization
CG nodes with `VisDG2CG`, and writes VTK appended raw-binary files using
`Float32` point data.

## ParaView files

For steady or single-output serial runs:

```text
dataout/outvis.vtu
```

For steady MPI runs:

```text
dataout/outvis.pvtu
dataout/outvis_00000.vtu
dataout/outvis_00001.vtu
...
```

For time-dependent runs, rank 0 also writes a collection file:

```text
dataout/outvis.pvd
dataout/outvis_000001.vtu        # serial
dataout/outvis_000001.pvtu       # MPI, plus rank pieces
```

Open the collection file in ParaView for transient data:

```bash
paraview --data=dataout/outvis.pvd
```

Open the `.pvtu` file for MPI steady data:

```bash
paraview --data=dataout/outvis.pvtu
```

!!! note "Relative paths in `.pvd`"
    The `.pvd` writer stores relative file names, so moving the whole
    `dataout/` directory preserves the collection-file references.

## Quantities of interest

Volume and surface QoIs are computed by model callbacks:

| Callback | Count | Output |
|---|---:|---|
| `qoivolume(...)` / `QoIvolume` | `nvqoi` | Integrated over elements. |
| `qoiboundary(...)` / `QoIboundary` | `nsurf` | Integrated over boundary/surface faces. |

`outqoi.txt` is written by rank 0. Each row contains:

```text
time  qoi_volume_1 ... qoi_volume_n  qoi_surface_1 ... qoi_surface_m
```

For steady problems, the first column is `0.0`. For time-dependent problems, it
is the current simulation time.

## Frontend visualization in MATLAB, Python, and Julia

The language frontends provide a separate `vis(...)` helper that converts the
returned `sol` array to `.vtu` or `.pvd` files and optionally launches ParaView.
This is independent of backend `saveParaview` output.

### MATLAB

```matlab
[pde, mesh] = initializeexasim();
pde.saveParaview = 1;             % backend VTK during solve/postprocess
% ... define pde, mesh, and run ...
[sol, pde, mesh] = exasim(pde, mesh);

pde.visscalars = ["temperature", 1];
pde.visvectors = ["gradient", [2, 3]];
pde.paraview = "paraview";        % or full path to ParaView executable
vis(sol, pde, mesh);
```

### Python

```python
pde, mesh = exasim.initializeexasim()
pde['saveParaview'] = 1
# ... define pde, mesh, and run ...
sol, pde, mesh = exasim.exasim(pde, mesh)[0:3]

pde['visscalars'] = ["temperature", 0]
pde['visvectors'] = ["gradient", numpy.array([1, 2])]
pde['paraview'] = "paraview"
exasim.vis(sol, pde, mesh)
```

### Julia

```julia
pde, mesh = Exasim.initializeexasim()
pde.saveParaview = 1
# ... define pde, mesh, and run ...
sol, pde, mesh, master, dmd, compilerstr, runstr = Exasim.exasim(pde, mesh)

pde.visscalars = ["temperature", 1]
pde.visvectors = ["gradient", [2, 3]]
pde.paraview = "paraview"
Exasim.vis(sol, pde, mesh)
```

Frontend `vis(...)` settings:

| Setting | Meaning |
|---|---|
| `visscalars` | Names and component indices for scalar fields in the frontend `sol` array. |
| `visvectors` | Names and component indices for vector fields. |
| `viselem` | Optional element subset for visualization. Empty means all elements. |
| `visfilename` | Output file prefix, defaulting to `dataout/output`. |
| `visdt` | Time spacing used by frontend `.pvd` output. |
| `paraview` | Executable name or path. Empty string disables auto-launch. |

## Explicit standalone postprocess mode

Generated frontend apps and `text2code`/C++ apps accept a `postprocess`
subcommand:

```bash
build/exasimapp postprocess <nummodels> <datain0/> <dataout0/out> \
    [<datain1/> <dataout1/out> ...] [restart] [postmode] \
    [nsca] [nvec] [nten] [nsurf] [nvqoi]
```

For the common one-model case:

```bash
build/exasimapp postprocess 1 datain/ dataout/out 0 0 2 1 0 0 0
```

Argument meanings:

| Argument | Meaning |
|---|---|
| `nummodels` | Number of model/input-output pairs. Coupled two-domain runs may encode the first-domain MPI count as `100 + mpiprocs0`. |
| `datain/` | Directory containing `app.bin`, `master.bin`, `mesh.bin`, and `sol.bin`. |
| `dataout/out` | Output prefix used by the original solve. |
| `restart` | Saved record index for `postmode >= 1`. |
| `postmode = 0` | Read the current/default saved solution with `ReadSolutions()`. |
| `postmode = 1` | Read saved record `restart` with `GetSolutions(restart)`. |
| `postmode > 1` | Loop over records `restart, restart+1, ..., restart+postmode-1`. |
| `nsca`, `nvec`, `nten`, `nsurf`, `nvqoi` | Optional overrides for postprocess-only output counts. Use them when the app input does not already carry the desired counts. |

If `saveSolOpt == 0` and the saved records contain only the state variables, the
postprocess reader reconstructs `udg` and recomputes `q` when gradient
components are needed.

## Steady-state workflow

1. Enable the desired model callbacks and output flags.
2. Run the solve.
3. Inspect `dataout/outqoi.txt` and `dataout/outvis.vtu` or `outvis.pvtu`.
4. Optionally rerun explicit postprocess mode if you changed output counts or
   need to regenerate derived files from saved solution data.

Example:

```python
pde['saveParaview'] = 1
pde['saveResNorm'] = 1
```

```bash
python3 pdeapp.py
paraview --data=dataout/outvis.vtu
```

For MPI:

```bash
mpirun -np 4 build/exasimapp 1 datain/ dataout/out
paraview --data=dataout/outvis.pvtu
```

## Time-dependent workflow

Time-dependent output is controlled primarily by:

| Setting | Meaning |
|---|---|
| `dt` | Time-step sequence. |
| `saveSolFreq` | Frequency for saved solution records, VTK output, CG output, and solve-time postprocessing. |
| `timestepOffset` | Offset added to output step numbers. |
| `visdt` | Frontend visualization collection spacing. |

For backend VTK output, Exasim writes a `.pvd` collection on the first time step
and one `.vtu`/`.pvtu` dataset at each `saveSolFreq` step.

Example:

```python
pde['dt'] = 0.01*numpy.ones(100)
pde['soltime'] = numpy.arange(1, 101)
pde['saveSolFreq'] = 10
pde['saveParaview'] = 1
```

```bash
python3 pdeapp.py
paraview --data=dataout/outvis.pvd
```

To replay records 20 through 29:

```bash
build/exasimapp postprocess 1 datain/ dataout/out 20 10
```

## Parallel, CPU, and GPU behavior

Postprocessing uses the same backend selected for the executable:

| Backend | Behavior |
|---|---|
| CPU serial | Writes one `.vtu` and one set of `*_np0.bin` files. |
| CPU MPI | Each rank reads/writes its own `*_np<rank>.bin` files. Rank 0 writes `.pvtu` and `.pvd` metadata. |
| CUDA / HIP | Kernels evaluate derived fields on the device; visualization buffers are copied or mapped to host memory before VTK writing. File output is still host-side. |
| MPI + GPU | Same file layout as MPI CPU; each rank owns its local partition and backend device data. |

The VTK writer pads coordinates and vectors to three components so 2D data can
be loaded naturally by ParaView.

## Surface and boundary data

Use `saveSolBouFreq > 0` and select a boundary marker with `ibs` to write
boundary extracts. Exasim writes boundary geometry (`bouxdg`), normals
(`boundg`), state (`bouudg`), trace (`bouuhat`), and optional `wdg`
(`bouwdg`) records.

Boundary QoIs are different from boundary extracts: `QoIboundary`/`nsurf`
integrate model-defined quantities and write them into `outqoi.txt`.

## Known limitations and checks

- Backend VTK field names are currently generic (`Scalar Field 0`,
  `Vector Field 0`, ...). Frontend `vis(...)` can write user-provided names.
- Explicit postprocess mode requires existing compatible `datain/` and
  `dataout/out*` files from the same mesh, discretization, MPI partitioning, and
  model.
- MPI postprocess should be run with the same number of ranks and partitioning
  used to generate the saved files.
- ParaView reads Exasim backend VTK files as raw-appended binary
  unstructured-grid data. Generic XML readers that do not understand VTK
  appended data are not sufficient validators.
- If you change `mpiprocs`, backend, or model callbacks, clean/reconfigure the
  generated build directory before rerunning a standalone app.

## Related references

- [`pdeapp.txt` fields](../reference/pdeapp.md) for `saveParaview`,
  `saveSolFreq`, `saveSolOpt`, `saveSolBouFreq`, `saveResNorm`, and QoI/field
  counts.
- [Parameter sweeps](parameter-sweeps.md) for comparing per-case outputs under
  `dataout/paramcase_####/`.
- [`pdemodel.txt` syntax](../reference/pdemodel.md) for `VisScalars`,
  `VisVectors`, `VisTensors`, `QoIvolume`, `QoIboundary`, and `Output`.
- [Frontends](../frontends/index.md) for MATLAB/Python/Julia authoring and
  frontend `vis(...)`.
- [Driving the solver](../driving-the-solver.md) for direct C++ API usage.
