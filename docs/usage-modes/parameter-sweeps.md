# Parameter sweeps

Exasim can run the same PDE model over a set of `physicsparam` vectors. This is
useful for continuation studies, Reynolds-number or Mach-number sweeps,
parameter sensitivity checks, and design-space exploration where the mesh,
discretization, state dimensions, and generated kernels remain fixed.

The built-in sweep feature currently supports **`physicsparam` only**. To sweep
boundary conditions, mesh parameters, solver tolerances, polynomial order, or
model structure, script multiple Exasim runs in MATLAB/Python/Julia or generate
multiple `pdeapp.txt` inputs.

## What changes between cases

Each parameter case is one concrete `physicsparam` vector:

```text
case 1 -> physicsparam = [mu_1, mu_2, ..., mu_n]
case 2 -> physicsparam = [mu_1, mu_2, ..., mu_n]
...
```

Everything else is assumed compatible across cases:

| Quantity | Must remain fixed? | Reason |
|---|---:|---|
| Mesh and partitioning | yes | Saved files and warm-start data are partitioned by element/rank. |
| Discretization, element type, polynomial order | yes | Generated kernels and binary dimensions are fixed. |
| Number of state components and auxiliary fields | yes | `udg`, `wdg`, `odg`, and trace dimensions must match. |
| Model callbacks / generated kernels | yes | The executable is built once and reused. |
| `physicsparam` values | no | These are the swept values. |

## Supported workflows

```text
Frontend sweep
  pde.physicsparamsweep -> frontend loop -> write app.bin per case -> run exasimapp

Exported standalone sweep
  pde.physicsparamsweep + exportapp -> datain/physicsparamcases.bin
  -> run.sh -> exasimapp detects file -> internal C++ sweep

pdeapp.txt / text2code sweep
  physicsparamcases in pdeapp.txt -> datain/physicsparamcases.bin
  -> exasimapp detects file -> internal C++ sweep
```

| Workflow | Requires MATLAB/Python/Julia at run time? | Case loop location | Typical use |
|---|---:|---|---|
| Frontend-driven sweep | yes | Frontend `exasim(...)` loop | Interactive studies, immediate access to returned `sol` arrays. |
| `exportapp` standalone sweep | no | Generated C++ executable | Move a frontend-defined sweep to HPC or a machine without the frontend. |
| `pdeapp.txt` / text2code sweep | no | Generated C++ executable | Pure text/C++ workflow and batch execution. |

All workflows write the same case directory convention:

```text
dataout/
  physicsparam_sweep_manifest.txt
  paramcase_0001/
    outqoi.txt
    outudg_np0.bin
    outvis.vtu
    physicsparam.txt
    physicsparam_metadata.txt        # standalone C++ sweeps
  paramcase_0002/
    ...
```

`physicsparam.txt` stores the exact vector used by that case. The manifest
records the number of cases, parameter count, warm-start flag, output directory,
and values for each case.

## Frontend API

All frontends initialize:

```text
physicsparamsweep = empty
physicsparamwarmstart = 0
```

Set `physicsparamsweep` to enable a multi-case run. Existing single-case
examples do not need any changes.

### Explicit samples

Use one row per case and one column per entry of `physicsparam`.

=== "MATLAB"

    ```matlab
    pde.physicsparam = [1.4 1000 0.72 0.2];
    pde.physicsparamsweep = [
        1.4  500 0.72 0.2
        1.4 1000 0.72 0.2
        1.4 1500 0.72 0.2
    ];
    [sol,pde,mesh,master,dmd] = exasim(pde,mesh);
    ```

=== "Python"

    ```python
    pde['physicsparam'] = numpy.array([1.4, 1000, 0.72, 0.2])
    pde['physicsparamsweep'] = numpy.array([
        [1.4,  500, 0.72, 0.2],
        [1.4, 1000, 0.72, 0.2],
        [1.4, 1500, 0.72, 0.2],
    ])
    sol, pde, mesh = exasim.exasim(pde, mesh)[0:3]
    ```

=== "Julia"

    ```julia
    pde.physicsparam = [1.4 1000 0.72 0.2]
    pde.physicsparamsweep = [
        1.4  500 0.72 0.2
        1.4 1000 0.72 0.2
        1.4 1500 0.72 0.2
    ]
    sol, pde, mesh, master, dmd, compilerstr, runstr = Exasim.exasim(pde, mesh)
    ```

### Scalar sweeps

For a scalar `physicsparam`, a column vector or simple list is treated as one
case per value:

=== "MATLAB"

    ```matlab
    pde.physicsparam = 1.0;
    pde.physicsparamsweep = [0.5; 1.0; 2.0];
    ```

=== "Python"

    ```python
    pde['physicsparam'] = numpy.array([1.0])
    pde['physicsparamsweep'] = [0.5, 1.0, 2.0]
    ```

=== "Julia"

    ```julia
    pde.physicsparam = [1.0]
    pde.physicsparamsweep = [0.5, 1.0, 2.0]
    ```

### Cartesian grids

Use `grid` to form the Cartesian product of one value list per parameter.

=== "MATLAB"

    ```matlab
    pde.physicsparam = [1.0 0.0];
    pde.physicsparamsweep.grid = {[0.5 1.0 2.0], [0.0 1.0]};
    ```

=== "Python"

    ```python
    pde['physicsparam'] = numpy.array([1.0, 0.0])
    pde['physicsparamsweep'] = {'grid': [[0.5, 1.0, 2.0], [0.0, 1.0]]}
    ```

=== "Julia"

    ```julia
    pde.physicsparam = [1.0, 0.0]
    pde.physicsparamsweep = Dict(:grid => [[0.5, 1.0, 2.0], [0.0, 1.0]])
    ```

This produces six cases in deterministic product order.

### Samples and values aliases

The frontends also accept structured sample lists:

=== "MATLAB"

    ```matlab
    pde.physicsparamsweep.samples = {[0.5 0.0], [1.0 1.0], [2.0 0.0]};
    % values is equivalent:
    pde.physicsparamsweep.values = {[0.5 0.0], [1.0 1.0], [2.0 0.0]};
    ```

=== "Python"

    ```python
    pde['physicsparamsweep'] = {
        'samples': [[0.5, 0.0], [1.0, 1.0], [2.0, 0.0]],
    }
    ```

=== "Julia"

    ```julia
    pde.physicsparamsweep = Dict(:samples => [[0.5, 0.0], [1.0, 1.0], [2.0, 0.0]])
    ```

## `pdeapp.txt` / text2code syntax

For standalone text workflows, define `physicsparamcases` in `pdeapp.txt`.
Each row must have the same length as `physicsparam`.

```text
physicsparam = [1.4, 1000, 0.72, 0.2, 1, 1, 0, 45.1429];
physicsparamcases = [
  [1.4,  500, 0.72, 0.2, 1, 1, 0, 45.1429],
  [1.4, 1000, 0.72, 0.2, 1, 1, 0, 45.1429],
  [1.4, 1500, 0.72, 0.2, 1, 1, 0, 45.1429],
  [1.4, 2000, 0.72, 0.2, 1, 1, 0, 45.1429]
];
physicsparamwarmstart = 1;
```

Semicolon-separated rows are also accepted:

```text
physicsparamcases = [0.5, 0.0; 1.0, 1.0; 2.0, 0.0];
```

During preprocessing, Exasim writes:

```text
datain/physicsparamcases.bin
```

The generated executable detects this file and runs the sweep internally in
solve mode.

## Shared sweep file format

`physicsparamcases.bin` is a binary `Float64` file:

```text
double ncases
double nparam
double values[ncases*nparam]
```

The payload is case-major:

```text
values[icase*nparam + iparam]
```

The standalone C++ runner validates:

- `ncases > 0`
- `nparam > 0`
- `nparam` matches the `physicsparam` length stored in `app.bin`
- all values are finite
- the run is a single-model sweep

## Execution behavior

### Frontend-driven sweeps

The frontend workflow runs cases sequentially:

1. Normalize `physicsparamsweep` to an `ncases x nparam` matrix.
2. Preprocess once using the first case to build mesh, master, partitioning, and
   generated kernels.
3. For each case:
    - set `pde.physicsparam`;
    - set `pde.dataoutpath = dataout/paramcase_####`;
    - write `physicsparam.txt`;
    - rewrite `app.bin` with the current `physicsparam`;
    - run the generated executable;
    - fetch the solution and optional residual history.
4. Write `physicsparam_sweep_manifest.txt`.

`master` and `dmd` are reused because the mesh and partitioning do not change.

### Standalone C++ sweeps

For `exportapp` and `pdeapp.txt` workflows, the executable detects
`datain/physicsparamcases.bin` and runs all cases internally. It is still a
sequential case loop, but it does not require MATLAB/Python/Julia at runtime.

Cold-start mode (`physicsparamwarmstart = 0`) rebuilds the model for each case
after replacing `physicsparam`. This preserves parameter-dependent initial
condition behavior from callbacks such as `initu`, `initq`/`initudg`, `initv`,
and `initw`.

Warm-start mode (`physicsparamwarmstart = 1`) builds the first case, then reuses
the converged solution for later cases and updates `app.physicsparam` in the
existing model. This avoids rebuilding and is intended for continuation studies.

### Serial, MPI, and GPU runs

Sweep parallelism is currently **within each case**, not across cases. If the
base executable is serial CPU, each case runs serial CPU. If it is MPI, GPU, or
MPI+GPU, each case uses that same backend and rank layout.

Examples:

```bash
# serial standalone sweep
build/exasimapp 1 datain/ dataout/out

# MPI standalone sweep; each parameter case runs on 4 ranks
mpirun -np 4 build/exasimapp 1 datain/ dataout/out

# exported app; choose the installed backend variant when building the bundle
EXASIM_ROOT=/path/to/install EXASIM_VARIANT=gpumpi MPIRUN=mpirun ./run.sh
```

For MPI sweeps, each `paramcase_####` directory contains per-rank files such as
`outudg_np0.bin`, `outudg_np1.bin`, and MPI visualization metadata such as
`outvis.pvtu`.

## Warm-start continuation

Enable warm-starting with:

=== "MATLAB"

    ```matlab
    pde.physicsparamwarmstart = 1;
    ```

=== "Python"

    ```python
    pde['physicsparamwarmstart'] = 1
    ```

=== "Julia"

    ```julia
    pde.physicsparamwarmstart = 1
    ```

=== "pdeapp.txt"

    ```text
    physicsparamwarmstart = 1;
    ```

Behavior:

| Case | Initial state |
|---|---|
| First case | Standard initialization from `sol.bin` or model initial-condition callbacks. |
| Later cases, frontend sweeps | Frontends read the previous case's final `outudg`/`outwdg` using `getsolutions()` and rewrite `datain/sol*.bin`. |
| Later cases, standalone sweeps | The C++ executable keeps the model alive, updates `app.physicsparam`, resets time counters, redirects outputs, and continues from the previous converged state. |

Warm-starting is useful when neighboring cases are close enough that the
previous solution is a good initial guess. It can reduce nonlinear iterations in
continuation studies, but it is not guaranteed to help for large parameter
jumps or bifurcating solution branches.

For unsteady runs, warm-starting starts the next parameter case from the
previous case's final state while resetting the run time bookkeeping. It is not
a physical time-history continuation across parameter values.

## Exported standalone apps

When `pde.exportapp` / `pde['exportapp']` is set, the frontend writes the sweep
file into the exported bundle:

```text
my-sweep-app/
  CMakeLists.txt
  main.cpp
  run.sh
  datain/
    app.bin
    mesh.bin
    master.bin
    sol.bin
    physicsparamcases.bin
  dataout/
```

Run the sweep without the frontend:

```bash
cd my-sweep-app
EXASIM_ROOT=/path/to/exasim/install ./run.sh
```

For MPI bundles:

```bash
EXASIM_ROOT=/path/to/exasim/install MPIRUN=srun ./run.sh
```

The bundle builds once and the generated executable runs all parameter cases.

## Restarting interrupted sweeps

The current sweep runner does not have a built-in "skip completed cases" flag.
Recommended restart workflows:

| Workflow | Restart approach |
|---|---|
| Frontend sweep | Restrict `physicsparamsweep` to the remaining rows and rerun. Existing completed `paramcase_####` directories are not automatically reused. |
| Exported standalone sweep | Edit/regenerate `physicsparamcases.bin` to contain only remaining cases, or regenerate the export bundle with the remaining cases. |
| `pdeapp.txt` sweep | Edit `physicsparamcases` to contain remaining cases, rerun preprocessing/text2code, then rerun the executable. |
| Warm-start continuation | If restarting after an interruption, include the last converged case as the first new case when you need the following case to warm-start from it. |

Keep the original `physicsparam_sweep_manifest.txt` with archived results if
you need a complete record of the original case numbering.

## Postprocessing and comparison

Each case is a normal Exasim output tree. You can use the same tools described
in [Postprocessing](postprocessing.md):

```bash
paraview --data=dataout/paramcase_0001/outvis.vtu
paraview --data=dataout/paramcase_0002/outvis.vtu
```

For MPI:

```bash
paraview --data=dataout/paramcase_0001/outvis.pvtu
```

Compare QoIs or residual histories across cases by reading:

```text
dataout/paramcase_0001/outqoi.txt
dataout/paramcase_0002/outqoi.txt
dataout/paramcase_0001/out_residualnorms0.bin
```

The manifest provides a stable mapping from case index to parameter values and
output directory.

Frontend `pde.executionmode` should normally remain at its default value `0`
when running a sweep, because a sweep generates new case solutions. Use
`executionmode = 1` only after compatible case output files already exist and
you want to replay postprocessing for those files. Standalone C++ sweeps
detected from `physicsparamcases.bin` are supported in solve mode only.

## Examples in the repository

| Example | Workflow | What it demonstrates |
|---|---|---|
| `examples/Poisson/poisson2d/pdeapp_sweep.m` | MATLAB frontend sweep | Minimal scalar `physicsparam` sweep. |
| `examples/NavierStokes/naca2d/pdeapp_sweep.m` | MATLAB frontend sweep | Reynolds-number sweep for NACA0012. |
| `examples/NavierStokes/naca2d/pdeapp_sweep_exportapp.m` | Exported standalone sweep | Frontend-defined sweep exported to a standalone app. |
| `examples/NavierStokes/eppler/pdeapp_sweep.m` | MATLAB frontend sweep | Multi-parameter aerodynamic sweep over Reynolds number and angle of attack. |
| `apps/navierstokes/naca0012steady/pdeapp_sweep.txt` | `pdeapp.txt` / text2code | Standalone text workflow with Reynolds-number sweep and warm-starting. |

## Design-space and sensitivity studies

A simple sensitivity study sweeps one parameter while holding others fixed:

```python
base = numpy.array([gamma, Re0, Pr, Minf, rho, ux, uy, rhoE])
pde['physicsparam'] = base
pde['physicsparamsweep'] = numpy.vstack([
    base,
    base + numpy.array([0, -100, 0, 0, 0, 0, 0, 0]),
    base + numpy.array([0,  100, 0, 0, 0, 0, 0, 0]),
])
```

A design-space exploration can use a Cartesian grid:

```python
pde['physicsparamsweep'] = {
    'grid': [
        [500, 1000, 1500],      # Reynolds number
        [0.0, 4.0, 8.0],        # angle of attack in degrees, if stored in physicsparam
    ]
}
```

If the angle of attack is represented by `cos(alpha)` and `sin(alpha)` in
`physicsparam`, build explicit sample rows instead of using a two-parameter
grid.

## Validation and error handling

Exasim validates malformed sweep inputs before launching cases:

- every case must have exactly `numel(physicsparam)` entries;
- all values must be finite;
- `physicsparamcases.bin` must have a valid header and payload;
- standalone binary sweep files must match the `physicsparam` length stored in
  `app.bin`;
- standalone C++ sweeps currently require a single model.

Single-case behavior is backward compatible: if no sweep is configured, Exasim
runs exactly one simulation as before.

## Current limitations

- Only `physicsparam` is swept by the built-in feature.
- Cases run sequentially. Parallelism is currently within each case through MPI
  and/or GPU backends, not across cases.
- Standalone C++ sweeps are limited to single-model runs.
- Automatic restart/skip of completed cases is not implemented.
- Warm-starting assumes the previous case's converged state is a meaningful
  initial condition for the next case.
- Changing mesh, discretization, model callbacks, state dimensions, or MPI
  partitioning requires a separate run, not a single sweep.

## Related documentation

- [`pdeapp.txt` fields](../reference/pdeapp.md) for `physicsparam`,
  `physicsparamcases`, and `physicsparamwarmstart`.
- [Frontends](../frontends/index.md) for MATLAB/Python/Julia setup and
  `exportapp`.
- [Postprocessing](postprocessing.md) for visualizing and comparing case
  outputs.
- [Application Modes](index.md) for built-in, external built-in, and shared-library
  execution paths.
