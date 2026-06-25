# pdeapp Reference

`pdeapp.txt` is the Text2Code application configuration file. It is a flat
list of assignments read by `text2code/text2code/readpdeapp.cpp`. The same
conceptual fields also appear in MATLAB/Python/Julia `pde` structures, but not
every frontend field is parsed from `pdeapp.txt`.

For model equations, see [pdemodel reference](pdemodel.md) and
[Physics Models](../physics-models/index.md).

## Syntax

```text
key = value;
key = [1, 2, 3];
key = ["expr1", "expr2"];
```

Supported forms:

- Strings use double quotes.
- Numeric lists use square brackets.
- `repeat(value, count)` is supported inside numeric lists.
- `physicsparamcases` supports bracketed rows or semicolon-separated rows.
- Statements end with `;`.

!!! warning "Integer versus floating-point parsing"
    The parser stores a scalar as floating-point only when the text contains
    `.` or `e`. Use `time = 0.0;`, not `time = 0;`, when setting a floating
    field.

## Required `pdeapp.txt` Keys

The Text2Code parser errors if any of these keys are missing:

| Key | Type | Purpose |
| --- | --- | --- |
| `model` | string | Model formulation: `ModelC`, `ModelD`, or `ModelW`. |
| `modelfile` | string | Path to `pdemodel.txt`. |
| `meshfile` | string | Path to binary mesh input. |
| `discretization` | string | `ldg` or `hdg`. |
| `platform` | string | `cpu` or `gpu`. |
| `mpiprocs` | int | Number of MPI ranks intended for generated data. |
| `porder` | int | Polynomial order. |
| `pgauss` | int | Quadrature order/degree used by preprocessing. |
| `physicsparam` | list(float) | Physics parameter vector `mu`. |
| `tau` | list(float) | Stabilization parameter vector. |
| `boundaryconditions` | list(int) | Boundary-condition code per boundary expression. |
| `boundaryexpressions` | list(string) | Geometric predicates defining boundary IDs. |

## Problem Definition

| Key | Type | Default | Required | Accepted values / notes |
| --- | --- | --- | --- | --- |
| `model` | string | `ModelD` | Yes | `ModelC`, `ModelD`, `ModelW`. |
| `modelfile` | string | `pdemodel.txt` | Yes | File path relative to `datapath` or working directory. |
| `meshfile` | string | `mesh.bin` | Yes | Binary mesh file read by preprocessing. |
| `discretization` | string | `ldg` | Yes | `ldg`, `LDG`, `hdg`, `HDG`; sets `hybrid`. |
| `platform` | string | `cpu` | Yes | `cpu` or `gpu`; actual backend also depends on linked executable. |
| `modelnumber` | int | `0` | No | Positive values suffix `datain`/`dataout` directories. |
| `builtinmodelID` | int | unset | No | Built-in provider dispatch ID where supported. |
| `runmode` | int | `0` | No | Runtime mode selector used by solver paths. |
| `debugmode` | int | `0` | No | `1` prints parsed parameters. |
| `linearproblem` | int | `0` | No | Marks linear problem paths where supported. |
| `subproblem` | int | `0` | No | Subproblem selector used by coupled workflows. |

## Paths and Generated Data

| Key | Type | Default | Required | Notes |
| --- | --- | --- | --- | --- |
| `exasimpath` | string | auto | No | Source/install root used by Text2Code for runtime data discovery. |
| `datapath` | string | path of `pdeapp.txt` or cwd | No | Base directory for `datain/` and `dataout/`. |
| `xdgfile` | string | empty | No | Optional coordinates/DG nodes input file. |
| `udgfile` | string | empty | No | Optional initial `udg` input file. |
| `vdgfile` | string | empty | No | Optional external/other-DG field input file. |
| `wdgfile` | string | empty | No | Optional auxiliary `w` field input file. |
| `uhatfile` | string | empty | No | Optional HDG trace input file. |
| `partitionfile` | string | empty | No | Optional partition input. |
| `gendatain` | int | `1` | No | Write backend binary input bundle. |
| `gencode` | int | `1` | No | Generate C++ model code. |
| `writemeshsol` | int | `1` | No | Write mesh and solution binary data. |

## Mesh and Dimensions

| Key | Type | Default | Required | Notes |
| --- | --- | --- | --- | --- |
| `ncu` | int | `1` | No | Number of primary solution components. |
| `ncv` | int | `0` | No | Number of external/other-DG components in text parser naming. |
| `ncw` | int | `0` | No | Number of auxiliary `w` components. |
| `nsca` | int | `0` | No | Number of scalar visualization fields. |
| `nvec` | int | `0` | No | Number of vector visualization fields. |
| `nten` | int | `0` | No | Number of tensor visualization fields. |
| `nsurf` | int | `0` | No | Surface output count. |
| `nvqoi` | int | `0` | No | Volume QoI count. |
| `neb` | int | `4096` | No | Element block size for batched kernels. |
| `nfb` | int | `8192` | No | Face block size for batched kernels. |
| `nodetype` | int | `1` | No | Node distribution selector. |

Some dimensions such as `nd`, `nc`, `ncq`, `ncx`, and element counts are
derived from mesh/model preprocessing rather than typically written by hand in
`pdeapp.txt`.

## Discretization and Time Integration

| Key | Type | Default | Required | Notes |
| --- | --- | --- | --- | --- |
| `porder` | int | `1` | Yes | Polynomial order. |
| `pgauss` | int | `2` | Yes | Quadrature order/degree. |
| `temporalscheme` | int | `0` | No | Temporal scheme selector. |
| `torder` | int | `1` | No | Time-integration order. |
| `nstage` | int | `1` | No | Number of stages. |
| `time` | float | `0.0` | No | Initial time; write with decimal. |
| `dt` | list(float) | empty | No | Time-step sequence; first positive value sets `tdep=1`. |
| `tdep` | int | `0` | No | Time-dependent flag; also inferred from `dt`. |
| `wave` | int | `0` | No | Wave-model flag used by ModelW-style paths. |

## Solver Configuration

| Key | Type | Default | Required | Notes |
| --- | --- | --- | --- | --- |
| `nonlinearsolver` | int | `0` | No | Nonlinear solver selector. |
| `NewtonIter` | int | `20` | No | Maximum Newton iterations. |
| `NewtonTol` | float | `1e-6` | No | Newton convergence tolerance. |
| `NLparam` | float | `0.0` | No | Nonlinear solver parameter. |
| `NLMatrixType` | int | `0` | No | Nonlinear matrix selector. |
| `linearsolver` | int | `0` | No | Linear solver selector. |
| `GMRESiter` | int | `200` | No | Maximum GMRES iterations. |
| `GMRESrestart` | int | `25` | No | GMRES restart length in Text2Code defaults. |
| `GMRESortho` | int | `0` | No | Orthogonalization selector. |
| `GMREStol` | float | `1e-3` | No | Linear solver tolerance. |
| `preconditioner` | int | `0` | No | Preconditioner selector in Text2Code defaults. |
| `precMatrixType` | int | `0` | No | Preconditioner matrix type selector. |
| `ppdegree` | int | `0` | No | Polynomial-preconditioner degree. |
| `RBdim` | int | `5` | No | Reduced-basis preconditioner dimension. |
| `matvecorder` | int | `1` | No | Matrix-vector approximation order. |
| `matvectol` | float | `1e-3` | No | Matrix-vector finite-difference tolerance. |

Frontend defaults can differ from Text2Code defaults. For example, MATLAB and
Python currently initialize `GMRESrestart = 100`, `preconditioner = 1`, and
`RBdim = 0`.

## Physics Parameters and Sweeps

| Key | Type | Default | Required | Notes |
| --- | --- | --- | --- | --- |
| `physicsparam` | list(float) | empty | Yes | Runtime physics parameter vector `mu`. |
| `physicsparamcases` | matrix(float) | empty | No | Standalone Text2Code sweep cases; each row has length `numel(physicsparam)`. |
| `physicsparamwarmstart` | int | `0` | No | `1` reuses previous case solution during standalone sweeps. |
| `externalparam` | list(float) | empty | No | Extra model parameters serialized separately from `physicsparam`. |
| `tau` | list(float) | empty | Yes | Stabilization parameters. |
| `uinf` | frontend field | frontend default | No | Frontend free-stream/reference values; not a Text2Code required key. |

Example:

```text
physicsparam = [1.4, 0.72, 1000.0, 0.3];
physicsparamcases = [
  [1.4, 0.72, 500.0, 0.3],
  [1.4, 0.72, 1000.0, 0.3],
  [1.4, 0.72, 1500.0, 0.3]
];
physicsparamwarmstart = 1;
```

See [Parameter sweeps](parameter-sweeps.md) for output layout and frontend
syntax.

## Boundary and Interface Configuration

| Key | Type | Default | Required | Notes |
| --- | --- | --- | --- | --- |
| `boundaryconditions` | list(int) | empty | Yes | Boundary condition code for each boundary expression. |
| `boundaryexpressions` | list(string) | empty | Yes | Geometry predicates evaluated on mesh faces. |
| `curvedboundaries` | list(int) | empty | No | Curved-boundary flags. |
| `curvedboundaryexprs` | list(string) | empty | No | Expressions for curved-boundary projection. |
| `periodicboundaries1` | list(int) | empty | No | First periodic boundary ID set. |
| `periodicexprs1` | list(string) | empty | No | Matching coordinate expressions for first periodic set. |
| `periodicboundaries2` | list(int) | empty | No | Second periodic boundary ID set. |
| `periodicexprs2` | list(string) | empty | No | Matching coordinate expressions for second periodic set. |
| `interfaceconditions` | list(int) | empty | No | Coupled-interface condition codes. |
| `interfacefluxmap` | list(int) | empty | No | Interface flux component map. |
| `cartgridpart` | list(int) | empty | No | Cartesian grid partition specification. |
| `extFhat` | int | `0` | No | Enable external interface flux hook. |
| `extUhat` | int | `0` | No | Enable external interface trace hook. |
| `extStab` | int | `0` | No | Enable external stabilization hook. |

See [Boundary conditions](boundary-conditions.md) for model-function
interfaces.

## Stabilization, Closure, and AV

| Key | Type | Default | Required | Notes |
| --- | --- | --- | --- | --- |
| `convStabMethod` | int | `0` | No | Convective stabilization selector. |
| `diffStabMethod` | int | `0` | No | Diffusive stabilization selector. |
| `viscosityModel` | int | `0` | No | Viscosity-model selector. |
| `SGSmodel` | int | `0` | No | Sub-grid-scale model selector. |
| `rotatingFrame` | int | `0` | No | Rotating-frame flag. |
| `ALE` | int | `0` | No | Arbitrary Lagrangian-Eulerian flag. |
| `AV` | int | `0` | No | Artificial-viscosity flag. |
| `AVdistfunction` | int | `0` | No | AV distance-function flag. |
| `AVsmoothingIter` | int | `2` | No | Number of AV smoothing iterations. |
| `frozenAVflag` | int | `1` | No | Freeze AV field where supported. |
| `avparam1` | list(float) | empty | No | Artificial-viscosity parameter vector. |
| `avparam2` | list(float) | empty | No | Additional artificial-viscosity parameter vector. |

## Synthetic Turbulence and DAE

| Key | Type | Default | Required | Notes |
| --- | --- | --- | --- | --- |
| `stgNmode` | int | `0` | No | Synthetic turbulence mode count. |
| `stgib` | list(float) | empty | No | Synthetic turbulence boundary data. |
| `stgdata` | list(float) | empty | No | Synthetic turbulence mode data. |
| `stgparam` | list(float) | empty | No | Synthetic turbulence parameters. |
| `dae_steps` | int | `0` | No | Number of DAE substeps. |
| `dae_dt` | list(float) | empty | No | DAE pseudo-time steps. |
| `dae_alpha` | float | `1.0` | No | DAE coefficient. |
| `dae_beta` | float | `0.0` | No | DAE coefficient. |
| `dae_gamma` | float | `0.0` | No | DAE coefficient. |
| `dae_epsilon` | float | `0.0` | No | DAE coefficient. |

## Output and Postprocessing

| Key | Type | Default | Required | Notes |
| --- | --- | --- | --- | --- |
| `saveParaview` | int | `0` | No | Enables VTK/ParaView output when visualization fields exist. |
| `saveSolFreq` | int | `1` | No | Saved solution frequency. |
| `saveSolOpt` | int | `1` | No | Saved solution option; implementation distinguishes compact/full solution storage. |
| `timestepOffset` | int | `0` | No | Offset used in time-dependent/restart output naming. |
| `saveSolBouFreq` | int | `0` | No | Boundary solution save frequency. |
| `ibs` | int | `0` | No | Boundary ID for boundary-solution output. |
| `saveResNorm` | int | `0` | No | Save residual norm history. |
| `compudgavg` | int | `0` | No | Compute averaged solution field where supported. |

Frontend-only execution field:

| Field | Type | Default | Scope | Notes |
| --- | --- | --- | --- | --- |
| `executionmode` | int | `0` | Frontend-only | `0` runs solve mode; `1` launches standalone postprocess mode. Not parsed from `pdeapp.txt`. |

See [Postprocessing](postprocessing.md).

## Minimal `pdeapp.txt`

```text
model = "ModelD";
modelfile = "pdemodel.txt";
meshfile = "mesh.bin";
discretization = "hdg";
platform = "cpu";
mpiprocs = 1;
porder = 2;
pgauss = 4;
physicsparam = [1.0];
tau = [1.0];
boundaryconditions = [1, 2, 3, 4];
boundaryexpressions = ["x < 1e-8", "x > 1-1e-8", "y < 1e-8", "y > 1-1e-8"];
```

## Complete-Style Example Fragment

```text
model = "ModelD";
modelfile = "pdemodel.txt";
meshfile = "mesh.bin";
discretization = "hdg";
platform = "cpu";
mpiprocs = 4;
porder = 3;
pgauss = 6;
physicsparam = [1.4, 0.72, 5000.0, 0.3];
tau = [1.0];
dt = [repeat(0.001, 100)];
saveSolFreq = 10;
saveParaview = 1;
nsca = 2;
nvec = 1;
boundaryconditions = [1, 2, 3, 4];
boundaryexpressions = ["x < -10", "x > 20", "wall_distance < 1e-8", "farfield > 0"];
```
