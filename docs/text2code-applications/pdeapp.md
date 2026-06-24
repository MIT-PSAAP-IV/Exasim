# `pdeapp.txt` Guide

`pdeapp.txt` is the Text2Code application file. It is a flat list of
semicolon-terminated `key = value;` statements that configure the model file,
mesh file, discretization, solver, runtime backend, parameter sweeps, and output
behavior.

For the exhaustive keyword table, see
[pdeapp.txt fields](../reference/pdeapp.md). This page explains how to use those
keywords in complete applications.

## Syntax Rules

```text
key = value;
```

Common value forms:

```text
model = "ModelD";
porder = 3;
NewtonTol = 1e-06;
physicsparam = [1.0, 0.72, 1000.0];
boundaryexpressions = ["abs(y)<1e-8", "abs(x-1)<1e-8"];
physicsparamcases = [0.5; 1.0; 2.0];
```

Important parser behavior:

- statements end with `;`;
- strings are quoted;
- numeric lists use brackets;
- matrices can use nested rows or semicolon-separated rows;
- numeric expressions in boundary expressions are evaluated by Text2Code;
- write floating-point values with a decimal or exponent when the backend field
  is floating point, for example `time = 0.0;`.

## Required Fields

The parser requires these keys:

| Key | Purpose |
| --- | --- |
| `model` | Model class selector such as `"ModelD"`. |
| `modelfile` | Path to `pdemodel.txt`. |
| `meshfile` | Path to mesh input, commonly `grid.bin`. |
| `discretization` | `"ldg"` or `"hdg"`. |
| `platform` | `"cpu"` or `"gpu"`. |
| `mpiprocs` | Number of MPI ranks. |
| `porder` | Polynomial order. |
| `pgauss` | Quadrature degree/order. |
| `physicsparam` | Runtime physical parameter vector. |
| `tau` | Stabilization parameter vector. |
| `boundaryconditions` | Boundary-condition IDs. |
| `boundaryexpressions` | Geometric predicates for boundary groups. |

## Application Settings

```text
model = "ModelD";
modelfile = "pdemodel.txt";
meshfile = "grid.bin";
discretization = "hdg";
platform = "cpu";
modelnumber = 0;
gencode = 1;
gendatain = 1;
```

`gencode = 1` asks Text2Code to generate model C++ kernels from
`pdemodel.txt`. `gendatain = 1` asks it to write the backend input bundle.

## Physics Configuration

```text
ncu = 1;
ncv = 0;
ncw = 0;
physicsparam = [1.0];
externalparam = [0.0, 0.0];
tau = [1.0];
```

`physicsparam` is passed to `pdemodel.txt` functions as `mu`. `externalparam`
is passed as `eta`/external data in the generated model interface. Keep the
length and meaning of these vectors fixed across runs and sweeps.

## Mesh Configuration

```text
meshfile = "grid.bin";
boundaryconditions = [1, 1, 1, 1];
boundaryexpressions = [
  "abs(y)<1e-8",
  "abs(x-1)<1e-8",
  "abs(y-1)<1e-8",
  "abs(x)<1e-8"
];
curvedboundaries = [0, 0, 0, 0];
curvedboundaryexprs = ["", "", "", ""];
```

Boundary expressions classify mesh faces geometrically. Boundary-condition IDs
must match the boundary logic implemented in `pdemodel.txt` functions such as
`Ubou`, `Fbou`, and `FbouHdg`.

## Solver Settings

```text
NewtonIter = 20;
NewtonTol = 1e-06;
GMRESiter = 200;
GMRESrestart = 50;
GMREStol = 1e-08;
preconditioner = 1;
precMatrixType = 0;
```

These fields configure the backend nonlinear and linear solvers. Use tighter
tolerances only when the physics and discretization justify the additional
work.

## Time Integration Settings

Steady-state run:

```text
dt = [0.0];
time = 0.0;
torder = 1;
nstage = 1;
```

Time-dependent run:

```text
time = 0.0;
dt = [repeat(1e-3, 100)];
saveSolFreq = 10;
torder = 2;
nstage = 2;
```

A positive entry in `dt` marks the problem as time-dependent.

## Execution, MPI, And GPU Settings

Serial CPU:

```text
platform = "cpu";
mpiprocs = 1;
```

MPI CPU:

```text
platform = "cpu";
mpiprocs = 4;
```

GPU:

```text
platform = "gpu";
mpiprocs = 1;
```

The installed Exasim package and CMake app determine whether CUDA or HIP is
used for GPU execution.

## Parameter Sweep Settings

```text
physicsparam = [1.4, 0.72, 500.0];
physicsparamcases = [
  1.4, 0.72, 500.0;
  1.4, 0.72, 1000.0;
  1.4, 0.72, 1500.0
];
physicsparamwarmstart = 1;
```

Each row of `physicsparamcases` is one run. The standalone app writes outputs
to deterministic directories such as `dataout/paramcase_0001/` and writes
`physicsparam.txt` metadata for reproducibility.

## Postprocessing And Output Settings

```text
saveParaview = 1;
nsca = 2;
nvec = 1;
nten = 0;
nvqoi = 2;
saveResNorm = 1;
saveSolFreq = 1;
saveSolOpt = 1;
```

Visualization counts must match generated `VisScalars`, `VisVectors`, and
`VisTensors` outputs in `pdemodel.txt`. QoI counts must match `QoIvolume` and
`QoIboundary` outputs.

## Complete Poisson App Example

```text
model = "ModelD";
modelfile = "pdemodel.txt";
meshfile = "grid.bin";
discretization = "hdg";
platform = "cpu";
mpiprocs = 1;

porder = 3;
pgauss = 6;
ncu = 1;
ncv = 0;
ncw = 0;
nsca = 2;
nvec = 1;
nvqoi = 2;

NewtonIter = 20;
NewtonTol = 1e-06;
GMRESiter = 200;
GMRESrestart = 50;
GMREStol = 1e-08;
preconditioner = 0;

time = 0.0;
dt = [0.0];
tau = [1.0];
physicsparam = [1.0];
externalparam = [0.0, 0.0];

boundaryconditions = [1, 1, 1, 1];
boundaryexpressions = ["abs(y)<1e-8", "abs(x-1)<1e-8", "abs(y-1)<1e-8", "abs(x)<1e-8"];
```

## See Also

- [Complete pdeapp.txt field reference](../reference/pdeapp.md)
- [pdemodel.txt guide](pdemodel.md)
- [Application setup guide](application-setup.md)
- [Parameter sweeps](../usage-modes/parameter-sweeps.md)
