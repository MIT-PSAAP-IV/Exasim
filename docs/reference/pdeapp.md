# pdeapp.txt field reference

`pdeapp.txt` is the solver-setup file: a flat list of `key = value;` statements
read by text2code (`readpdeapp.cpp`). It is consumed by every
[usage mode](../usage-modes/index.md). This page documents every recognized key.

For the *model* definition (the PDE math) see the
[pdemodel.txt syntax reference](pdemodel.md); for the C++ struct those keys
ultimately configure, see the [model contract](model-contract.md).

## Syntax notes

- Each statement ends at `;`. Strings are quoted (`key = "value";`); numeric
  **lists** use brackets (`key = [a, b, c];`); string lists use
  `key = ["a", "b"];`.
- A scalar is stored as a **float** if its text contains `.` or `e`
  (`1e-06`, `0.001`), otherwise as an **int**. So `time = 0;` is read as int and
  will *not* reach the float field `pde.time` — write `time = 0.0;`.
- Lists support `repeat(value, count)`, e.g. `tau = [repeat(1.0, 4)];`.
- Special keys are matched by substring, so order matters; the parser guards the
  built-in keys, but avoid custom keys that contain a recognized key as a
  substring.
- **Required keys** (the parser aborts if missing): `model`, `modelfile`,
  `meshfile`, `discretization`, `platform`, `mpiprocs`, `porder`, `pgauss`,
  `physicsparam`, `tau`, `boundaryconditions`, `boundaryexpressions`.

A default of `—` means the parser sets none (the key is applied only if present).

## Model / app selection

| Key | Type | Default | Meaning |
|---|---|---|---|
| `model` | string | `"ModelD"` | Model class selector. **Required.** |
| `modelfile` | string | `"pdemodel.txt"` | Path to the symbolic model file. **Required.** |
| `meshfile` | string | `"mesh.bin"` | Path to the binary mesh. **Required.** |
| `discretization` | string | `"ldg"` | `ldg`/`LDG` → `hybrid=0`; `hdg`/`HDG` → `hybrid=1`. **Required.** |
| `platform` | string | `"cpu"` | Compute backend (`cpu`/`gpu`). **Required.** |
| `modelnumber` | int | `0` | Model index; `>0` suffixes the `datain`/`dataout` dirs. |
| `builtinmodelID` | int | — | Built-in model ID (consumed by the provider; parsed but not stored in the PDE struct). |
| `gendatain` | int | `1` | Generate the `datain` input bundle. |
| `gencode` | int | `1` | Generate model C++ code. |
| `writemeshsol` | int | `1` | Write mesh/solution output. |
| `runmode` | int | `0` | Run-mode selector. |
| `debugmode` | int | `0` | `1` prints parsed params. |
| `mpiprocs` | int | `1` | Number of MPI processes. **Required.** |
| `nodetype` | int | `1` | Node distribution (uniform vs Gauss–Lobatto). |
| `exasimpath` / `datapath` | string | auto | Exasim root / base data dir (auto-derived if unset). |
| `xdgfile` / `udgfile` / `vdgfile` / `wdgfile` / `uhatfile` / `partitionfile` | string | `""` | Optional input-data files. |

## Field counts

| Key | Type | Default | Meaning |
|---|---|---|---|
| `ncu` | int | `1` | Number of state components $u$. |
| `ncv` | int | `0` | Number of VDG (`v`) components. |
| `ncw` | int | `0` | Number of WDG (`w`) components. |
| `nsca` / `nvec` / `nten` | int | `0` | Scalar / vector / tensor visualization output counts. |
| `nsurf` | int | `0` | Surface output count. |
| `nvqoi` | int | `0` | Volume quantity-of-interest count. |
| `neb` | int | `4096` | Element block size (kernel batching). |
| `nfb` | int | `8192` | Face block size (kernel batching). |

## Discretization

| Key | Type | Default | Meaning |
|---|---|---|---|
| `porder` | int | `1` | Polynomial order of the solution basis. **Required.** |
| `pgauss` | int | `2` | Gauss quadrature degree. **Required.** |
| `torder` | int | `1` | Temporal accuracy order. |
| `nstage` | int | `1` | Number of time-integrator stages. |
| `temporalscheme` | int | `0` | Time-integration scheme selector. |

## Solver

| Key | Type | Default | Meaning |
|---|---|---|---|
| `nonlinearsolver` | int | `0` | Nonlinear solver selector. |
| `NewtonIter` | int | `20` | Max Newton iterations. |
| `NewtonTol` | float | `1e-6` | Newton tolerance. |
| `NLparam` / `NLMatrixType` | float / int | `0.0` / `0` | Nonlinear solver parameter / matrix type. |
| `linearsolver` | int | `0` | Linear solver selector. |
| `GMRESiter` | int | `200` | Max GMRES iterations. |
| `GMRESrestart` | int | `25` | GMRES restart length. |
| `GMREStol` | float | `1e-3` | GMRES tolerance. |
| `GMRESortho` | int | `0` | GMRES orthogonalization scheme. |
| `preconditioner` / `precMatrixType` | int | `0` | Preconditioner / its matrix type. |
| `ppdegree` | int | `0` | Polynomial-preconditioner degree. |
| `RBdim` | int | `5` | Reduced-basis dimension. |
| `matvecorder` | int | `1` | Jacobian-matvec approximation order. |
| `matvectol` | float | `1e-3` | Matvec finite-difference tolerance. |

## Time / physics

| Key | Type | Default | Meaning |
|---|---|---|---|
| `time` | float | `0.0` | Initial/simulation time. |
| `tau` | list(float) | — | HDG/LDG stabilization parameter(s). **Required.** |
| `dt` | list(float) | — | Time-step sequence; first nonzero entry sets `tdep=1`. |
| `physicsparam` | list(float) | — | PDE physical coefficients ($\mu$). **Required.** |
| `physicsparamsweep` | samples/grid | empty | Optional sweep over multiple `physicsparam` vectors. |
| `externalparam` | list(float) | — | External/auxiliary parameters passed to the model. |
| `tdep` / `wave` / `tdfunc` / `sourcefunc` | int | `0` / `0` / `1` / `1` | Time-dependent / wave / time-derivative-fn / source-fn flags. |

### Parameter sweeps

Use `physicsparamsweep` to run the same generated model over multiple
`physicsparam` vectors without recompiling between cases. The frontend compiles
once using the first case, then rewrites the runtime app input and runs cases
sequentially. Each case writes to a deterministic directory:

```text
dataout/paramcase_0001/
dataout/paramcase_0002/
...
```

Each case directory contains `physicsparam.txt`, which records the parameter
vector used for that run.

Supported forms are:

```matlab
% one row per case
pde.physicsparam = [1.0];
pde.physicsparamsweep = [0.5; 1.0; 2.0];

% structured Cartesian product
pde.physicsparam = [1.0 0.0];
pde.physicsparamsweep.grid = {[0.5 1.0 2.0], [0.0 1.0]};
```

```python
pde['physicsparam'] = numpy.array([1.0])
pde['physicsparamsweep'] = numpy.array([[0.5], [1.0], [2.0]])

pde['physicsparam'] = numpy.array([1.0, 0.0])
pde['physicsparamsweep'] = {'grid': [[0.5, 1.0, 2.0], [0.0, 1.0]]}
```

```julia
pde.physicsparam = [1.0 0.0]
pde.physicsparamsweep = [0.5 0.0; 1.0 0.0; 2.0 0.0]

pde.physicsparamsweep = Dict(:grid => [[0.5, 1.0, 2.0], [0.0, 1.0]])
```

Existing single-case apps do not need to set `physicsparamsweep`.

## Stabilization & closure models

| Key | Type | Default | Meaning |
|---|---|---|---|
| `convStabMethod` / `diffStabMethod` | int | `0` | Convective / diffusive stabilization method. |
| `viscosityModel` / `SGSmodel` | int | `0` | Viscosity / sub-grid-scale model. |
| `rotatingFrame` / `ALE` | int | `0` | Rotating-frame / ALE flags. |
| `AV` | int | `0` | Artificial-viscosity flag. |
| `AVdistfunction` / `AVsmoothingIter` / `frozenAVflag` | int | `0` / `2` / `1` | AV distance function / smoothing iters / freeze flag. |
| `avparam1` / `avparam2` | list(float) | — | AV parameter sets. |

## Boundary / interface conditions

| Key | Type | Default | Meaning |
|---|---|---|---|
| `boundaryconditions` | list(int) | — | BC type per boundary. **Required.** |
| `boundaryexpressions` | list(string) | — | Geometric predicate selecting each boundary. **Required.** |
| `curvedboundaries` / `curvedboundaryexprs` | list(int) / list(string) | — | Curved-boundary flag / defining expression per boundary. |
| `periodicboundaries1` / `periodicexprs1` | list(int) / list(string) | — | First periodic set: boundary IDs / matching coordinate expressions. |
| `periodicboundaries2` / `periodicexprs2` | list(int) / list(string) | — | Second periodic set. |
| `interfaceconditions` | list(int) | — | Interface-condition codes (drives coupled-interface extraction). |
| `interfacefluxmap` | list(int) | — | Flux mapping across a coupled interface. |
| `cartgridpart` | list(int) | — | Cartesian grid partitioning spec. |

## Coupling / external-flux hooks

| Key | Type | Default | Meaning |
|---|---|---|---|
| `extFhat` / `extUhat` / `extStab` | int | `0` | Use external numerical flux $\hat F$ / trace $\hat u$ / stabilization (see [Driving the solver](../driving-the-solver.md)). |
| `compudgavg` | int | `0` | Compute time-averaged UDG. |
| `vindx` | list(float) | — | Variable-index map. |

## DAE / pseudo-transient

| Key | Type | Default | Meaning |
|---|---|---|---|
| `dae_steps` | int | `0` | Number of DAE sub-steps. |
| `dae_dt` | list(float) | — | DAE pseudo-time-step sequence. |
| `dae_alpha` / `dae_beta` / `dae_gamma` / `dae_epsilon` | float | `1.0` / `0.0` / `0.0` / `0.0` | DAE continuation coefficients. |

## Synthetic-turbulence generation (STG)

| Key | Type | Default | Meaning |
|---|---|---|---|
| `stgNmode` | int | `0` | Number of STG modes. |
| `stgib` / `stgdata` / `stgparam` | list(float) | — | STG inflow-boundary data / mode data / parameters. |

## I/O control

| Key | Type | Default | Meaning |
|---|---|---|---|
| `saveSolFreq` / `saveSolOpt` | int | `1` | Solution save frequency / option. |
| `saveSolBouFreq` | int | `0` | Boundary-solution save frequency. |
| `saveResNorm` | int | `0` | Save residual-norm history. |
| `timestepOffset` | int | `0` | Starting time-step index offset. |
| `ibs` | int | `0` | Boundary-solution save selector. |

## Relation to theory

Several keys set quantities in the [discretization theory](../theory/index.md):

- `physicsparam` → the PDE coefficients $\mu$ in the [flux/source terms](../theory/ldg-formulation.md).
- `tau` → the [HDG/LDG stabilization parameter](../theory/ldg-formulation.md) in
  the numerical flux $\hat f = f + \tau\,(u - \hat u)$.
- `porder` / `pgauss` → the polynomial basis order and quadrature degree of the
  [DG approximation](../theory/index.md).
- `torder` / `nstage` → the order and stage count of the DIRK time integration.
