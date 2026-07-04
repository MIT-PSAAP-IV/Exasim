# Frontend API Reference

This page summarizes the most commonly used frontend functions and fields.
Detailed behavior is implemented in the language-specific frontend directories:

- `frontends/Matlab`
- `frontends/Python/exasim`
- `frontends/Julia/Exasim`

## Function Reference

| Concept | MATLAB | Python | Julia | Purpose |
| --- | --- | --- | --- | --- |
| Initialize app | `initializeexasim()` | `initializeexasim()` | `Exasim.initializeexasim()` | Return default `pde` and `mesh`. |
| Initialize PDE only | `initializepde(version)` | `initializepde(version)` | `Exasim.initializepde(version)` | Return default PDE configuration. |
| Preprocess | `preprocessing(pde, mesh)` | `preprocessing(pde, mesh)` | `Exasim.preprocessing(pde, mesh)` | Write backend input files and return updated `pde`, `mesh`, `master`, and `dmd`. |
| Run workflow | `exasim(pde, mesh)` | `exasim(pde, mesh)` | `Exasim.exasim(pde, mesh)` | Preprocess, generate, build, run, and fetch outputs. |
| Compile | `cmakecompile(pde)` | `cmakecompile(pde)` | `Exasim.cmakecompile(pde)` | Configure/build generated CMake app. |
| Run executable | `runcode(pde, numpde)` | `runcode(pde, numpde)` | `Exasim.runcode(pde, numpde)` | Launch solve or postprocess executable. |
| Export app | `exportapp(pde, dest)` | `exportapp(pde, dest)` | `Exasim.exportapp(pde, dest)` | Create standalone app bundle. |
| Export Text2Code package | `exporttext2code(pde, mesh, dest)` | `exporttext2code(pde, mesh, dest)` | `Exasim.exporttext2code(pde, mesh, dest)` | Create `pdeapp.txt`, `pdemodel.txt`, mesh, and optional field binaries for Text2Code regeneration. |
| Visualize | `vis(visfields, app, mesh)` | `vis(visfields, app, mesh)` | `Exasim.vis(visfields, app, mesh)` | Write frontend visualization output. |

Example:

```python
from exasim import initializeexasim, exasim, exportapp, exporttext2code

pde, mesh = initializeexasim()
pde["modelfile"] = "pdemodel"
sol, pde, mesh = exasim(pde, mesh)
exportapp(pde, "standalone_app")
exporttext2code(pde, mesh, "text2code_package")
```

## Common `pde` Fields

| Field | Default role | Notes |
| --- | --- | --- |
| `model` / `pdemodel` | Model family name | MATLAB historically initializes `pdemodel`; Python/Julia expose `model`. |
| `modelfile` | Host-language model file | Omit extension in examples. |
| `physicsparam` | Single runtime physics vector | Passed to model callbacks as `param`. |
| `physicsparamsweep` | Optional matrix/list of physics vectors | Rows are cases. |
| `physicsparamwarmstart` | Optional sweep continuation flag | First case uses standard initialization; later cases reuse previous converged state. |
| `platform` | CPU/GPU selection | Must match installed backend support. |
| `mpiprocs` | Number of MPI ranks | Controls preprocessing and launch. |
| `datapath` | Runtime input/output root | Defaults to current directory. |
| `builddir` / `buildpath` | Generated build root | Defaults to `.exasim`. |
| `dataoutpath` | Optional output override | Used for sweep cases and custom output placement. |
| `executionmode` | Runtime mode | `0` solve, `1` postprocess. |
| `saveParaview` | Backend VTK output flag | Requires visualization field counts and callbacks. |
| `saveResNorm` | Residual-history output flag | Writes residual norm files. |

## Model Callback Names

Callbacks are generated only when the model and dimensions require them. Common
names include:

```text
initu, initq, initw, initodg
flux, source, tdfunc
ubou, fbou, fbouhdg, fhat, uhat, stab
visscalars, visvectors, vistensors
qoivolume, qoiboundary
```

See [Model contract](../reference/model-contract.md) for semantics.
See [pdemodel abstraction](pdemodel.md) for frontend-specific examples and data
flow.

## Language Differences

| Topic | MATLAB | Python | Julia |
| --- | --- | --- | --- |
| Container | `struct` | `dict` | `PDEStruct` |
| Field access | `pde.field` | `pde["field"]` | `pde.field` |
| Package setup | `exasim_setup.m` | Python package/import path | `using Exasim` |
| Array convention | MATLAB column-major arrays | NumPy arrays; Exasim data follows backend layout | Julia column-major arrays |
| Export standalone app | `exportapp(pde, dest)` | `exportapp(pde, dest)` | `Exasim.exportapp(pde, dest)` |
| Export Text2Code package | `exporttext2code(pde, mesh, dest)` | `exporttext2code(pde, mesh, dest)` | `Exasim.exporttext2code(pde, mesh, dest)` |

## Text Frontend Compatibility

The `pdeapp.txt` workflow is not an interactive language frontend, but it uses
the same core fields and the same backend runtime semantics. Use it when you
want a compact, frontend-free input file for text2code-generated standalone
applications.

```text
physicsparam = 1.4 0.72 2000.0
saveParaview = 1
```

See [pdeapp.txt fields](../reference/pdeapp.md) and
[text2code](../reference/text2code.md).

## See Also

- [Core data structures](data-structures.md)
- [Configuration](configuration.md)
- [Preprocessing](preprocessing.md)
- [pdemodel abstraction](pdemodel.md)
- [Execution workflow](execution.md)
