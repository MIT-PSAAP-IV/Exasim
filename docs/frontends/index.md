# Frontends (Python / Julia / MATLAB)

The frontends let you author a PDE model interactively in Python, Julia, or
MATLAB, then generate, build, and run it without writing C++ or CMake. They drive
the [shared-library](../usage-modes/shared-library.md) path: the symbolic stack
generates the model kernels, builds them into a model library, and runs a
pre-built solver against it.

Prerequisite: an [installed Exasim](../install/index.md) built with
`-DEXASIM_FRONTENDS=ON` (the default).

## The authoring flow

All three languages follow the same shape:

1. Initialize the `pde` and `mesh` objects.
2. Point `pde` at a model file (`pdemodel.{py,jl,m}`) defining the PDE — `flux`,
   `source`, boundary terms, etc. (the same functions as the
   [`pdemodel.txt`](../reference/pdemodel.md) DSL, written in the host language).
3. Set discretization and physics parameters and build/load the mesh.
4. Call `exasim(...)`, which generates the kernels, builds the model library, and
   runs the solver.

The fields you set on `pde` correspond to the
[`pdeapp.txt` keys](../reference/pdeapp.md); the model functions correspond to the
[model contract](../reference/model-contract.md) methods.

## Python

```python
from exasim import initializeexasim, exasim
pde, mesh = initializeexasim()
pde['model'] = "ModelD"
pde['modelfile'] = "pdemodel"      # pdemodel.py defining flux/source/...
# ... discretization / physics parameters, mesh ...
sol, pde, mesh = exasim(pde, mesh)
```

Run an example directly from its directory: `python3 pdeapp.py`. Configure
`-DEXASIM_PIP_INSTALL=ON` to pip-install the `exasim` package at install time.

## Julia

```julia
push!(LOAD_PATH, "/path/to/prefix/share/exasim/julia")   # or Pkg.develop(path=...)
using Exasim
pde, mesh = Exasim.initializeexasim()
pde.model = "ModelD"
include("pdemodel.jl")             # defines flux/source/... in Main
# ... discretization / physics parameters, mesh ...
sol, pde, mesh = Exasim.exasim(pde, mesh)
```

Configure `-DEXASIM_JULIA_DEVELOP=ON` and the install runs `Pkg.develop` on the
installed package, so `using Exasim` needs no `LOAD_PATH` setup.

## MATLAB

```matlab
run('/path/to/prefix/share/exasim/matlab/exasim_setup.m')
[pde, mesh] = initializeexasim();
pde.model = "ModelD"; pde.modelfile = "pdemodel";   % pdemodel.m on the path
% ... discretization / physics parameters, mesh ...
[sol, pde, mesh] = exasim(pde, mesh);
```

## Build reuse and the model cache

The generated model is compiled into a dynamic library under the hidden
`pde.builddir` (default `<cwd>/.exasim/`), and reuse is hash-based: an unchanged
model skips compilation entirely. Built libraries are also cached **per user**
under `~/.exasim/cache/<modelID>/<digest>/`, so an identical model reuses the
build even from a fresh directory. See
[Shared library → build artifacts and reuse](../usage-modes/shared-library.md#build-artifacts-and-reuse)
for the full layout and the cache semantics.

## exportapp — the data-transfer bundle

The frontends can export a self-contained application bundle (`exportapp` in each
frontend's `Gencode/`): the generated kernels, `datain` inputs, CMake project,
and `main.cpp`, packaged so the app can be built and run elsewhere (e.g. moved to
an HPC system). This is the hand-off from interactive authoring to a standalone
[built-in](../usage-modes/builtin.md) / [shared-library](../usage-modes/shared-library.md)
build.
