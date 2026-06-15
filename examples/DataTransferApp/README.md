# Data-transfer app examples

A "data-transfer app" is a self-contained, **relocatable** bundle that lets you
reproduce an Exasim run on another machine (e.g. an HPC cluster) without the
language frontend. You generate it on one machine and build + run it on another
that only needs an **Exasim install**.

## How to produce one

Any frontend app produces a bundle by setting one field before calling
`exasim(...)`:

```python
pde['exportapp'] = "/path/to/bundle"   # Python
```
```matlab
pde.exportapp = "/path/to/bundle";     % MATLAB
```
```julia
pde.exportapp = "/path/to/bundle"      # Julia
```

The [`poisson2d/`](poisson2d/) example provides all three frontends — each is
the standard Poisson2D example with exactly that one extra line:
[`pdeapp.py`](poisson2d/pdeapp.py), [`pdeapp.m`](poisson2d/pdeapp.m),
[`pdeapp.jl`](poisson2d/pdeapp.jl).

```sh
cd poisson2d
python pdeapp.py          # Python  — runs the solver AND writes ./poisson2d-bundle
# or:  matlab -batch pdeapp        # MATLAB
# or:  julia pdeapp.jl             # Julia
```

## What's in the bundle

```
poisson2d-bundle/
├── datain/         binary solver inputs (mesh, master, app, solution);
│                   boundary conditions are already resolved here.
├── dataout/        empty; the solver writes outputs here.
├── kernels/        generated kernel .cpp set for this model.
├── pdemodel.txt    text2code DSL regenerated from the symbolic model.
├── CMakeLists.txt  relocatable build (uses find_package(Exasim)).
├── main.cpp        the solver entry point.
├── run.sh          build + run helper.
├── manifest.json   provenance (model id, variant, process count, ...).
└── pdemodel.py     copy of the source model, for reference.
```

The bundle carries **no absolute paths**; the only external dependency is an
Exasim install on the build machine.

## Build + run it elsewhere

Copy the bundle to the target machine, then:

```sh
EXASIM_ROOT=/path/to/exasim/install ./run.sh
```

The kernels and `datain` are **architecture-independent**, so the same bundle
can target whatever variant the build machine provides:

```sh
EXASIM_ROOT=/path/to/install EXASIM_VARIANT=gpu ./run.sh   # cpu | cpumpi | gpu | gpumpi
```

By default the export step also builds + runs the bundle locally (in a throwaway
directory) to verify it works before hand-off, leaving the shipped bundle
pristine (no `build/`, empty `dataout/`).
