# Poisson on the unit cube

Solve `-Δu=3π² sin(πx)sin(πy)sin(πz)` with zero Dirichlet data.
The exact solution is the product of the three sines, and `q=-grad(u)`.
The default mesh has 4×4×4 hexahedra and degree 2. Elements with center
`x<=0.5` use HDG on rank 0; the rest use LDG on rank 1.
The model and solver settings are identical in all four paths.

## Requirements

Use the normal CPU/MPI Exasim installation, including CPU preprocessing and
your chosen frontend. It must contain `bin/exasim_hdgldg_preprocess` and
the public `hdgldgsolve` helper. No test-project or private development build
is needed. Set `EXASIM_PREFIX` for a nondefault installation location.
Usual dependencies apply: MATLAB Symbolic Math Toolbox, Python NumPy/SymPy,
or the Julia Exasim/SymPy environment, plus CMake, a C++ compiler and MPI.

## Native examples

Change to this directory. With your installed MATLAB frontend on the path:

```matlab
pdeapp
```

If necessary, first run your installation's
`share/exasim/matlab/exasim_setup.m`. The source checkout also has a setup
fallback. The script leaves `sol,pde,mesh` in the workspace.

With the installed Python package importable:

```sh
python pdeapp.py
```

With the installed Julia project available:

```sh
julia pdeapp.jl
# Or select its installed project explicitly:
julia --project="$EXASIM_PREFIX/share/exasim/julia/Exasim" pdeapp.jl
```

Each script defines readable PDE, mesh and selection settings, generates
its model natively, builds the application, prepares the H/L inputs and runs
the coupled MPI solver. It needs no arguments and no Python orchestration.
Edit `N`, `porder` and the visible controls directly. Use even N to keep the
interface face aligned. Label 1 means HDG and label 0 means LDG.

`hdgldgsolve` returns the solution in original full-mesh element order,
with components `[u,qx,qy,qz]`. The default shape is `[27,4,64]`.
A fresh run directory is created under `pde.datapath` and printed.
`HDGLDG_RUN_DIR` optionally chooses a disposable location for automated checks.
Existing pure `exasim(pde,mesh)` workflows remain unchanged.
The Python/Julia serializer may emit an unused Text2Code model file; their
solver kernels are nevertheless generated natively.

## Independent Text2Code example

The supplied `grid.bin` is the same 4×4×4 hex mesh.
`selection.txt` contains the corresponding original-element labels and the
two rank assignments. If changing the mesh, update the selection too.

From this directory:

```sh
work=$(mktemp -d)
cmake -S . -B "$work/build" -DCMAKE_BUILD_TYPE=Release \
  -DExasim_DIR="$EXASIM_PREFIX/lib/cmake/Exasim"
cmake --build "$work/build" --parallel 2
"$EXASIM_PREFIX/bin/exasim_hdgldg_preprocess" \
  pdeapp.txt selection.txt "$work/datain"
mpiexec -np 2 "$work/build/exasimapp" \
  --hdgldg "$work/datain" "$work/dataout"
```

This uses installed Text2Code and the installed external-model helper. Choose
another new absolute work directory for persistent output. Existing output
directories are not overwritten. Only the CPU scope is verified.

The scripts in `../common` are optional development campaigns, not prerequisites
for running these user examples.
