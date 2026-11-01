# Poisson on the unit square

This thin variant reuses `examples/Poisson/poisson2d` and the native Exasim
diffusion operators. It solves `-Δu = 2π² sin(πx)sin(πy)` with zero Dirichlet
data. The exact solution is `u=sin(πx)sin(πy)` and the stored auxiliary field
is `q=-∇u`. Common physical parameters are `[κ,ub]=[1,0]` and stabilization is
`tau=1`. The H region is the fixed half-square `x<=0.5`; use even N so its
interface remains face aligned on every mesh. Pure schemes are measured on
those same subregions.

## Requirements

Build and install the current Exasim with its CPU, MPI, CPU preprocessing and
frontend support. The normal installation now supplies
`bin/exasim_hdgldg_preprocess`; no test-project build is required. Set up the
chosen installed frontend as for `examples/Poisson/poisson2d`. An older Exasim
installation must be rebuilt/reinstalled to obtain the new `hdgldgsolve` API
and coupled runtime.

The ordinary frontend dependencies apply: MATLAB Symbolic Math Toolbox,
Python NumPy/SymPy, or Julia Exasim/SymPy. CMake, a C++ compiler and MPI must
be available. Only the verified CPU scope is advertised here.

## MATLAB

With the MATLAB frontend on the path, change to this directory and run:

```matlab
pdeapp
```

If needed, first run your installation's
`share/exasim/matlab/exasim_setup.m`. A complete source checkout also has a
source-frontend setup fallback, as in the standard Poisson examples. For a
custom installation location, set `EXASIM_PREFIX` to that prefix.

The script defines `pde` and `mesh`, generates native MATLAB kernels, builds
the application, prepares both regions, runs MPI, and leaves `sol`, `pde`
and `mesh` in the workspace. It plots the temperature in a desktop session.
There is no required function argument or Python launcher.

## Python

After the usual installed-package setup (`import exasim` must work), run:

```sh
python pdeapp.py
```

The script uses native Python/SymPy generation and the installed solver. It
does not invoke MATLAB, Julia or the Text2Code compiler. Boundary expressions
are scalar strings accepted by the shared backend preprocessor.

## Julia

With the installed Exasim package available:

```sh
julia pdeapp.jl
```

Alternatively, select its installed project explicitly:

```sh
julia --project="$EXASIM_PREFIX/share/exasim/julia/Exasim" pdeapp.jl
```

This uses native Julia symbolic generation. No Python orchestration or
Text2Code compiler is needed.

## Parameters and results

The scripts use `N=4`, `porder=2`, one HDG rank and one LDG rank. Edit the
visible settings in the selected `pdeapp` to change them. Use even N and
recompute the labels from cell centers, keeping the interface at x=1/2.
`labels=1` selects HDG, `labels=0` selects LDG. `nhdg` selects the number of
HDG ranks; the remaining `pde.mpiprocs` ranks use LDG.

For selection by boundary-condition code and element-layer count instead of
explicit labels, see the boundary-layer examples in the parent
[`README.md`](../README.md). NACA2D uses this option directly.

Each native entrypoint calls `hdgldgsolve(pde,mesh,labels,nhdg)`, which reuses
the native generator/compiler, existing serializers, installed shared CPU
preprocessor and coupled solver. Pure `exasim(pde,mesh)` remains unchanged.
The Python/Julia serializer may also emit an unused `pdemodel.txt`; their
solver kernels are still generated natively, not from that file.

Each run creates a new directory under `pde.datapath` and prints its location.
Model builds use `pde.builddir`. There are no developer-specific build paths
or required temporary dependencies. Native H/L files retain their own
numbering; the frontend assembles `sol` in the **original full-mesh element
order**. Its components are `[u,qx,qy]`, with `q=-grad(u)`. At the default
settings its shape is `[9,3,16]`.

`HDGLDG_RUN_DIR` is an optional environment override for disposable automated
checks; normal users do not need it.

## Independent Text2Code workflow

`pdeapp.txt` and `pdemodel.txt` define the same problem independently. The
supplied `grid.bin` is the 4x4 unit-square grid; `selection.txt` labels its
left half HDG and right half LDG. If changing the grid, update its selection
consistently.

From this directory, with `EXASIM_PREFIX` pointing to your installed Exasim:

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

The CMake project uses the installed Text2Code and external-model helper; no
source-build package or `tests/hdgldg` executable is required. Choose another
new absolute work directory if you prefer persistent output. Existing output
directories are not overwritten.

The scripts in `../common` remain optional development campaigns, not
prerequisites for running this example.
