# Mixed HDG/LDG examples

These are user examples, like `examples/Poisson/poisson2d`. Each directory
contains native MATLAB, Python and Julia scripts, plus an independent
Text2Code application. No test-project build or campaign runner is required.

| Directory | Problem | HDG region |
| --- | --- | --- |
| `Poisson` | Unit-square Poisson | Left half of the square |
| `Poisson3D` | Unit-cube Poisson | Left half of the cube |
| `Couette` | Compressible viscous Couette flow | Left half of the square |
| `ManufacturedNavierStokes` | Steady manufactured Navier–Stokes solution | Left half of the square |
| `NACA2D` | Viscous flow around an airfoil | One element layer from the wall |

## Setup and execution

Use a current CPU/MPI Exasim installation with frontend support and the
installed `bin/exasim_hdgldg_preprocess` executable. An older installation
must be rebuilt/reinstalled to include the `hdgldgsolve` helpers. Set
`EXASIM_PREFIX` to your installation when using a custom prefix.

Choose a directory, set up the corresponding installed frontend as usual,
and run its script:

```matlab
pdeapp
```

```sh
python pdeapp.py
julia pdeapp.jl
```

MATLAB requires its Symbolic Math Toolbox; Python requires NumPy and SymPy;
Julia requires Exasim and its usual dependencies. CMake, a C++ compiler and
MPI must be available. If the installed Julia project is not already active,
use `julia --project="$EXASIM_PREFIX/share/exasim/julia/Exasim" pdeapp.jl`.
For MATLAB, run `share/exasim/matlab/exasim_setup.m` first if needed.

Each script visibly defines the physics, solver settings, mesh and HDG
selection, then calls `hdgldgsolve`. It generates native model kernels,
prepares H/L inputs, runs one coupled MPI solve, and returns all solution
components in the original full-mesh element order. The native scripts do
not invoke the Text2Code compiler. Output uses Exasim's `q=-grad(u)` convention.

The returned mesh also contains `mesh.hdgldg` (MATLAB/Julia), or
`mesh["hdgldg"]` (Python): one entry per original mesh element, `1` for HDG
and `0` for LDG. This is the accepted preprocessor selection, including for
boundary-layer inputs, not an estimate from the coordinates. MATLAB returns
a row vector; Python and Julia return one-dimensional integer arrays.

The default is two ranks: rank 0 owns HDG elements and rank 1 owns LDG
elements. A fresh output directory is printed after each solve. Optional
`HDGLDG_RUN_DIR` selects a disposable run location for checks; it is not
required for normal use. GPU execution is not advertised by these examples.

## Choosing HDG elements

The third argument to `hdgldgsolve` can be an explicit array with one label
per original mesh element (`1` for HDG, `0` for LDG), or a boundary-layer
selection. For example, two layers from boundary-condition code 1:

```matlab
selection = struct('seed_boundaries',1,'layers',2);
[sol,pde,mesh] = hdgldgsolve(pde,mesh,selection,1);
```

```python
selection = {"seed_boundaries": [1], "layers": 2}
sol, pde, mesh = exasim.hdgldgsolve(pde, mesh, selection, nhdg=1)
```

```julia
selection = (seed_boundaries=[1], layers=2)
sol,pde,mesh = Exasim.hdgldgsolve(pde,mesh,selection,1)
```

The first layer contains elements touching the selected physical BC codes;
later layers expand through face neighbours. These are **BC codes**, not
indices in `boundaryexpr`. Selection and partitioning use the shared C++
preprocessor, including periodic connectivity. Both H and L regions must
remain nonempty and large enough for their rank groups.

## Text2Code

Each case has its own `CMakeLists.txt`, `main.cpp`, `pdeapp.txt`,
`pdemodel.txt`, `grid.bin` and `selection.txt`. From that case directory:

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

The supplied `grid.bin` and `selection.txt` form a matching pair. Update both
if you change a labelled mesh. Boundary-layer manifests select from the
new mesh automatically. Use a new absolute work directory for each solve.

The `common` directory contains optional refinement, measurement and parity
campaign tools. It is not needed to run any of the five examples.
