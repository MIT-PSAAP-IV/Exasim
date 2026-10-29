# Steady laminar NACA0012 flow

This reuses the donor `examples/NavierStokes/naca2d` mesh (`grid.bin`),
controls and physics (Re = 100, M = 0.2, Pr = 0.72, alpha = 0, p = 4,
`tau = 3`, freestream initial state) without modifying it. Boundary 1 is
the no-slip adiabatic airfoil wall and boundary 2 the characteristic far
field. `pdemodel.m` is the donor MATLAB model; `pdemodel.py` and
`pdemodel.jl` transcribe it. The donor `pdemodel.txt` leaves the LDG
callbacks `Ubou`/`Fbou` as zero placeholders, so this example's Text2Code
model keeps the donor HDG callbacks and adds the donor MATLAB LDG wall and
far-field callbacks. The original pure-LDG donor failure is a separate,
retained baseline and is not changed by this example.

All schemes run on two ranks. Mixed uses one H rank for the single layer of
elements touching the airfoil (`selection = boundary_layers`,
`seed_boundaries = 1`, `layers = 1`) and one L rank for the rest. Each
scheme uses preconditioner option 1: native H Schwarz, native L
block-Jacobi, and augmented H Schwarz plus L block-Jacobi. The Newton,
GMRES (400 iterations, restart 400, tolerance 1e-7) and matvec FD controls
are the donor's for every scheme.

## Run as an ordinary example

Use a current Exasim installation with CPU/MPI preprocessing and the chosen
frontend enabled. Set `EXASIM_PREFIX` for a custom installation. The normal
installation supplies `bin/exasim_hdgldg_preprocess`; no test-project build
or private development directory is required. The usual MATLAB Symbolic
Math Toolbox, Python NumPy/SymPy, or Julia Exasim/SymPy dependencies apply.
GPU validation is not claimed.

Change to this directory and run one of:

```matlab
% First run the installed share/exasim/matlab/exasim_setup.m if needed.
pdeapp
```

```sh
python pdeapp.py
julia --project="$EXASIM_PREFIX/share/exasim/julia/Exasim" pdeapp.jl
```

The scripts visibly define the model, solver controls, mesh and selection,
then call `hdgldgsolve`. No arguments are required. MATLAB leaves `sol`,
`pde` and `mesh` in the workspace. Python and Julia print the result shape.
Every route uses native symbolic model generation and the installed coupled
solver. The returned `sol` is in the original `mesh.t` element order, with
12 components: four conservative variables followed by their x/y auxiliary
components, using the existing `q=-grad(u)` convention.

`grid.bin` is a byte-identical copy of the small donor mesh (256 quadrilateral
elements). Keeping it in this directory makes the example self-contained.
The boundary-layer selection is computed by shared CPU preprocessing:

```matlab
selection = struct('seed_boundaries',1,'layers',1);
[sol,pde,mesh] = hdgldgsolve(pde,mesh,selection,1);
```

Code 1 means the wall **boundary-condition code**, not the position of a
boundary expression. `layers=1` selects elements touching the wall;
`layers=2` adds their face neighbours. The last argument reserves the first
world rank for HDG; the other rank uses LDG. Do not change the physics or
solver tolerances to suppress reported convergence failures or linear-limit
warnings. This nonlinear benchmark has no exact solution.
At the supplied settings the mixed example reaches its Newton tolerance in
four updates, while the last three linear solves reach the existing
400-iteration limit. Those warnings remain visible; this workflow change
does not claim that every linear solve satisfies its GMRES tolerance.

The helper prints the generated input/output directory. `HDGLDG_RUN_DIR`
optionally selects a disposable working directory for automated checks.

## Independent Text2Code workflow

The supplied `pdeapp.txt`, `pdemodel.txt`, `grid.bin` and `selection.txt`
define the same case without a native frontend. From this directory:

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

The ordinary installed package supplies Text2Code and its CMake model helper.
The standalone build generates kernels but not datain; the shared installed
preprocessor creates the H/L inputs. Another new absolute work directory may
be chosen for persistent output. Existing output is not overwritten.

The scripts in `../common` remain optional development campaigns; they are
not prerequisites for running any of these user entrypoints.
