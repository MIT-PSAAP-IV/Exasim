# Steady compressible Navier–Stokes manufactured solution

There is no clean-master steady NS MMS donor, so this is a thin exact-field
and source adaptation of the existing Exasim compressible NS flux
(`examples/NavierStokes/naca2d`) on the structured unit-square mesh.
`physicsparam = [gamma, Re, Pr, Mach] = [1.4, 10, 0.72, 0.5]`, `tau = 3`,
constant viscosity 1/Re.

The exact primitive state is

    rho = 1 + 0.1 sin(pi x) cos(pi y),   u = 0.5 + 0.1 cos(pi x) sin(pi y),
    v = 0.2 + 0.1 sin(pi x) sin(pi y),   p = (1 + 0.1 cos(pi x) cos(pi y))/(gamma M^2),

so rho and p stay positive. The steady source is S = div F(u_e, -grad u_e).
Each route differentiates it symbolically from the same exact state:
SymPy in Python, the Symbolic Math Toolbox in MATLAB, SymPy.jl in Julia
and SymEngine `diff` in the Text2Code model (Text2Code keeps only
output-vector assignments inside loops, so its model uses helper
functions rather than local vectors). `common/cases.py` derives the source
independently for measurement. All sides use exact Dirichlet data
(HDG `uhat = u_exact`; LDG `F(u_b, q).n + tau (u - u_b)`). The run starts
from the uniform state rho = 1, u = 0.5, v = 0.2, p = 1/(gamma M^2). The H
region is the fixed half x <= 1/2.

## Run directly

Use the ordinary installed Exasim CPU/MPI frontend setup, as for
`examples/Poisson/poisson2d`. The installation must include the current
`hdgldgsolve` helper, coupled runtime and `bin/exasim_hdgldg_preprocess`.
MATLAB needs Symbolic Math Toolbox; Python needs NumPy/SymPy; Julia needs
the Exasim package and SymPy.jl. No test-project build is required.

Change to this example directory and choose one native frontend:

- MATLAB: run `pdeapp` (no arguments).
- Python: run `python pdeapp.py`.
- Julia: run `julia pdeapp.jl`, or
  `julia --project="$EXASIM_PREFIX/share/exasim/julia/Exasim" pdeapp.jl`.

For MATLAB, first run your installed `share/exasim/matlab/exasim_setup.m`
if the frontend is not already on the path. A complete checkout also has
the usual source setup fallback. Set `EXASIM_PREFIX` for a custom install.

Each script visibly defines the model parameters, solver controls and a
4x4 unit-square mesh at p=2. Elements on x<=1/2 use HDG (label 1);
the remainder use LDG (label 0). One MPI rank owns each scheme.
Edit these settings directly, keeping N even and the interface face aligned.
The native helper generates kernels, compiles, preprocesses and solves.
The returned `sol` has shape [9,12,16] in original mesh element order,
with components [rho,rho*u,rho*v,rho*E,qx(1:4),qy(1:4)] and q=-grad(u).
MATLAB leaves sol, pde and mesh in the workspace and plots density.

Each run creates a fresh directory beneath pde.datapath and prints its
location; existing runs are not overwritten. `HDGLDG_RUN_DIR` optionally
selects a new disposable directory for automated checks. Pure schemes and
numerical tolerances are unchanged. This example advertises CPU support
only; it does not establish GPU validation.

## Independent Text2Code path

The supplied `grid.bin` is the same 4x4 mesh; `selection.txt` labels its
left half HDG. The independent model is `pdemodel.txt`. From this directory:

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

This uses the normal installed Text2Code external-model helper and shared
preprocessor, not a private build or Python campaign launcher. If replacing
the mesh, update its labels to match the new element order.
The optional `../common` scripts remain development campaigns, not
prerequisites for human users.
