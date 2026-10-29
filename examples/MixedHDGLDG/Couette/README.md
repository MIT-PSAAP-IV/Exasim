# Compressible planar Couette flow

Steady compressible Navier–Stokes between a fixed lower wall (y=0) and an
upper wall moving with speed U (y=1), with constant viscosity 1/Re and
viscous heating. There is no named Couette donor on master, so this is a
thin exact-field adaptation of the existing Exasim compressible NS flux
(`examples/NavierStokes/naca2d`) on the existing structured unit-square mesh.

`physicsparam = [gamma, Re, Pr, Mach, U, T0, T1] = [1.4, 10, 0.72, 0.8, 1, 1, 1.1]`
and `tau = 3`. The exact state is u = U y, v = 0, p = 1/(gamma M^2),
rho = gamma M^2 p / T with

    T(y) = T0 + (T1 - T0) y + (gamma - 1) M^2 Pr U^2 y (1 - y) / 2,

which balances conduction against viscous heating
(T'' = -(gamma - 1) M^2 Pr U^2). `common/cases.py` checks symbolically that
the steady NS residual of this state is exactly zero. Every side uses the
exact conservative state as Dirichlet data: HDG imposes `uhat = u_exact`,
LDG uses `u_b = u_exact` with flux `F(u_b, q).n + tau (u - u_b)`. The run
starts from the uniform state rho = 1, u = U/2, v = 0, p = 1/(gamma M^2).
The H region is the fixed half x <= 1/2 (use even N), so the HL interface
crosses the shear and heating layers.

`compare.py measure` also reports signed wall diagnostics with the fluid's
outward normal n: the mean fluid stress tau_xy, the wall work on the fluid
`int u.(tau.n)`, the conductive heat into the fluid `int k dT/dn`, their sum
(the physical energy balance), and separately the stabilized native energy
flux. Exact values: tau_xy = U/Re, upper-wall work U^2/Re, lower-wall heat
-k(T1 - T0 + a), upper-wall heat k(T1 - T0 - a), a = (gamma-1) M^2 Pr U^2/2,
k = 1/((gamma-1) M^2 Re Pr).

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
