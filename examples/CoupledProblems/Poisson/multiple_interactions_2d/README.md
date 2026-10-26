# Four-domain non-matching Poisson chain (2D quads)

This example divides the unit square into four physical domains and couples adjacent
domains through three non-matching interfaces at `x=0.25`, `x=0.50`, and
`x=0.75`. All four domains solve the same Poisson equation with manufactured
solution `sin(pi*x)*sin(pi*y)`.

The files follow the same organization as the two-domain `Poisson/matching`
example:

- `pdeapp.m` contains the user controls, loads the four domains, calls Exasim,
  and reports the error.
- `pdeapp1.m` through `pdeapp4.m` describe the four physical domains.
- `interaction.m` describes the three edges of the interaction graph.
- `pdemodel.m` defines the common Poisson model and its two interface
  conditions.

In `interaction.m`, condition 1 imposes solution continuity and condition 2
imposes normal-flux balance. Adjacent domains use different tangential mesh
resolutions, so the three interfaces are non-matching.

The first physical parameter is the unit diffusion coefficient. The second
selects the local HDG interface equation while retaining one shared model:
domains 1 and 3 use the numerical-flux equation, and domains 2 and 4 use the
trace equation.

## Run

Start MATLAB in this directory and run:

```matlab
pdeapp
```

Python:

```bash
python3 pdeapp.py
```

Julia:

```bash
julia --project=<path-to-Exasim>/frontends/Julia/Exasim pdeapp.jl
```

Generated data and builds are written under the language-specific run
directory in the system temporary area. The interactive defaults are
`porder=1`, `N=2`, and one MPI rank per domain in every frontend.

For reproducible runs, use the callable entry points and provide an absolute
output directory outside the repository:

```matlab
run_multiple_interactions(2,2,[1 1 1 1],'/tmp/exasim-multiple-2d')
```

```bash
python3 run_multiple_interactions.py --porder 2 --N 2 \
  --ranks 1,1,1,1 --outdir /tmp/exasim-multiple-2d-python
julia --project=<path-to-Exasim>/frontends/Julia/Exasim \
  run_multiple_interactions.jl 2 2 1,1,1,1 /tmp/exasim-multiple-2d-julia
```
