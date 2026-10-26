# Four-domain non-matching Poisson chain (3D hexes)

This example divides the unit cube into four physical domains and couples adjacent
domains through three non-matching interfaces at `x=0.25`, `x=0.50`, and
`x=0.75`. All four domains solve the same Poisson equation with manufactured
solution `sin(pi*x)*sin(pi*y)*sin(pi*z)`.

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

Generated data and builds are written under the language-specific run
directory in the system temporary area. The interactive defaults are
`porder=1`, `N=2`, and one MPI rank per domain. For a reproducible run, call
the MATLAB entry point with an absolute output directory outside the
repository:

```matlab
run_multiple_interactions(2,2,[1 1 1 1],'/tmp/exasim-multiple-3d')
```
