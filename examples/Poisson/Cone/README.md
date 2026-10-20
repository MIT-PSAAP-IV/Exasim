# Poisson equation on a cone-in-cube geometry

This three-dimensional Poisson example uses Gmsh to generate the
`coneincube` mesh. It applies mixed boundary data on the outer cube and cone,
uses polynomial order two, and defaults to four MPI ranks.

MATLAB, Python, and Julia application/model pairs are provided.

## Requirements

Gmsh must be available to the selected frontend, and the run must start in this
directory so `coneincube.geo` and generated mesh files resolve correctly.

## Run

```bash
export EXASIM_ROOT=/path/to/Exasim
cd "$EXASIM_ROOT/examples/Poisson/Cone"
matlab -batch "pdeapp"
PYTHONPATH="$EXASIM_ROOT/frontends/Python" python3 pdeapp.py
julia --project="$EXASIM_ROOT/frontends/Julia/Exasim" pdeapp.jl
```

Run frontends separately and compare mesh dimensions before comparing the
scalar and gradient solutions.
