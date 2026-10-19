# Poisson equation on an L-shaped domain

This steady two-dimensional example solves Poisson's equation on the
non-convex L-shaped domain defined by `lshape.geo`. It uses Gmsh, polynomial
order three, homogeneous boundary data, and defaults to four MPI ranks.

Equivalent MATLAB, Python, and Julia application/model files are included.

## Run

```bash
export EXASIM_ROOT=/path/to/Exasim
cd "$EXASIM_ROOT/examples/Poisson/Lshape"
matlab -batch "pdeapp"
PYTHONPATH="$EXASIM_ROOT/frontends/Python" python3 pdeapp.py
julia --project="$EXASIM_ROOT/frontends/Julia/Exasim" pdeapp.jl
```

Gmsh is required. Run from this directory and verify that all frontends produce
the same mesh before comparing the singular-corner solution.
