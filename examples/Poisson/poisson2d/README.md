# Two-dimensional Poisson equation

This is the basic steady HDG Poisson example on an 8-by-8 unit-square mesh. It
uses polynomial order two, unit conductivity, homogeneous boundary data, and
one MPI rank by default. Auxiliary drivers demonstrate parameter sweeps and
Text2Code export.

Equivalent MATLAB, Python, and Julia application/model files are provided.

## Run

```bash
export EXASIM_ROOT=/path/to/Exasim
cd "$EXASIM_ROOT/examples/Poisson/poisson2d"
matlab -batch "pdeapp"
PYTHONPATH="$EXASIM_ROOT/frontends/Python" python3 pdeapp.py
julia --project="$EXASIM_ROOT/frontends/Julia/Exasim" pdeapp.jl
```

Run one frontend at a time. For a frontend comparison, check the mesh and
configuration binaries before comparing the scalar and gradient fields.
