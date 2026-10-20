# Periodic Poisson problem

This steady two-dimensional Poisson example demonstrates periodic boundary
matching on an 8-by-8 unit-square mesh. The right and left boundaries are
paired by their transverse coordinate. The case uses polynomial order three,
unit conductivity, and one MPI rank by default.

Equivalent MATLAB, Python, and Julia application/model files are included.

## Run

```bash
export EXASIM_ROOT=/path/to/Exasim
cd "$EXASIM_ROOT/examples/Poisson/Periodic"
matlab -batch "pdeapp"
PYTHONPATH="$EXASIM_ROOT/frontends/Python" python3 pdeapp.py
julia --project="$EXASIM_ROOT/frontends/Julia/Exasim" pdeapp.jl
```

Run one frontend at a time. Verify the periodic face pairing and node ordering
before comparing scalar and gradient solutions.
