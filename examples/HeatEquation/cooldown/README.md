# Transient cooling problem

This two-dimensional heat equation example follows the cooling of a prescribed
initial temperature field on an 8-by-8 unit-square mesh. It uses polynomial
order three, unit conductivity, 20 time steps of size `0.025`, and defaults to
four MPI ranks.

Equivalent MATLAB, Python, and Julia application/model files are included.

## Run

```bash
export EXASIM_ROOT=/path/to/Exasim
cd "$EXASIM_ROOT/examples/HeatEquation/cooldown"
matlab -batch "pdeapp"
PYTHONPATH="$EXASIM_ROOT/frontends/Python" python3 pdeapp.py
julia --project="$EXASIM_ROOT/frontends/Julia/Exasim" pdeapp.jl
```

Run one frontend at a time. Compare temperature at identical saved times and
use the same MPI rank count for reproducibility.
