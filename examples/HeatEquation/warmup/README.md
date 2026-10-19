# Transient warming problem

This two-dimensional heat equation example evolves a temperature field under
the configured heating boundary data. It uses an 8-by-8 unit-square mesh,
polynomial order three, unit conductivity, and 20 time steps of size `0.1`.

The case has equivalent MATLAB, Python, and Julia `pdeapp` and `pdemodel` files.

## Run

```bash
export EXASIM_ROOT=/path/to/Exasim
cd "$EXASIM_ROOT/examples/HeatEquation/warmup"
matlab -batch "pdeapp"
PYTHONPATH="$EXASIM_ROOT/frontends/Python" python3 pdeapp.py
julia --project="$EXASIM_ROOT/frontends/Julia/Exasim" pdeapp.jl
```

Run one frontend at a time. The principal output is the transient scalar
temperature field; compare solutions at the same physical time.
