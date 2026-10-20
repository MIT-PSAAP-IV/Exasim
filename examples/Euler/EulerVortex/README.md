# Isentropic Euler vortex

This transient two-dimensional compressible Euler example evolves an
isentropic vortex on a periodic domain. It uses a fourth-order discretization,
an explicit time-step sequence, and the ideal-gas conservative variables.

Equivalent MATLAB, Python, and Julia application/model files are included.

## Run

```bash
export EXASIM_ROOT=/path/to/Exasim
cd "$EXASIM_ROOT/examples/Euler/EulerVortex"
matlab -batch "pdeapp"
PYTHONPATH="$EXASIM_ROOT/frontends/Python" python3 pdeapp.py
julia --project="$EXASIM_ROOT/frontends/Julia/Exasim" pdeapp.jl
```

Run one frontend at a time. Useful comparisons include conservation, vortex
translation, and the density and total-energy fields at common output times.
