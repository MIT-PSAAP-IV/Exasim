# Inviscid flow over a NACA 0012 airfoil

This transient two-dimensional Euler case computes ideal-gas flow around a
curved NACA 0012 airfoil. It reads the supplied mesh, marks the curved airfoil
boundary, uses polynomial order two, and defaults to one MPI rank.

MATLAB, Python, and Julia application/model pairs are provided.

## Run

```bash
export EXASIM_ROOT=/path/to/Exasim
cd "$EXASIM_ROOT/examples/Euler/naca0012"
matlab -batch "pdeapp"
PYTHONPATH="$EXASIM_ROOT/frontends/Python" python3 pdeapp.py
julia --project="$EXASIM_ROOT/frontends/Julia/Exasim" pdeapp.jl
```

Run from this directory so relative mesh files resolve. Compare conservative
variables and derived pressure/Mach fields when checking frontend consistency.
