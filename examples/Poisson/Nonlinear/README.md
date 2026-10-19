# Nonlinear Poisson problem

This steady two-dimensional example exercises Exasim's nonlinear residual and
Jacobian generation for a scalar Poisson-type equation. It uses polynomial
order three on an 8-by-8 unit-square mesh with mixed boundary conditions.

MATLAB, Python, and Julia application/model pairs are included.

## Run

```bash
export EXASIM_ROOT=/path/to/Exasim
cd "$EXASIM_ROOT/examples/Poisson/Nonlinear"
matlab -batch "pdeapp"
PYTHONPATH="$EXASIM_ROOT/frontends/Python" python3 pdeapp.py
julia --project="$EXASIM_ROOT/frontends/Julia/Exasim" pdeapp.jl
```

Run frontends separately. Compare the complete nonlinear residual history as
well as the converged scalar and gradient fields.
