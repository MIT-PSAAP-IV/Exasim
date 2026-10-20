# Axisymmetric Poisson problem

This steady two-dimensional case exercises the axisymmetric Poisson
formulation on an 8-by-8 unit-square mesh. It uses polynomial order three,
unit conductivity, and four distinct boundary-condition markers.

Equivalent MATLAB, Python, and Julia application/model files are provided.

## Run

```bash
export EXASIM_ROOT=/path/to/Exasim
cd "$EXASIM_ROOT/examples/Poisson/Axisymmetric"
matlab -batch "pdeapp"
PYTHONPATH="$EXASIM_ROOT/frontends/Python" python3 pdeapp.py
julia --project="$EXASIM_ROOT/frontends/Julia/Exasim" pdeapp.jl
```

Run one frontend at a time. Compare the scalar solution and gradient fields
using the same node ordering and axisymmetric weighting conventions.
