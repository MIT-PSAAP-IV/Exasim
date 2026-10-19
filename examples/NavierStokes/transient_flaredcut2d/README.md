# Transient flared-cut turbulent inflow

The transient case transfers a prior flared-plate solution and artificial
viscosity field before constructing synthetic-turbulence data. The portable
entry points therefore preserve the canonical MATLAB execution instead of
inventing replacement initialization data.

Run `python3 pdeapp.py` or `julia pdeapp.jl` after preparing `sol1` and `av1`
as described in `pdeapp.m`. MATLAB is required.
