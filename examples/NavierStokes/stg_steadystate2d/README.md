# Synthetic-turbulence-generator steady initialization

This case requires `sol1` and `av1` transferred from a prior flared-plate
solution, then constructs MATLAB STG data. Those inputs are not a standalone
portable frontend record.

`pdeapp.py` and `pdeapp.jl` invoke the canonical MATLAB workflow in isolation.
Prepare the source fields as documented in `pdeapp.m` before running either
entry point.
