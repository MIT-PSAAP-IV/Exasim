# Variable-density square Stokes problem

The three frontends solve the same constrained two-dimensional Stokes system
on `[-1,1]^2`. Python and Julia compute the density normalization `theta` by a
deterministic fine-grid domain average before code generation; MATLAB computes
the equivalent finite-element weighted average.

Run `python3 pdeapp.py`, `julia pdeapp.jl`, or `pdeapp.m`. The small difference
between the two normalization quadratures should be considered when comparing
solutions at round-off level.
