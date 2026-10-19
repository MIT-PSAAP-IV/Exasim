# Turbulent-inflow flat case

This legacy case initializes an unsteady turbulent-inflow solve from the
solution and artificial-viscosity field produced by `flaredplate2d`. Those
cross-case MATLAB workspace objects are not representable as a standalone
Python or Julia frontend input. `pdeapp.py` and `pdeapp.jl` therefore run the
canonical `pdeapp2d.m` workflow in an isolated directory.

Run `python3 pdeapp.py` or `julia pdeapp.jl` after preparing the required
initial solution described at the top of `pdeapp2d.m`. MATLAB and a configured
Exasim installation are required.
