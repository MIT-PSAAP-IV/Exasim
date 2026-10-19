# Rounded-nose flat plate with ideal-gas flow

This high-Mach-number Navier--Stokes example exercises backend artificial
viscosity continuation and mesh-adaptivity data on a rounded-nose flat plate.
The MATLAB, Python, and Julia drivers use the same physical parameters,
boundary-condition tags, polynomial order, and initialization. The
frontend-native mesh helpers reproduce the six physical boundary groups; the
Python and Julia grid uses a uniform logical grid rather than MATLAB's legacy
`lesmesh2d_rect` stretching.

Run `pdeapp.m`, `python3 pdeapp.py`, or `julia pdeapp.jl` from this directory.
Each frontend writes generated data and solver output to its own run directory.
