# Density-equalizing square mesh

This example computes a mesh map on `[-1,1]^2`. A Poisson solve produces the
velocity potential for a ring-shaped monitor function; two transient transport
solves then move the `x` and `y` coordinates. The Python and Julia ports retain
the staged algorithm used by `pdeapp.m` and write their generated data beneath
frontend-specific run directories.

Run `pdeapp.m`, `python3 pdeapp.py`, or `julia pdeapp.jl` from this directory.
The Python result is stored in `python_run/adapted_mesh.npz`; the Julia result is
stored in `julia_run/adapted_mesh.bin`.
