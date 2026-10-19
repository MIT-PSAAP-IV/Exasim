# HCA flared-plate 3D Poisson problem

This example solves a scalar Laplace problem on a short extrusion of
an HCA-like flared plate. Python and Julia use the checked-in `hca_flare.geo`
mesh because the legacy MATLAB script depends on an untracked
`meshparams.mat` file and MATLAB-only extrusion utilities. The replacement
retains the three PDE boundary operators and partitions all eight exterior
surfaces of the replacement geometry, but is not node-identical to the
historical MATLAB mesh. Gmsh is required.

The MATLAB case pairs the two extrusion planes periodically. The Python
frontend's current 3D periodic-face preprocessing does not support hexahedral
faces, so the Python and Julia ports assign the same zero-normal-flux condition
to both planes. For this extrusion-invariant Poisson problem that retains the
intended two-dimensional cross-sectional solution.

Run `pdeapp_poi.m`, `python3 pdeapp.py`, or `julia pdeapp.jl` from this
directory.
