# Orion axisymmetric Poisson problem

This example solves the axisymmetric scalar diffusion problem on the tracked
Orion geometry. The Python and Julia drivers mesh `domain6.geo` with Gmsh and
use the same four boundary-condition classes and solver parameters as MATLAB.
Gmsh must be available on `PATH`.

Run `pdeapp.m`, `python3 pdeapp.py`, or `julia pdeapp.jl` from this directory.
