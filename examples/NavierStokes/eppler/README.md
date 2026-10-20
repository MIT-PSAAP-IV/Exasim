# Eppler 387 airfoil

This case solves the unsteady low-Mach compressible Navier--Stokes equations
around the Eppler 387 airfoil. Its body-fitted high-order mesh is constructed
by MATLAB-specific geometry utilities from `epp387_smoothed`.

`pdeapp.py` and `pdeapp.jl` run the canonical MATLAB workflow in isolated run
directories so the mesh and boundary ordering remain identical. MATLAB is
required; no approximate replacement mesh is generated.
