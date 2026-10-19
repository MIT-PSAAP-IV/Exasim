# Axisymmetric ISOQ flow

The Python and Julia entry points execute the canonical axisymmetric
Navier--Stokes application in `apps/navierstokes/isoq`. The generated model
retains the cylindrical source terms, wall treatment, curved mesh, and
artificial-viscosity configuration.

Set `EXASIM_PREFIX`, then run `python3 pdeapp.py` or `julia pdeapp.jl`. Four
MPI ranks are used, matching `pdeapp.txt`.
