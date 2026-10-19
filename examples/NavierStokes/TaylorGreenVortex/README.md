# Taylor--Green vortex

The MATLAB, Python, and Julia frontends define the same periodic compressible
Navier--Stokes problem on `[0,2*pi]^3`: polynomial degree two, a `24^3` hex
mesh, `Re=1600`, `Pr=0.71`, `M=0.2`, and third-order time stepping.

Run `python3 pdeapp.py`, `julia pdeapp.jl`, or the MATLAB `pdeapp.m`. The case
uses eight MPI ranks and is intentionally a substantial three-dimensional
solve. `pdemodel.py` and `pdemodel.jl` are native translations of the MATLAB
inviscid, viscous, initial-condition, and periodic-boundary definitions.
