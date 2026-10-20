# Supersonic Taylor-Green Vortex

This production-scale three-dimensional Navier-Stokes case initializes the
Mach 1.25 Taylor-Green vortex of Lusher and Sandham (AIAA Journal, 2020). The
MATLAB and Julia drivers prepare a periodic `128^3` hexahedral mesh for 456 MPI
ranks. The Python driver exports the corresponding Text2Code application.

The case is intentionally configured for preprocessing/export rather than a
local workstation solve. Its artificial-viscosity model and volume quantities
are implemented consistently in `pdemodel.m`, `pdemodel.py`, and
`pdemodel.jl`.

Run the desired export/preprocessing frontend from this directory after
configuring Exasim:

```bash
matlab -batch "pdeapp"
python3 pdeapp_text2code.py
julia --project=../../../frontends/Julia/Exasim pdeapp.jl
```
