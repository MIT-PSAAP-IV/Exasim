# One-Dimensional Convection-Diffusion

This example solves the same steady one-dimensional convection-diffusion
problem through the MATLAB, Python, and Julia frontends. It uses 16 elements,
polynomial degree 3, homogeneous boundary data, diffusion coefficient 1, and
advection speed `2*pi`.

Run one frontend from this directory:

```bash
matlab -batch "pdeapp"
python3 pdeapp.py
julia --project=../../../frontends/Julia/Exasim pdeapp.jl
```

Configure the frontend as described in the Exasim installation documentation
before running the example.
