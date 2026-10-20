# Mach-8.98 cylinder

This legacy viscous Euler/Navier--Stokes example resolves hypersonic flow over
a circular cylinder with a curved 100-by-100 quadrilateral mesh and a
dilatation-based artificial bulk viscosity. The wall is isothermal; the outer
boundaries use supersonic inflow/outflow data.

MATLAB, Python, and Julia define the same polynomial order, physical
parameters, boundary labels, curved mesh, and initial wall layer. The scripts
generate and compile the application but do not launch the 10,000-step run.
The default is NVIDIA GPU execution, matching the original MATLAB case.

```bash
matlab -batch "run('pdeapp.m')"
python3 pdeapp.py
julia --project=../../../frontends/Julia/Exasim pdeapp.jl
```

This is an expensive production-scale setup. A CUDA-enabled Exasim install is
required unless `platform` is changed consistently in the selected driver.
