# Hafnium concentration diffusion

This case applies the one-dimensional concentration diffusion/reaction model
to hafnium oxidation parameters. The MATLAB, Python, and Julia versions use
the same 300-element mesh, nondimensional coefficients, initial state, and
boundary conditions.

```bash
matlab -batch "run('pdeapp.m')"
python3 pdeapp.py
julia --project=../../../frontends/Julia/Exasim pdeapp.jl
```
