# Coupled heat and concentration in one dimension

This example couples temperature and concentration diffusion/reaction in a
one-dimensional layer. The three frontend drivers preserve the MATLAB
nondimensionalization, nonuniform time-step sequence, 300-element mesh, and
boundary conditions.

```bash
matlab -batch "run('pdeapp.m')"
python3 pdeapp.py
julia --project=../../../frontends/Julia/Exasim pdeapp.jl
```
