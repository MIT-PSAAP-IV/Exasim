# Nonlinear stretching block

A unit square made of compressible neo-Hookean material is fixed on one side
and stretched by end traction. All three frontends use a 16-by-16 quadrilateral
mesh, cubic approximation, and the same HDG solver settings.

```bash
matlab -batch "run('pdeapp.m')"
python3 pdeapp.py
julia --project=../../../frontends/Julia/Exasim pdeapp.jl
```
