# Axisymmetric Sharp-B Poisson problem

This example solves an axisymmetric Poisson problem on the curved Sharp-B
geometry. The Python and Julia drivers reuse the established Sharp-B mesh
helper and retain the MATLAB boundary numbering and solver parameters.

```bash
matlab -batch "run('pdeapp.m')"
python3 pdeapp.py
julia --project=../../../frontends/Julia/Exasim pdeapp.jl
```
