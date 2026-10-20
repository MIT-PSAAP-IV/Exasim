# Axisymmetric ISOQ Poisson problem

This case solves the radius-weighted axisymmetric Poisson equation on the
curved ISOQ geometry. Python and Julia reuse the reviewed ISOQ mesh helper from
`MeshAdaptivity/isoq2d_idealgas` and preserve the four MATLAB boundary types.

```bash
matlab -batch "run('pdeapp.m')"
python3 pdeapp.py
julia --project=../../../frontends/Julia/Exasim pdeapp.jl
```
