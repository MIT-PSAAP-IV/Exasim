# One-dimensional heat conduction

This transient GSI example solves heat conduction through a one-dimensional
layer with temperature-dependent physical scaling and mixed boundary data.
All frontends use 300 elements and degree-three approximation.

```bash
matlab -batch "run('pdeapp.m')"
python3 pdeapp.py
julia --project=../../../frontends/Julia/Exasim pdeapp.jl
```
