# One-dimensional concentration diffusion

This transient GSI example solves diffusion and first-order reaction of a
single concentration through a one-dimensional layer. It uses 300 elements,
degree-three approximation, a prescribed concentration at the left boundary,
and a natural condition at the right boundary.

```bash
matlab -batch "run('pdeapp.m')"
python3 pdeapp.py
julia --project=../../../frontends/Julia/Exasim pdeapp.jl
```
