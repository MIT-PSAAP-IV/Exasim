# Poisson equation with an algebraic state

The conductivity depends on an auxiliary material state `w`, constrained by
`exp(w) - 1 - u^2 = 0`. This example exercises Exasim's local equation-of-state
solve together with nonlinear HDG diffusion.

```bash
matlab -batch "run('pdeapp.m')"
python3 pdeapp.py
julia --project=../../../frontends/Julia/Exasim pdeapp.jl
```
