# Nonlinear Poisson conductivity

This unit-square example uses the nonlinear conductivity
`k(u) = 1 + 1 + u^2`, mixed Dirichlet/Neumann data, and cubic HDG elements.

```bash
matlab -batch "run('pdeapp.m')"
python3 pdeapp.py
julia --project=../../../frontends/Julia/Exasim pdeapp.jl
```
