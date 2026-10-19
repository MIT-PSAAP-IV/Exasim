# Nonlinear heat-flux Poisson problem

This example solves diffusion in the transformed state
`w(u)=log(1+u^2)`. The auxiliary equation-of-state variable and the analytic
chain-rule flux are represented consistently in MATLAB, Python, and Julia.

```bash
matlab -batch "run('pdeapp.m')"
python3 pdeapp.py
julia --project=../../../frontends/Julia/Exasim pdeapp.jl
```
