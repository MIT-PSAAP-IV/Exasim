# Nonlinear cube block

This three-dimensional compressible neo-Hookean benchmark stretches a unit
cube with prescribed end traction. It uses a 16-by-4-by-4 hexahedral mesh,
cubic approximation, and two MPI ranks in all frontends.

```bash
matlab -batch "run('pdeapp.m')"
python3 pdeapp.py
julia --project=../../../frontends/Julia/Exasim pdeapp.jl
```
