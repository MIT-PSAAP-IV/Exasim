# Nonlinear Cook membrane

This compressible neo-Hookean benchmark deforms Cook's tapered membrane under
a vertical end traction. The unknown is the deformed coordinate, so the
initial state is the undeformed mesh. MATLAB, Python, and Julia use the same
10-by-10 quadrilateral mesh, cubic approximation, material constants, and HDG
boundary conditions.

```bash
matlab -batch "run('pdeapp.m')"
python3 pdeapp.py
julia --project=../../../frontends/Julia/Exasim pdeapp.jl
```
