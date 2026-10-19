# Elastic plate in a fluid-shaped domain

This plane-strain example solves linear elasticity on the exterior region
surrounding a thin plate. MATLAB constructs the geometry procedurally; Python
and Julia use the equivalent local `platefluid.geo` geometry through Gmsh.
The plate boundary carries the distinguished boundary condition.

```bash
matlab -batch "run('pdeapp.m')"
python3 pdeapp.py
julia --project=../../../frontends/Julia/Exasim pdeapp.jl
```

Python and Julia therefore require `gmsh` on `PATH`.
