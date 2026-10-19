# Nonlinear compression block

This two-dimensional compressible neo-Hookean block combines fixed-coordinate,
traction, and top-pressure boundary segments. The Python and Julia ports retain
the MATLAB 32-by-32 quadrilateral mesh, cubic basis, and six HDG boundary
columns.

```bash
matlab -batch "run('pdeapp.m')"
python3 pdeapp.py
julia --project=../../../frontends/Julia/Exasim pdeapp.jl
```
