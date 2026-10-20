# Plate with a Gaussian lower boundary

This two-dimensional linear-elasticity example imposes a Gaussian vertical
displacement along the lower surface of a straight plate. The Python and Julia
mesh helpers reproduce MATLAB's structured quadrilateral mesh with refinement
toward that boundary. The remaining boundaries use the model's traction
condition.

```bash
matlab -batch "run('pdeapp.m')"
python3 pdeapp.py
julia --project=../../../frontends/Julia/Exasim pdeapp.jl
```
