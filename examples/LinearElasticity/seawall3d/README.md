# Three-dimensional seawall

This example extrudes and maps the seawall cross-section into a curved
three-dimensional hexahedral mesh, then solves linear elasticity with degree
two on eight MPI ranks. The local Python and Julia helpers reproduce the
cross-section extrusion and coordinate map used by MATLAB.

```bash
matlab -batch "run('pdeapp.m')"
python3 pdeapp.py
julia --project=../../../frontends/Julia/Exasim pdeapp.jl
```

The default run requires an MPI-enabled CPU Exasim installation with at least
eight ranks available.
