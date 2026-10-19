# Mapped Euler vortex

This example transports an isentropic Euler vortex on a time-dependent mapped
quadrilateral mesh. It exercises Exasim's ALE metric, grid-velocity, periodic
boundary, error-functional, and visualization-vector interfaces.

The MATLAB, Python, and Julia drivers prepare the same 64-by-48, `p = 2`
problem and generate its solver sources; they intentionally do not launch the
16-rank transient solve.

```bash
matlab -batch "run('pdeapp.m')"
python3 pdeapp.py
julia --project=../../../frontends/Julia/Exasim pdeapp.jl
```

Run each command from this directory after configuring the corresponding
frontend. The generated run uses periodic boundaries in both directions.
