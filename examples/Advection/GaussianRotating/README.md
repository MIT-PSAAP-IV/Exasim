# Rotating Gaussian advection

This transient two-dimensional example transports a Gaussian pulse in the
rotational velocity field `(y,-x)`. It uses a fourth-order discretization on a
10-by-10 unit-square mesh and defaults to four MPI ranks.

Equivalent application and model files are provided for MATLAB (`pdeapp.m`,
`pdemodel.m`), Python (`pdeapp.py`, `pdemodel.py`), and Julia (`pdeapp.jl`,
`pdemodel.jl`).

## Run

From the repository root, set `EXASIM_ROOT=$PWD`, enter this directory, and run
one frontend at a time:

```bash
matlab -batch "pdeapp"
PYTHONPATH="$EXASIM_ROOT/frontends/Python" python3 pdeapp.py
julia --project="$EXASIM_ROOT/frontends/Julia/Exasim" pdeapp.jl
```

The frontends use the same mesh, initial pulse, time-step sequence, and
stabilization. They write build and solution products below the case directory,
so remove or isolate previous outputs before comparing frontends.
