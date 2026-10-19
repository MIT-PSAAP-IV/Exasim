# Periodic Gaussian transport

This transient two-dimensional example advects a Gaussian pulse with constant
velocity `(1,1)` across a periodic unit square. It uses a fourth-order
discretization on a 10-by-10 mesh and defaults to one MPI rank.

MATLAB, Python, and Julia implementations are provided by the corresponding
`pdeapp` and `pdemodel` files in this directory.

## Run

From the repository root:

```bash
export EXASIM_ROOT=$PWD
cd examples/Advection/GaussianTransport
matlab -batch "pdeapp"
PYTHONPATH="$EXASIM_ROOT/frontends/Python" python3 pdeapp.py
julia --project="$EXASIM_ROOT/frontends/Julia/Exasim" pdeapp.jl
```

Run one frontend at a time because their default output and build locations may
overlap. The paired horizontal and vertical boundaries implement periodicity.
