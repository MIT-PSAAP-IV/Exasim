# Three-dimensional Poisson equation

This steady HDG example solves Poisson's equation on an 8-by-8-by-8 hexahedral
unit-cube mesh. It uses polynomial order two, unit conductivity, homogeneous
boundary data, and one MPI rank by default.

MATLAB, Python, and Julia application/model pairs are included.

## Run

```bash
export EXASIM_ROOT=/path/to/Exasim
cd "$EXASIM_ROOT/examples/Poisson/poisson3d"
matlab -batch "pdeapp"
PYTHONPATH="$EXASIM_ROOT/frontends/Python" python3 pdeapp.py
julia --project="$EXASIM_ROOT/frontends/Julia/Exasim" pdeapp.jl
```

This case is larger than the two-dimensional tutorial. Run frontends
separately and compare mesh dimensions and node ordering before solution norms.
