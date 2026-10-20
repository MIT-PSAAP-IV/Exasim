# Wave scattering

This transient two-dimensional wave example models scattering on a supplied
curved mesh. It uses polynomial order three, a configured wave-number vector,
100 time steps of size `0.1`, mixed boundary conditions, and four MPI ranks by
default.

MATLAB, Python, and Julia application/model pairs are included.

## Run

```bash
export EXASIM_ROOT=/path/to/Exasim
cd "$EXASIM_ROOT/examples/WaveEquation/Scattering"
matlab -batch "pdeapp"
PYTHONPATH="$EXASIM_ROOT/frontends/Python" python3 pdeapp.py
julia --project="$EXASIM_ROOT/frontends/Julia/Exasim" pdeapp.jl
```

Run from this directory so the mesh file resolves, and run frontends separately.
Compare displacement and velocity at common output times and verify identical
curved-boundary classification.
