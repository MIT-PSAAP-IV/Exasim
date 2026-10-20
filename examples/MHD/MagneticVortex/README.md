# Ideal-MHD magnetic vortex

This transient two-dimensional ideal magnetohydrodynamics example evolves a
smooth magnetic vortex on a periodic square. The seven conservative variables
include density, momentum, total energy, magnetic field, and the divergence
cleaning field. The case uses polynomial order four on a 16-by-16 mesh and
defaults to four MPI ranks.

MATLAB, Python, and Julia application/model pairs are provided.

## Run

```bash
export EXASIM_ROOT=/path/to/Exasim
cd "$EXASIM_ROOT/examples/MHD/MagneticVortex"
matlab -batch "pdeapp"
PYTHONPATH="$EXASIM_ROOT/frontends/Python" python3 pdeapp.py
julia --project="$EXASIM_ROOT/frontends/Julia/Exasim" pdeapp.jl
```

Run the frontends separately. Compare all seven conservative components and
monitor periodic conservation and magnetic-field symmetry.
