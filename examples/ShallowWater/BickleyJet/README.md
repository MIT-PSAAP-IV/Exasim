# Shallow-water Bickley jet

This transient two-dimensional shallow-water example perturbs a Bickley jet
with a vortical disturbance. It evolves fluid height and two momentum
components on a periodic square, using polynomial order four and four MPI ranks
by default.

MATLAB, Python, and Julia application/model files are included; additional
model variants in this directory support related experiments.

## Run

```bash
export EXASIM_ROOT=/path/to/Exasim
cd "$EXASIM_ROOT/examples/ShallowWater/BickleyJet"
matlab -batch "pdeapp"
PYTHONPATH="$EXASIM_ROOT/frontends/Python" python3 pdeapp.py
julia --project="$EXASIM_ROOT/frontends/Julia/Exasim" pdeapp.jl
```

The default run has 5,000 small time steps and is not a quick smoke test. Run
frontends separately and compare conserved variables at common saved times.
