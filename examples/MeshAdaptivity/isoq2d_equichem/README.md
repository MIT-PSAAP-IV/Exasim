# Equilibrium-air ISOQ mesh adaptation

This axisymmetric hypersonic ISOQ case combines equilibrium five-species-air
thermodynamics, artificial-viscosity continuation, wall-distance evaluation,
and backend mesh adaptation. It uses polynomial order two and four MPI ranks.

The MATLAB, Python, and Julia applications share the mesh construction and
material database. Their models expose density, pressure, temperature, Mach
number, artificial viscosity, velocity, and species fractions.

## Run

```bash
export EXASIM_ROOT=/path/to/Exasim
cd "$EXASIM_ROOT/examples/MeshAdaptivity/isoq2d_equichem"
matlab -batch "pdeapp"
PYTHONPATH="$EXASIM_ROOT/frontends/Python" python3 pdeapp.py
julia --project="$EXASIM_ROOT/frontends/Julia/Exasim" pdeapp.jl
```

Run one frontend at a time. Verify both the flow solution and final adapted
mesh; matching residual histories alone do not establish mesh equivalence.
