# Equilibrium-air Mach-8 cylinder adaptation

This hypersonic cylinder example replaces the ideal-gas closure with the
equilibrium five-species-air material database while retaining artificial
viscosity and mesh adaptation. It uses a curved 51-by-32 quadrilateral mesh,
polynomial order two, and four MPI ranks.

Equivalent MATLAB, Python, and Julia application/model files are included. The
material database is
`apps/materialdatabases/equilibriumAir5logdensityexasim.dat`.

## Run

```bash
export EXASIM_ROOT=/path/to/Exasim
cd "$EXASIM_ROOT/examples/MeshAdaptivity/cylindermach8_equichem"
matlab -batch "pdeapp"
PYTHONPATH="$EXASIM_ROOT/frontends/Python" python3 pdeapp.py
julia --project="$EXASIM_ROOT/frontends/Julia/Exasim" pdeapp.jl
```

This is a comparatively expensive MPI case. Run frontends separately and
compare the adapted coordinates, conservative state, artificial viscosity,
thermodynamic fields, and species mass fractions.
