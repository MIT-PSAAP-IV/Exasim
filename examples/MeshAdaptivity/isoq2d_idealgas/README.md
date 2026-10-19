# Ideal-gas ISOQ mesh adaptation

This axisymmetric hypersonic ISOQ example combines an ideal-gas Navier--Stokes
model with artificial-viscosity continuation, wall-distance evaluation, and
backend mesh adaptation. It uses polynomial order two and four MPI ranks and
reports wall surface quantities including pressure, skin friction, and heat
flux coefficients.

MATLAB, Python, and Julia drivers are provided. The MATLAB model is named
`pdemodel_axialns.m`; Python and Julia use their local `pdemodel` files.

## Run

```bash
export EXASIM_ROOT=/path/to/Exasim
cd "$EXASIM_ROOT/examples/MeshAdaptivity/isoq2d_idealgas"
matlab -batch "pdeapp"
PYTHONPATH="$EXASIM_ROOT/frontends/Python" python3 pdeapp.py
julia --project="$EXASIM_ROOT/frontends/Julia/Exasim" pdeapp.jl
```

Compare the initial data, adapted coordinates, conservative solution,
artificial viscosity, and wall quantities. Run frontends separately.
