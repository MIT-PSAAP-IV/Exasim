# Ideal-gas Sharp-B mesh adaptation

This axisymmetric Mach-21 Sharp-B example combines an ideal-gas Navier--Stokes
model, artificial-viscosity continuation, wall-distance evaluation, and
backend mesh adaptation. It uses polynomial order two and defaults to eight MPI
ranks.

MATLAB, Python, and Julia drivers are included. The MATLAB driver uses
`pdemodel_axialns.m`; the other frontends use their local model files.

## Run

```bash
export EXASIM_ROOT=/path/to/Exasim
cd "$EXASIM_ROOT/examples/MeshAdaptivity/sharpb2_idealgas"
matlab -batch "pdeapp"
PYTHONPATH="$EXASIM_ROOT/frontends/Python" python3 pdeapp.py
julia --project="$EXASIM_ROOT/frontends/Julia/Exasim" pdeapp.jl
```

Run one frontend at a time and compare the final adapted coordinates,
conservative flow state, artificial viscosity, and wall quantities.
