# Equilibrium-air Sharp-B mesh adaptation

This axisymmetric Sharp-B hypersonic-flow case uses equilibrium five-species
air, artificial-viscosity continuation, wall-distance evaluation, and backend
mesh adaptation. It preserves the physical targets of the ideal-gas Sharp-B
case and uses polynomial order two with eight MPI ranks.

Equivalent MATLAB, Python, and Julia application/model pairs are included.

## Run

```bash
export EXASIM_ROOT=/path/to/Exasim
cd "$EXASIM_ROOT/examples/MeshAdaptivity/sharpb2_equichem"
matlab -batch "pdeapp"
PYTHONPATH="$EXASIM_ROOT/frontends/Python" python3 pdeapp.py
julia --project="$EXASIM_ROOT/frontends/Julia/Exasim" pdeapp.jl
```

This is an expensive MPI case. Compare the adapted mesh, flow and AV fields,
thermodynamic quantities, and species fractions across frontends.
