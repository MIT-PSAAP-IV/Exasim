# Matching-interface Mach-8 conjugate heat transfer

This two-model example couples compressible Navier--Stokes flow around a Mach-8
cylinder to heat conduction in a solid. The fluid and solid meshes share a
conformal curved interface. A fluid-only precursor is solved first and then
used to initialize the monolithic coupled solve.

`pdeapp.m`, `pdeapp.py`, and `pdeapp.jl` are the coupled drivers;
`pdemodel1.*` defines the fluid and `pdemodel2.*` defines the solid. The MATLAB
helper applications `pdeapp1.m` and `pdeapp2.m` expose the individual models.

## Run

```bash
export EXASIM_ROOT=/path/to/Exasim
cd "$EXASIM_ROOT/examples/CoupledProblems/HeatTransfer/cylindermach8_matching"
matlab -batch "pdeapp"
PYTHONPATH="$EXASIM_ROOT/frontends/Python" python3 pdeapp.py
julia --project="$EXASIM_ROOT/frontends/Julia/Exasim" pdeapp.jl
```

The coupled executable requires MPI. Run frontends separately because they
create solver inputs and outputs beneath this case directory. Successful runs
should show a common nonlinear history for both models and continuous
temperature/heat-flux transfer across the matching interface.
