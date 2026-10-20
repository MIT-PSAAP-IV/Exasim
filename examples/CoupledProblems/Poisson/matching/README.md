# Coupled Poisson problem with a matching interface

Two third-order Poisson models cover the left and right halves of the unit
square and share the conformal interface `x=0.5`. The models use matching face
nodes and the standard conformal interface path.

Each frontend supplies a coupled driver (`pdeapp.*`), two model setup files
(`pdeapp1.*`, `pdeapp2.*`), and two model definitions (`pdemodel1.*`,
`pdemodel2.*`).

## Run

```bash
export EXASIM_ROOT=/path/to/Exasim
cd "$EXASIM_ROOT/examples/CoupledProblems/Poisson/matching"
matlab -batch "pdeapp"
PYTHONPATH="$EXASIM_ROOT/frontends/Python" python3 pdeapp.py
julia --project="$EXASIM_ROOT/frontends/Julia/Exasim" pdeapp.jl
```

The default MATLAB setup assigns three ranks to model 1 and four to model 2;
the coupled executable therefore requires MPI. Run only one frontend at a time
unless its output directory has been explicitly isolated.
