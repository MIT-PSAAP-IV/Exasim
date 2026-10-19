# Two-dimensional viscous Burgers problem

This steady scalar example combines diffusion, linear convection, and a
quadratic Burgers flux on an 8-by-8 unit-square mesh. It uses polynomial order
three and mixed boundary conditions.

The MATLAB, Python, and Julia `pdeapp` files configure equivalent models using
the matching `pdemodel` implementation for each frontend.

## Run

```bash
export EXASIM_ROOT=/path/to/Exasim
cd "$EXASIM_ROOT/examples/Burgers/Burgers2d"
matlab -batch "pdeapp"
PYTHONPATH="$EXASIM_ROOT/frontends/Python" python3 pdeapp.py
julia --project="$EXASIM_ROOT/frontends/Julia/Exasim" pdeapp.jl
```

Run the frontends separately to avoid overwriting generated files. The primary
solution is the scalar field; gradient components are available for
visualization in the mixed formulation.
