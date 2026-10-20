# Convection-diffusion boundary layer

This manufactured steady problem exercises a convection-dominated scalar
equation on an 8-by-8 unit-square mesh. The application uses polynomial order
three, unit diffusion, convection parameters `(100,100)`, and homogeneous
boundary data, producing sharp boundary-layer behavior.

Equivalent MATLAB, Python, and Julia application/model pairs are included.

## Run

```bash
export EXASIM_ROOT=/path/to/Exasim
cd "$EXASIM_ROOT/examples/ConvectionDiffusion/BoundaryLayer"
matlab -batch "pdeapp"
PYTHONPATH="$EXASIM_ROOT/frontends/Python" python3 pdeapp.py
julia --project="$EXASIM_ROOT/frontends/Julia/Exasim" pdeapp.jl
```

Run one frontend at a time. Compare both the scalar solution and its gradients
when checking frontend equivalence, since the boundary layer is sensitive to
mesh resolution and stabilization.
