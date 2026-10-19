# Vibrating membrane

This transient two-dimensional wave-equation example models a vibrating
membrane on an 8-by-8 unit-square mesh. It uses polynomial order four and 100
time steps of size `0.02`.

Equivalent MATLAB, Python, and Julia application/model files are provided.

## Run

```bash
export EXASIM_ROOT=/path/to/Exasim
cd "$EXASIM_ROOT/examples/WaveEquation/Membrane"
matlab -batch "pdeapp"
PYTHONPATH="$EXASIM_ROOT/frontends/Python" python3 pdeapp.py
julia --project="$EXASIM_ROOT/frontends/Julia/Exasim" pdeapp.jl
```

Run one frontend at a time. Compare displacement, velocity, and gradient fields
at matching physical times.
