# Wetting and drying on a sloping beach

Native MATLAB, Python, and Julia frontends solve the same regularized
one-dimensional shallow-water system with a tidal right boundary and dry left
boundary. The ports retain the depth regularization, momentum damping,
artificial-viscosity sensor, HDG characteristic boundary state, and `400`
element mesh.

Run `python3 pdeapp.py`, `julia pdeapp.jl`, or `pdeapp.m`. The three-period
simulation contains 2700 time steps and is not intended as a quick smoke test.
