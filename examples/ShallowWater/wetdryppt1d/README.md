# Positivity-preserving wet/dry shallow water

The MATLAB, Python, and Julia models use the transformed depth
`H = Hmin + exp(phi)` on the same 400-element sloping beach. All ports retain
the Rusanov boundary flux, tidal forcing, bathymetric source, and two-stage
time integration.

Run `python3 pdeapp.py`, `julia pdeapp.jl`, or `pdeapp.m`. The scripts report
the final transformed-depth range; plotting remains optional postprocessing.
