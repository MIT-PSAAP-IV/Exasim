# Kovasznay flow

Native MATLAB, Python, and Julia frontends solve the same manufactured
two-dimensional Stokes problem on a `16 x 16` quadrilateral mesh. The
exponential Kovasznay velocity is imposed through the HDG boundary residual.

Run `python3 pdeapp.py`, `julia pdeapp.jl`, or `pdeapp.m`. The ports retain
polynomial order three, viscosity `0.1`, and the original solver settings.
