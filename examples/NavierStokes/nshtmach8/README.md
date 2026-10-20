# Mach-8 cylinder flow and heat transfer

This legacy driver performs separate flow and heat solves with explicit
solution transfer between nonidentical meshes. The transfer utilities and
staged initialization are MATLAB-specific. The newer monolithic coupled
examples live under `examples/CoupledProblems/HeatTransfer`.

`pdeapp.py` and `pdeapp.jl` preserve this historical workflow by launching
`pdeapp.m` in isolation. MATLAB is required.
