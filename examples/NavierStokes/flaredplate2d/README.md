# Hypersonic flared plate

This case uses a body-fitted flared-plate mesh followed by a staged
artificial-viscosity continuation procedure. Those mesh-distance and restart
operations are implemented only in the MATLAB workflow.

`pdeapp.py` and `pdeapp.jl` run the canonical `pdeapp.m` from isolated copies.
MATLAB is required. This preserves the mesh, continuation schedule, boundary
labels, and solver settings rather than replacing them with a simpler case.
