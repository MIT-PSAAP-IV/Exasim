# Three-dimensional Eppler 387 wing

The canonical workflow first constructs the curved two-dimensional Eppler
mesh and then extrudes it spanwise. Its mesh mapping and result-transfer steps
are MATLAB-specific.

Use `python3 pdeapp.py` or `julia pdeapp.jl` to execute `pdeapp.m` in an
isolated copy. MATLAB is required. The launchers intentionally preserve the
32-rank production configuration and do not replace the wing mesh.
