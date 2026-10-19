# Reynolds-5000 NACA0012 workflow

This parameter-continuation case uses MATLAB-specific airfoil mesh generation
and pseudo-transient continuation scripts. `pdeapp.py` and `pdeapp.jl` execute
the canonical `pdeapp.m` in isolated directories rather than mapping it to the
different maintained steady NACA application.

MATLAB is required. The original Reynolds number, curved mesh, and
continuation schedule are preserved.
