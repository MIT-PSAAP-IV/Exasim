# Staged ideal-gas cylinder

The MATLAB case combines a curved cylinder mesh, wall-distance initialization,
and several artificial-viscosity continuation solves. The Python and Julia
entry points execute that canonical workflow from isolated copies.

Run `python3 pdeapp.py` or `julia pdeapp.jl`; MATLAB is required. A maintained
Text2Code-based single-stage cylinder example is available in
`hypersoniccylinder_mach8` when a MATLAB-independent workflow is preferred.
