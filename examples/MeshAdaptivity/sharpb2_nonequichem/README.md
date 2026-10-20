# Axisymmetric finite-rate five-species-air Sharp-B

The native MATLAB driver depends on the MATLAB-only `CNS5air` symbolic
library. The Python and Julia launchers run the canonical reacting Sharp-B
Text2Code bundle in `apps/navierstokes/reactingsharpb2`. This preserves a
tested reacting-flow workflow without copying thousands of generated
expressions into each frontend.

The launchers do not reproduce the MATLAB driver's mesh-adaptivity cycle.
That remains a MATLAB-only workflow until `CNS5air` and its mesh-adaptivity
initialization are available in the Python and Julia frontends. Each launcher
uses four MPI ranks and writes into its own run directory.

Set `EXASIM_PREFIX`, then run `python3 pdeapp.py` or `julia pdeapp.jl`.
`pdeapp.m` remains the authoritative native-frontend configuration.
