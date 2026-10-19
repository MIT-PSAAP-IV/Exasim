# AU hypersonic plate

The canonical case performs a sequence of artificial-viscosity continuation
solves and uses MATLAB-specific wall-distance and field-transfer utilities.
To preserve that staged algorithm exactly, `pdeapp.py` and `pdeapp.jl` run
`pdeapp2.m` from an isolated copy rather than substituting a single solve.

Run `python3 pdeapp.py` or `julia pdeapp.jl`. MATLAB is required. A separate,
native cross-frontend preprocessing example is available in
`examples/MeshAdaptivity/auplate_idealgas`.
