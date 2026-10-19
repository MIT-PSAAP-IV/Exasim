# Axisymmetric finite-rate five-species-air ISOQ

The MATLAB model uses the large `CNS5air` symbolic library. The Python and
Julia scripts preserve that model by launching the canonical Text2Code
application in `apps/meshadaptivity/isoq2d_nonequichem`; they do not replace it
with an ideal-gas or inviscid approximation.

Set `EXASIM_PREFIX`, then run `python3 pdeapp.py` or `julia pdeapp.jl`. The
eight-rank run is isolated in `python_run` or `julia_run`. Use `pdeapp.m` for
the native MATLAB frontend workflow.
