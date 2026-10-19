# Finite-rate five-species-air Mach-8 cylinder

The governing model depends on the MATLAB `CNS5air` library, which has no
native Python or Julia symbolic-code-generation implementation. To preserve
the equations exactly, `pdeapp.py` and `pdeapp.jl` launch the canonical
Text2Code application in `apps/meshadaptivity/cylindermach8_nonequichem`
instead of substituting a reduced chemistry model.

Set `EXASIM_PREFIX`, then run `python3 pdeapp.py` or `julia pdeapp.jl`. Each
launcher copies the immutable application inputs into its own run directory,
generates the model, and runs the eight-rank solve. The MATLAB-native frontend
workflow remains available as `pdeapp.m`.
