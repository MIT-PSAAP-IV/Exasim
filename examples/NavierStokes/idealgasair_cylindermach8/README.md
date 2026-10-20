# Ideal-gas-air Mach-8 cylinder

The portable Python and Julia entry points use the maintained ideal-gas
shock-capturing application in `apps/shockcapturing/cylindermach8`. They copy
the app, generate it with `text2code`, and run the one-rank solve.

Set `EXASIM_PREFIX` before running `python3 pdeapp.py` or `julia pdeapp.jl`.
The original MATLAB script remains the reference for its longer staged
continuation study.
