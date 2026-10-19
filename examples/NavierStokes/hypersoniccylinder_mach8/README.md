# Shock-capturing hypersonic cylinder

`pdeapp.py` and `pdeapp.jl` run the canonical generated application in
`apps/shockcapturing/cylindermach8`. This retains the production shock sensor,
artificial-viscosity model, curved cylinder mesh, and boundary conventions.

Set `EXASIM_PREFIX`, then run either entry point. The application is generated
with `text2code` and executed on one MPI rank in a frontend-specific run
directory.
