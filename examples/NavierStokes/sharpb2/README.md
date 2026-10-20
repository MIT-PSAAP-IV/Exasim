# Axisymmetric Sharp-B flow

The portable entry points use `apps/navierstokes/sharpb2`, preserving the
axisymmetric flux and source terms, body-fitted mesh, boundary labels, and
solver settings from the maintained generated application.

Set `EXASIM_PREFIX`, then run `python3 pdeapp.py` or `julia pdeapp.jl`. The
four-rank solve writes into an isolated frontend-specific directory.
