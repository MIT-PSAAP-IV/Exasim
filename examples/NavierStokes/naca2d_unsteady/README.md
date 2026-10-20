# Unsteady NACA0012 flow

The Python and Julia entry points run the maintained generated application in
`apps/navierstokes/naca0012unsteady`. It retains the curved airfoil geometry,
time-step schedule, and eight-rank partitioning.

Set `EXASIM_PREFIX`, then run `python3 pdeapp.py` or `julia pdeapp.jl`.
Generated data and outputs remain in frontend-specific run directories.
