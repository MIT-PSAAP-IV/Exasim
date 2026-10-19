# Three-dimensional ISOQ flow

`pdeapp.py` and `pdeapp.jl` run the maintained three-dimensional Text2Code
application in `apps/navierstokes/isoq3d`. This avoids approximating the
specialized spherical-frustum geometry in another frontend.

Set `EXASIM_PREFIX` before running. The case uses 16 MPI ranks and can be
expensive; outputs are isolated under `python_run` or `julia_run`.
