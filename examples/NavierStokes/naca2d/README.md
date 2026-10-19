# Steady NACA0012 flow

The Python and Julia entry points run the canonical generated steady-airfoil
application in `apps/navierstokes/naca0012steady`. This preserves the curved
airfoil mesh, far-field treatment, and solver configuration.

Set `EXASIM_PREFIX`, then run either entry point. `text2code` generates the
application and the solver runs on one MPI rank in an isolated directory.
