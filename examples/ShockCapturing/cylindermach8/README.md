# Mach-8 cylinder shock-capturing example

This directory contains equivalent MATLAB, Python, and Julia frontends for the
same Exasim `ModelD` problem.  All three use the same curved 31-by-21
quadrilateral mesh, fourth-order HDG discretization, boundary markers, initial
state, solver settings, and artificial-viscosity continuation with the internal
HDG Helmholtz filter.

Run the frontends from the repository checkout with:

```bash
/Applications/MATLAB_R2025b.app/bin/matlab -batch \
  "set(0,'DefaultFigureVisible','off'); cd('examples/ShockCapturing/cylindermach8'); pdeapp"

MPLCONFIGDIR=/tmp/exasim-mpl \
PYTHONPATH=frontends/Python \
/usr/bin/python3 examples/ShockCapturing/cylindermach8/pdeapp.py

JULIA_DEPOT_PATH=/tmp/exasim-julia-depot:$HOME/.julia \
julia --project=frontends/Julia/Exasim \
  examples/ShockCapturing/cylindermach8/pdeapp.jl
```

The Python and Julia runs use isolated `python_run` and `julia_run`
subdirectories so their generated input and output files can be compared.  Once
all three runs have finished, create the numerical report and common plots with:

```bash
MPLCONFIGDIR=/tmp/exasim-mpl \
/usr/bin/python3 examples/ShockCapturing/cylindermach8/compare_frontends.py
```

The comparison script checks `app.bin`, `mesh.bin`, `sol.bin`, and `master.bin`,
then reports pairwise L2, Linf, and relative-L2 differences for the conservative
variables, primitive variables, Mach number, wall distance, filtered AV sensor,
and effective artificial viscosity.
