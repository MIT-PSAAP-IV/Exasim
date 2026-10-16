"""Python version of the non-matching coupled Poisson example (pdeapp.m).

Coupled Poisson problem on two independently meshed subdomains with a
non-matching interface (interfacetype = 1). The exact solution is
u = sin(pi x) sin(pi y) on [0,1]^2; model 1 covers x <= 0.5, model 2 x >= 0.5.

Set curved = True to bend the interface to x = 0.5 + 0.05 sin(pi y): both sides
then interpolate the curve at their own face nodes, so the two discrete
interfaces leave small gaps and overlaps that the non-matching coupling bridges.
Run with `python3 pdeapp.py`; data and the solver build go to ./python_run.
"""

import os
import sys

import numpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from coupledpoisson import (  # noqa: E402
    dgnodes_of, global_order, nodal_error, save_results, setup_exasim, squaremesh)

setup_exasim(HERE)
import exasim  # noqa: E402

curved = False

pde = [None] * 2
mesh = [None] * 2
exec(open(os.path.join(HERE, "pdeapp1.py")).read())
exec(open(os.path.join(HERE, "pdeapp2.py")).read())

run_directory = os.path.join(HERE, "python_run")
for m in range(2):
    pde[m]['datapath'] = run_directory
    pde[m]['builddir'] = os.path.join(run_directory, ".exasim")
    pde[m]['buildpath'] = pde[m]['builddir']

# DG nodes in the global element order of each mesh, optionally mapped to the
# curved interface (the map fixes the outer boundaries). preprocessing() uses
# mesh[m]['dgnodes'] when it is present.
for m in range(2):
    mesh[m]['dgnodes'] = dgnodes_of(mesh[m], pde[m]['porder'], m + 1, curved)

# The solver prints "Non-matching donor search, Gamma^1" and "..., Gamma^2"
# diagnostics (faces, points, maximum point-to-face distance, points off their
# donor face) for both coupling directions during setup.
sol, pde, mesh, master, dmd = exasim.exasim(pde, mesh)[0:5]

# Put the solution of a one-rank model into the global element order as well
# (several ranks are already assembled in that order), then compare with the
# exact solution at the DG nodes.
xdg = [dgnodes_of(mesh[m], pde[m]['porder'], m + 1, curved) for m in range(2)]
sol = [global_order(sol[m], dmd[m]) for m in range(2)]
err = max(nodal_error(sol[m], xdg[m]) for m in range(2))
print("Maximum nodal error against sin(pi x) sin(pi y): %.3e" % err)
save_results(os.path.join(run_directory, "results"), sol, xdg)
