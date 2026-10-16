"""Python version of the mixed triangle/quadrilateral non-matching Poisson example (pdeapp.m).

A 2D non-matching interface at x = 0.5 between a triangle subdomain and a
quadrilateral subdomain (triangles on model 1 by default). The exact solution
is u = sin(pi x) sin(pi y).

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

triangleOnModel1 = True  # False reverses the triangle/quad orientation
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

for m in range(2):
    mesh[m]['dgnodes'] = dgnodes_of(mesh[m], pde[m]['porder'], m + 1, curved, 0.5)
sol, pde, mesh, master, dmd = exasim.exasim(pde, mesh)[0:5]

xdg = [dgnodes_of(mesh[m], pde[m]['porder'], m + 1, curved, 0.5) for m in range(2)]
sol = [global_order(sol[m], dmd[m]) for m in range(2)]
err = max(nodal_error(sol[m], xdg[m]) for m in range(2))
print("Mixed triangle/quad maximum nodal error: %.3e" % err)
save_results(os.path.join(run_directory, "results"), sol, xdg)
