"""Python version of the matching-interface coupled Poisson example (pdeapp.m).

Two Poisson models on [0,0.5] x [0,1] (3 MPI ranks) and [0.5,1] x [0,1]
(4 MPI ranks) are coupled through the matching interface x = 0.5. Run with
`python3 pdeapp.py`; data and the solver build go to ./python_run.
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

# create pde and mesh for each PDE model
pde = [None] * 2
mesh = [None] * 2
exec(open(os.path.join(HERE, "pdeapp1.py")).read())
exec(open(os.path.join(HERE, "pdeapp2.py")).read())

run_directory = os.path.join(HERE, "python_run")
for m in range(2):
    pde[m]['datapath'] = run_directory
    pde[m]['builddir'] = os.path.join(run_directory, ".exasim")
    pde[m]['buildpath'] = pde[m]['builddir']

# set an interface condition for the interface of pde1
# the size of this array must be equal to the number of boundaries for pde1
# a nonzero element of this array corresponds to a particular fint of pdemodel1
mesh[0]['interfacecondition'] = numpy.array([0, 1, 0, 0])

# a list of fbou components that are used to impose interface condition
pde[0]['interfacefluxmap'] = numpy.array([1])

# set an interface condition for the interface of pde2
mesh[1]['interfacecondition'] = numpy.array([0, 0, 0, 1])
pde[1]['interfacefluxmap'] = numpy.array([1])

# call exasim to generate and run C++ code to solve the coupled PDE models
sol, pde, mesh, master, dmd = exasim.exasim(pde, mesh)[0:5]

# solutions and DG nodes in the global element order of each mesh
xdg = [dgnodes_of(mesh[m], pde[m]['porder'], m + 1) for m in range(2)]
sol = [global_order(sol[m], dmd[m]) for m in range(2)]
err = max(nodal_error(sol[m], xdg[m]) for m in range(2))
print("Maximum nodal error against sin(pi x) sin(pi y): %.3e" % err)
save_results(os.path.join(run_directory, "results"), sol, xdg)
