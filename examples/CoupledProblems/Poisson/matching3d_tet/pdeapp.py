"""Python version of the 3D matching coupled Poisson example (pdeapp.m).

Two Poisson models on the unit cube (tetrahedra, interfacetype = 0) coupled
through the interface x = 0.5. The exact solution is
u = exp(x) sin(pi y) sin(pi z). Both meshes come from cubemesh.
Run with `python3 pdeapp.py`; data and the solver build go to ./python_run.
"""

import os
import sys

import numpy

HERE = os.path.dirname(os.path.abspath(__file__))
REPOSITORY = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
# Use EXASIM_PREFIX if set, otherwise a sibling <repo>/../exasim_install.
SOURCE_INSTALL = os.path.join(os.path.dirname(REPOSITORY), "exasim_install")
if "EXASIM_PREFIX" not in os.environ and os.path.isfile(
        os.path.join(SOURCE_INSTALL, "lib", "cmake", "Exasim", "ExasimConfig.cmake")):
    os.environ["EXASIM_PREFIX"] = SOURCE_INSTALL
# The frontend imports pdemodel1.py and pdemodel2.py by module name.
sys.path[0:0] = [os.path.join(REPOSITORY, "frontends", "Python"), HERE]
import exasim  # noqa: E402
from exasim.Preprocessing.createdgnodes import createdgnodes  # noqa: E402
from exasim.Preprocessing.facenumbering import facenumbering  # noqa: E402

pde = [None] * 2
mesh = [None] * 2
exec(open(os.path.join(HERE, "pdeapp1.py")).read())
exec(open(os.path.join(HERE, "pdeapp2.py")).read())

run_directory = os.path.join(HERE, "python_run")
for m in range(2):
    pde[m]['datapath'] = run_directory
    pde[m]['builddir'] = os.path.join(run_directory, ".exasim")
    pde[m]['buildpath'] = pde[m]['builddir']

sol, pde, mesh, master, dmd = exasim.exasim(pde, mesh)[0:5]

# Maximum nodal error against the exact solution. A one-rank model returns its
# solution in partition order, so it is put back into global element order first.
err = 0.0
for m in range(2):
    elemtype = 1 if mesh[m]['t'].shape[0] == 8 else 0
    f = facenumbering(mesh[m]['p'], mesh[m]['t'], elemtype, mesh[m]['boundaryexpr'], [])[0]
    xdg = createdgnodes(mesh[m]['p'], mesh[m]['t'], f, [], [], pde[m]['porder'])
    u = numpy.asarray(sol[m])
    if u.ndim == 4:
        u = u[:, :, :, -1]
    if len(dmd[m]) == 1:
        elempart = numpy.asarray(dmd[m][0]['elempart']).flatten()
        ordered = numpy.empty_like(u)
        ordered[:, :, elempart[:u.shape[2]]] = u
        u = ordered
    exact = (numpy.exp(xdg[:, 0, :]) * numpy.sin(numpy.pi * xdg[:, 1, :])
             * numpy.sin(numpy.pi * xdg[:, 2, :]))
    err = max(err, numpy.max(numpy.abs(u[:, 0, :] - exact)))
print("Maximum nodal error against exp(x) sin(pi y) sin(pi z): %.3e" % err)
