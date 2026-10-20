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


def squaremesh(m, n, parity=0, elemtype=0):
    """Regular mesh of the unit square, identical to MATLAB squaremesh.m.

    Returns p (2, np) and zero-based t (nve, ne) with the same node and element
    ordering as the MATLAB frontend, so the three frontends solve on the same mesh.
    elemtype 0 gives triangles (parity 0: SW-NE diagonals, 1: NW-SE diagonals),
    elemtype 1 gives quadrilaterals.
    """
    mx, ny = m + 1, n + 1
    x, y = numpy.meshgrid(numpy.arange(mx) / (mx - 1), numpy.arange(ny) / (ny - 1), indexing="ij")
    p = numpy.vstack((x.flatten("F"), y.flatten("F")))

    elements = []
    for j in range(ny - 1):
        for i in range(mx - 1):
            k = i + j * mx
            if elemtype == 1:
                elements.append([k, k + 1, k + mx + 1, k + mx])
            elif parity == 0:
                elements.append([k, k + mx + 1, k + mx])
                elements.append([k, k + 1, k + mx + 1])
            else:
                elements.append([k, k + 1, k + mx])
                elements.append([k + 1, k + mx + 1, k + mx])
    t = numpy.array(elements, dtype=int).T
    return p, t


def dgnodes_of(mesh, porder, model, curved=False, width=0.5):
    """DG nodes in global element order, optionally mapped to a curved interface.

    The curved map x += 0.05 sin(pi y) w, with w the normalized distance from the
    outer boundary of the subdomain (width = distance from that boundary to the
    interface), fixes the outer boundaries and bends the interface.
    """
    elemtype = 1 if mesh["t"].shape[0] == 4 else 0
    f = facenumbering(mesh["p"], mesh["t"], elemtype, mesh["boundaryexpr"], [])[0]
    dg = createdgnodes(mesh["p"], mesh["t"], f, [], [], porder)
    if curved:
        if model == 1:
            w = dg[:, 0, :] / width
        else:
            w = (1.0 - dg[:, 0, :]) / width
        dg[:, 0, :] = dg[:, 0, :] + 0.05 * numpy.sin(numpy.pi * dg[:, 1, :]) * w
    return dg


def global_order(solution, dmd):
    """Return the solution in global element order.

    A one-rank model is returned in its partition order (coupled interface
    elements last); several ranks are already assembled in global order.
    """
    solution = numpy.asarray(solution)
    if solution.ndim == 4:
        solution = solution[:, :, :, -1]
    if len(dmd) == 1:
        elempart = numpy.asarray(dmd[0]["elempart"]).flatten()
        ordered = numpy.empty_like(solution)
        ordered[:, :, elempart[: solution.shape[2]]] = solution
        return ordered
    return solution


def nodal_error(solution, dgnodes):
    """Maximum nodal error of u(:,1,:) against sin(pi x) sin(pi y)."""
    exact = numpy.sin(numpy.pi * dgnodes[:, 0, :]) * numpy.sin(numpy.pi * dgnodes[:, 1, :])
    return numpy.max(numpy.abs(solution[:, 0, :] - exact))

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
