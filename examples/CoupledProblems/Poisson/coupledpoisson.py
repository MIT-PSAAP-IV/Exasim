"""Helpers shared by the Python versions of the coupled Poisson examples."""

from __future__ import annotations

import os
import sys

import numpy as np

POISSON = os.path.dirname(os.path.abspath(__file__))
REPOSITORY = os.path.abspath(os.path.join(POISSON, "..", "..", ".."))


def setup_exasim(example_directory):
    """Put the example directory and the source-tree Python frontend on the path.

    The example directory must be on sys.path because the frontend imports the
    PDE model files (pdemodel1.py, pdemodel2.py) by module name. A sibling
    source install (<repo>/../exasim_install) is used unless EXASIM_PREFIX is set.
    """
    source_install = os.path.join(os.path.dirname(REPOSITORY), "exasim_install")
    if "EXASIM_PREFIX" not in os.environ and os.path.isfile(
        os.path.join(source_install, "lib", "cmake", "Exasim", "ExasimConfig.cmake")
    ):
        os.environ["EXASIM_PREFIX"] = source_install
    for path in (os.path.join(REPOSITORY, "frontends", "Python"), example_directory):
        if path not in sys.path:
            sys.path.insert(0, path)


def squaremesh(m, n, parity=0, elemtype=0):
    """Regular mesh of the unit square, identical to MATLAB squaremesh.m.

    Returns p (2, np) and zero-based t (nve, ne) with the same node and element
    ordering as the MATLAB frontend, so the three frontends solve on the same mesh.
    elemtype 0 gives triangles (parity 0: SW-NE diagonals, 1: NW-SE diagonals),
    elemtype 1 gives quadrilaterals.
    """
    mx, ny = m + 1, n + 1
    x, y = np.meshgrid(np.arange(mx) / (mx - 1), np.arange(ny) / (ny - 1), indexing="ij")
    p = np.vstack((x.flatten("F"), y.flatten("F")))

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
    t = np.array(elements, dtype=int).T
    return p, t


def dgnodes_of(mesh, porder, model, curved=False, width=0.5):
    """DG nodes in global element order, optionally mapped to a curved interface.

    The curved map x += 0.05 sin(pi y) w, with w the normalized distance from the
    outer boundary of the subdomain (width = distance from that boundary to the
    interface), fixes the outer boundaries and bends the interface.
    """
    from exasim.Preprocessing.createdgnodes import createdgnodes
    from exasim.Preprocessing.facenumbering import facenumbering

    elemtype = 1 if mesh["t"].shape[0] == 4 else 0
    f = facenumbering(mesh["p"], mesh["t"], elemtype, mesh["boundaryexpr"], [])[0]
    dg = createdgnodes(mesh["p"], mesh["t"], f, [], [], porder)
    if curved:
        if model == 1:
            w = dg[:, 0, :] / width
        else:
            w = (1.0 - dg[:, 0, :]) / width
        dg[:, 0, :] = dg[:, 0, :] + 0.05 * np.sin(np.pi * dg[:, 1, :]) * w
    return dg


def global_order(solution, dmd):
    """Return the solution in global element order.

    A one-rank model is returned in its partition order (coupled interface
    elements last); several ranks are already assembled in global order.
    """
    solution = np.asarray(solution)
    if solution.ndim == 4:
        solution = solution[:, :, :, -1]
    if len(dmd) == 1:
        elempart = np.asarray(dmd[0]["elempart"]).flatten()
        ordered = np.empty_like(solution)
        ordered[:, :, elempart[: solution.shape[2]]] = solution
        return ordered
    return solution


def nodal_error(solution, dgnodes):
    """Maximum nodal error of u(:,1,:) against sin(pi x) sin(pi y)."""
    exact = np.sin(np.pi * dgnodes[:, 0, :]) * np.sin(np.pi * dgnodes[:, 1, :])
    return np.max(np.abs(solution[:, 0, :] - exact))


def save_results(directory, solutions, dgnodes):
    """Write sol<m>.npy and xdg<m>.npy (global element order) for cross-frontend checks."""
    os.makedirs(directory, exist_ok=True)
    for m, (solution, nodes) in enumerate(zip(solutions, dgnodes), start=1):
        np.save(os.path.join(directory, f"sol{m}.npy"), solution)
        np.save(os.path.join(directory, f"xdg{m}.npy"), nodes)
