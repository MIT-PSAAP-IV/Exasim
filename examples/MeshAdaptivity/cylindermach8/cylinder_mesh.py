"""Mesh and initialization shared by the Python cylinder-Mach-8 driver.

This is a direct translation of ``mkmesh_cyl.m`` plus the wall-distance
construction in ``pdeapp.m``.  The discrete distance intentionally uses the
same set of p-order wall nodes as MATLAB's ``meshdist3``; using the analytic
circle distance would define a slightly different initial condition.
"""

from __future__ import annotations

import numpy as np

from exasim.Preprocessing.createdgnodes import createdgnodes
from exasim.Preprocessing.facenumbering import facenumbering
from exasim.Preprocessing.masternodes import masternodes


def _logdec(values, alpha):
    a = np.min(values)
    b = np.max(values)
    return a + (b - a) * (
        1.0 - np.exp(-alpha * (values - a) / (b - a))
    ) / (1.0 - np.exp(-alpha))


def _map_halfcircle(points, a=1.0, b=3.0, c=4.7):
    theta = 1.5 * np.pi + points[1, ...] * (0.5 * np.pi - 1.5 * np.pi)
    outer_radius = -b * np.cos(theta) + c * (1.0 + np.cos(theta))
    radius = outer_radius + points[0, ...] * (a - outer_radius)
    return np.stack((radius * np.cos(theta), radius * np.sin(theta)), axis=0)


def _rectangular_quad_mesh(nx, ny):
    """Equivalent of MATLAB squaremesh(nx,ny,1,1), including element order."""
    x = np.tile(np.linspace(0.0, 1.0, nx + 1), ny + 1)
    y = np.repeat(np.linspace(0.0, 1.0, ny + 1), nx + 1)
    p = np.vstack((x, y))
    elements = []
    for iy in range(ny):
        for ix in range(nx):
            lower_left = ix + iy * (nx + 1)
            elements.append(
                [lower_left, lower_left + 1, lower_left + nx + 2, lower_left + nx + 1]
            )
    return p, np.asarray(elements, dtype=np.int64).T


def make_cylinder_mesh(
    porder, nx=31, ny=21, radial_decay=6.0, outer_offset=4.7
):
    p, t = _rectangular_quad_mesh(nx, ny)
    # Construct the tensor-product DG nodes before applying the nonlinear map,
    # exactly as mkmesh_square followed by mkmesh_halfcircle does in MATLAB.
    f0 = np.zeros((4, t.shape[1]), dtype=np.int64)
    dgnodes = createdgnodes(p, t, f0, [], [], porder)

    p[0, :] = _logdec(p[0, :], radial_decay)
    dgnodes[:, 0, :] = _logdec(dgnodes[:, 0, :], radial_decay)
    p = _map_halfcircle(p, c=outer_offset)
    dgnodes = np.transpose(
        _map_halfcircle(np.transpose(dgnodes, (1, 0, 2)), c=outer_offset),
        (1, 0, 2),
    )

    boundaryexpr = [
        lambda x: np.sqrt(x[0, :] ** 2 + x[1, :] ** 2) < 1.0 + 1.0e-6,
        lambda x: x[0, :] > -1.0e-7,
        lambda x: np.abs(x[0, :]) < 20.0,
    ]
    f = facenumbering(p, t, 1, boundaryexpr, [])[0]

    return {
        "p": p,
        "t": t,
        "f": f,
        "dgnodes": dgnodes,
        "udg": [],
        "vdg": [],
        "wdg": [],
        "tprd": [],
        "boundaryexpr": boundaryexpr,
        "boundarycondition": np.array([3, 6, 5], dtype=np.int64),
        "periodicexpr": [],
        "curvedboundary": np.array([], dtype=np.int64),
        "curvedboundaryexpr": [],
        "periodicboundary": np.array([], dtype=np.int64),
    }


def wall_distance(mesh, porder):
    """Reproduce meshdist3(mesh.f, mesh.dgnodes, mesh.perm, [1])."""
    perm = np.asarray(masternodes(porder, 2, 1)[4], dtype=np.int64) - 1
    local_faces, elements = np.nonzero(mesh["f"] == 1)
    wall_nodes = np.concatenate(
        [mesh["dgnodes"][perm[:, face], :, elem] for face, elem in zip(local_faces, elements)],
        axis=0,
    )

    npe, _, ne = mesh["dgnodes"].shape
    distance = np.empty((npe, 1, ne), dtype=np.float64, order="F")
    for elem in range(ne):
        delta = mesh["dgnodes"][:, :, elem, None] - wall_nodes.T[None, :, :]
        distance[:, 0, elem] = np.sqrt(np.sum(delta * delta, axis=1)).min(axis=1)
    return distance


def initialize_solution(mesh, distance, physicsparam):
    rinf, ruinf, rvinf, rEinf = physicsparam[4:8]
    tinf, tref, twall = physicsparam[8:11]
    npe, _, ne = mesh["dgnodes"].shape
    udg = np.empty((npe, 4, ne), dtype=np.float64, order="F")
    udg[:, 0, :] = rinf
    udg[:, 1, :] = ruinf * np.tanh(10.0 * distance[:, 0, :])
    udg[:, 2, :] = rvinf * np.tanh(10.0 * distance[:, 0, :])
    t_near_wall = (
        tinf * (twall / tref - 1.0) * np.exp(-10.0 * distance[:, 0, :]) + tinf
    )
    udg[:, 3, :] = t_near_wall + 0.5 * (
        udg[:, 1, :] ** 2 + udg[:, 2, :] ** 2
    )
    return udg
