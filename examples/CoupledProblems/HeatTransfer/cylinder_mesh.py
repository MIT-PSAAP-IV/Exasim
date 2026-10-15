"""Meshes and initialization shared by the Mach-8 CHT examples."""

import numpy as np
from exasim.Preprocessing.createdgnodes import createdgnodes
from exasim.Preprocessing.facenumbering import facenumbering
from exasim.Preprocessing.masternodes import masternodes


def _stretch(values, alpha, increasing):
    lo, hi = np.min(values), np.max(values)
    s = (values - lo) / (hi - lo)
    if increasing:
        mapped = (np.exp(alpha * s) - 1.0) / (np.exp(alpha) - 1.0)
    else:
        mapped = (1.0 - np.exp(-alpha * s)) / (1.0 - np.exp(-alpha))
    return lo + (hi - lo) * mapped


def _logical_mesh(nx, ny, elemtype):
    x = np.tile(np.linspace(0.0, 1.0, nx + 1), ny + 1)
    y = np.repeat(np.linspace(0.0, 1.0, ny + 1), nx + 1)
    p = np.vstack((x, y))
    cells = []
    for iy in range(ny):
        for ix in range(nx):
            ll = ix + iy * (nx + 1)
            if elemtype == 1:
                cells.append([ll, ll + 1, ll + nx + 2, ll + nx + 1])
            else:
                # Match mkmesh_square(..., parity=1): NW-SE diagonal.
                cells.extend(([ll, ll + 1, ll + nx + 1], [ll + 1, ll + nx + 2, ll + nx + 1]))
    return p, np.asarray(cells, dtype=np.int64).T


def _half_circle(points, a, b, c):
    theta = 1.5 * np.pi - np.pi * points[1, ...]
    outer = -b * np.cos(theta) + c * (1.0 + np.cos(theta))
    radius = outer + points[0, ...] * (a - outer)
    return np.stack((radius * np.cos(theta), radius * np.sin(theta)), axis=0)


def make_cylinder_mesh(porder, nx, ny, elemtype, solid=False):
    p, t = _logical_mesh(nx, ny, elemtype)
    nfe = 4 if elemtype == 1 else 3
    dgnodes = createdgnodes(p, t, np.zeros((nfe, t.shape[1]), dtype=np.int64), [], [], porder)
    p[0, :] = _stretch(p[0, :], 2.0 if solid else 5.0, solid)
    dgnodes[:, 0, :] = _stretch(dgnodes[:, 0, :], 2.0 if solid else 5.0, solid)
    if solid:
        map_args = (0.85, 1.0, 1.0)
    else:
        map_args = (1.0, 3.0, 4.7)
    p = _half_circle(p, *map_args)
    dgnodes = np.transpose(_half_circle(np.transpose(dgnodes, (1, 0, 2)), *map_args), (1, 0, 2))
    if solid:
        boundaryexpr = [
            lambda z: np.sqrt(z[0, :] ** 2 + z[1, :] ** 2) > 1.0 - 1.0e-6,
            lambda z: z[0, :] > -1.0e-7,
            lambda z: np.abs(z[0, :]) < 20.0,
        ]
    else:
        boundaryexpr = [
            lambda z: np.sqrt(z[0, :] ** 2 + z[1, :] ** 2) < 1.0 + 1.0e-6,
            lambda z: z[0, :] > -1.0e-7,
            lambda z: np.abs(z[0, :]) < 20.0,
        ]
    f = facenumbering(p, t, elemtype, boundaryexpr, [])[0]
    return {
        "p": p, "t": t, "f": f, "dgnodes": dgnodes,
        "udg": [], "vdg": [], "wdg": [], "tprd": [],
        "boundaryexpr": boundaryexpr, "boundarycondition": np.array([3, 2, 1]),
        "periodicexpr": [], "curvedboundary": np.array([], dtype=np.int64),
        "curvedboundaryexpr": [], "periodicboundary": np.array([], dtype=np.int64),
    }


def wall_distance(mesh, porder, elemtype):
    perm = np.asarray(masternodes(porder, 2, elemtype)[4], dtype=np.int64) - 1
    lf, elem = np.nonzero(mesh["f"] == 1)
    wall = np.concatenate([mesh["dgnodes"][perm[:, face], :, e] for face, e in zip(lf, elem)])
    nodes = mesh["dgnodes"]
    distance = np.empty((nodes.shape[0], 1, nodes.shape[2]), order="F")
    for e in range(nodes.shape[2]):
        delta = nodes[:, :, e, None] - wall.T[None, :, :]
        distance[:, 0, e] = np.sqrt(np.sum(delta * delta, axis=1)).min(axis=1)
    return distance
