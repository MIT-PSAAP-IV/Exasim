"""Structured curved annulus used by the legacy Mach-8.98 cylinder case."""

import numpy as np

from exasim.Preprocessing.createdgnodes import createdgnodes
from exasim.Preprocessing.facenumbering import facenumbering
from exasim.Preprocessing.masternodes import masternodes


def _loginc(values, alpha):
    lower, upper = np.min(values), np.max(values)
    return lower + (upper - lower) * (
        np.exp(alpha * (values - lower) / (upper - lower)) - 1.0
    ) / (np.exp(alpha) - 1.0)


def _logdec(values, alpha):
    lower, upper = np.min(values), np.max(values)
    return lower + (upper - lower) * (
        1.0 - np.exp(-alpha * (values - lower) / (upper - lower))
    ) / (1.0 - np.exp(-alpha))


def _quad_mesh(nx, ny):
    x = np.tile(np.linspace(0.0, 1.0, nx + 1), ny + 1)
    y = np.repeat(np.linspace(0.0, 1.0, ny + 1), nx + 1)
    points = np.vstack((x, y))
    elements = []
    for iy in range(ny):
        for ix in range(nx):
            node = ix + iy * (nx + 1)
            elements.append([node, node + 1, node + nx + 2, node + nx + 1])
    return points, np.asarray(elements, dtype=np.int64).T


def _map_annulus(points, inner=1.0, outer=3.0):
    angle = np.pi * points[0, ...]
    radius = inner + (outer - inner) * points[1, ...]
    return np.stack((-radius * np.sin(angle), -radius * np.cos(angle)), axis=0)


def make_cylinder_mesh(porder, m=101, n=101):
    points, elements = _quad_mesh(m - 1, n - 1)

    n1 = int(np.ceil(5.5 * n / 10.0))
    n2 = int(np.ceil(3.05 * n / 10.0))
    n3 = n - n1 - n2
    radial = np.concatenate((
        _loginc(np.linspace(0.0, 0.1, n1 + 1), 3.0)[:-1],
        _loginc(np.linspace(0.1, 0.375, n2 + 1), 1.0)[:-1],
        _loginc(np.linspace(0.375, 1.0, n3), 2.0),
    ))
    points[1, :] = np.tile(radial, m)
    left = points[0, :] <= 0.5
    points[0, left] = _logdec(points[0, left], 1.0)
    points[0, ~left] = _loginc(points[0, ~left], 1.0)

    reference_points, reference_elements = _quad_mesh(m - 1, n - 1)
    reference_points[1, :] = np.tile(radial, m)
    left = reference_points[0, :] <= 0.5
    reference_points[0, left] = _logdec(reference_points[0, left], 1.0)
    reference_points[0, ~left] = _loginc(reference_points[0, ~left], 1.0)
    dgnodes = createdgnodes(
        reference_points, reference_elements,
        np.zeros((4, reference_elements.shape[1]), dtype=np.int64), [], [], porder
    )
    points = _map_annulus(points)
    dgnodes = np.transpose(_map_annulus(np.transpose(dgnodes, (1, 0, 2))), (1, 0, 2))

    boundaryexpr = [
        lambda p: np.sqrt(p[0, :] ** 2 + p[1, :] ** 2) < 1.0 + 1.0e-6,
        lambda p: p[0, :] > -1.0e-7,
        lambda p: np.abs(p[0, :]) < 20.0,
    ]
    faces = facenumbering(points, elements, 1, boundaryexpr, [])[0]
    return {
        "p": points, "t": elements, "f": faces, "dgnodes": dgnodes,
        "udg": [], "vdg": [], "wdg": [], "tprd": [],
        "boundaryexpr": boundaryexpr,
        "boundarycondition": np.array([3, 6, 5], dtype=np.int64),
        "periodicexpr": [], "curvedboundary": np.array([], dtype=np.int64),
        "curvedboundaryexpr": [], "periodicboundary": np.array([], dtype=np.int64),
    }


def wall_distance(mesh, porder):
    perm = np.asarray(masternodes(porder, 2, 1)[4], dtype=np.int64) - 1
    local_faces, elements = np.nonzero(mesh["f"] == 1)
    wall_nodes = np.concatenate([
        mesh["dgnodes"][perm[:, face], :, elem]
        for face, elem in zip(local_faces, elements)
    ])
    npe, _, ne = mesh["dgnodes"].shape
    distance = np.empty((npe, 1, ne), order="F")
    for elem in range(ne):
        delta = mesh["dgnodes"][:, :, elem, None] - wall_nodes.T[None, :, :]
        distance[:, 0, elem] = np.sqrt(np.sum(delta * delta, axis=1)).min(axis=1)
    return distance


def curved_map_scale(mesh):
    """Return sqrt(J) for the analytic half-annulus map at every DG node."""
    radius = np.sqrt(mesh["dgnodes"][:, 0, :] ** 2 + mesh["dgnodes"][:, 1, :] ** 2)
    return np.sqrt(2.0 * np.pi * radius)


def initialize_solution(mesh, distance, parameters):
    npe, _, ne = mesh["dgnodes"].shape
    solution = np.empty((npe, 4, ne), order="F")
    solution[:, 0, :] = parameters[4]
    solution[:, 1, :] = parameters[5] * np.tanh(10.0 * distance[:, 0, :])
    solution[:, 2, :] = parameters[6] * np.tanh(10.0 * distance[:, 0, :])
    near_wall_temperature = parameters[8] * (parameters[10] / parameters[9] - 1.0)
    near_wall_temperature *= np.exp(-10.0 * distance[:, 0, :])
    near_wall_temperature += parameters[8]
    solution[:, 3, :] = near_wall_temperature + 0.5 * (
        solution[:, 1, :] ** 2 + solution[:, 2, :] ** 2
    )
    return solution
