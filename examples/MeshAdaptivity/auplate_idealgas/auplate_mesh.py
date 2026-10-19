"""Frontend-neutral rounded-nose flat-plate mesh construction."""

import numpy as np

from exasim.Preprocessing.createdgnodes import createdgnodes
from exasim.Preprocessing.facenumbering import facenumbering


RNOSE = 8.0e-5
YSHIFT = 1.0e-4
XEND = 0.5
HNOSE = 0.02
HEND = 0.15


def _structured_quads(nx, ny):
    x, y = np.meshgrid(np.linspace(0.0, 1.0, nx + 1),
                       np.linspace(0.0, 1.0, ny + 1), indexing="xy")
    points = np.vstack((x.ravel(), y.ravel()))
    nodes = np.arange((nx + 1) * (ny + 1)).reshape(ny + 1, nx + 1)
    elements = np.vstack((nodes[:-1, :-1].ravel(), nodes[:-1, 1:].ravel(),
                          nodes[1:, 1:].ravel(), nodes[1:, :-1].ravel()))
    return points, elements


def _map(reference):
    xi, eta = reference[0], reference[1]
    x = np.empty_like(xi)
    y = np.empty_like(eta)
    nose = xi <= 1.0 + 1.0e-12
    angle = 0.5 * np.pi * xi[nose]
    height = HNOSE * eta[nose]
    x[nose] = RNOSE - (RNOSE + height) * np.cos(angle)
    y[nose] = (RNOSE + height) * np.sin(angle) + YSHIFT
    body = ~nose
    fraction = xi[body] - 1.0
    wall_x = RNOSE + (XEND - RNOSE) * fraction
    top_x = RNOSE + (XEND - RNOSE) * fraction
    top_y = RNOSE + YSHIFT + HNOSE + (HEND - HNOSE) * fraction
    x[body] = (1.0 - eta[body]) * wall_x + eta[body] * top_x
    y[body] = (1.0 - eta[body]) * (RNOSE + YSHIFT) + eta[body] * top_y
    return np.vstack((x, y))


def make_auplate_mesh(porder):
    import exasim

    points, elements = _structured_quads(116, 41)
    reference_points = np.vstack((2.0 * points[0], points[1]))
    reference_nodes = createdgnodes(reference_points, elements, np.zeros((4, elements.shape[1]), dtype=np.int64), [], [], porder)
    mapped_points = _map(reference_points)
    flat = reference_nodes.transpose(1, 0, 2).reshape(2, -1, order="F")
    dgnodes = _map(flat).reshape(2, reference_nodes.shape[0], reference_nodes.shape[2], order="F").transpose(1, 0, 2)

    center_x, center_y = RNOSE, YSHIFT
    outer_radius = RNOSE + HNOSE
    wall_y = RNOSE + YSHIFT
    boundaryexpr = [
        lambda p: (np.abs(p[1] - YSHIFT) < 1.0e-6) & (p[0] <= 1.0e-6),
        lambda p: np.abs(np.hypot(p[0] - center_x, p[1] - center_y) - outer_radius) < 5.0e-6,
        lambda p: np.abs(np.hypot(p[0] - center_x, p[1] - center_y) - RNOSE) < 1.0e-6,
        lambda p: (np.abs(p[1] - wall_y) < 1.0e-6) & (p[0] >= RNOSE - 1.0e-6),
        lambda p: p[0] > XEND - 1.0e-6,
        lambda p: p[1] > wall_y + 0.5 * HNOSE,
    ]
    faces = facenumbering(mapped_points, elements, 1, boundaryexpr, [])[0]
    _, mesh = exasim.initializeexasim()
    mesh.update({"p": mapped_points, "t": elements, "f": faces, "dgnodes": dgnodes,
                 "boundaryexpr": boundaryexpr,
                 "boundarycondition": np.array([4, 1, 3, 3, 2, 1], dtype=np.int64),
                 "periodicexpr": []})
    return mesh


def wall_distance(mesh):
    x, y = mesh["dgnodes"][:, 0, :], mesh["dgnodes"][:, 1, :]
    nose = np.abs(np.hypot(x - RNOSE, y - YSHIFT) - RNOSE)
    plate = np.hypot(np.maximum(RNOSE - x, 0.0), y - (RNOSE + YSHIFT))
    return np.minimum(nose, np.abs(plate))[:, None, :]
