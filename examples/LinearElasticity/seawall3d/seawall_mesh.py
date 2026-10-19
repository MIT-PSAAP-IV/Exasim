from pathlib import Path

import numpy as np

import exasim
from exasim.Preprocessing.createdgnodes import createdgnodes


def _extrude_quads(points2d, elements2d, z):
    nxy = points2d.shape[1]
    points = np.empty((3, nxy * z.size))
    for layer, coordinate in enumerate(z):
        offset = layer * nxy
        points[0:2, offset : offset + nxy] = points2d
        points[2, offset : offset + nxy] = coordinate
    elements = np.empty((8, elements2d.shape[1] * (z.size - 1)), dtype=int)
    for layer in range(z.size - 1):
        start = layer * elements2d.shape[1]
        stop = start + elements2d.shape[1]
        elements[0:4, start:stop] = elements2d + layer * nxy
        elements[4:8, start:stop] = elements2d + (layer + 1) * nxy
    return points, elements


def _map_xz(x, z, xmax, l1, radius, l2, l3):
    mapped_x, mapped_z = np.array(x, copy=True), np.array(z, copy=True)
    mask = (z >= l2 + 1.0) & (z <= l2 + 1.0 + l3)
    mapped_x[mask] = xmax + radius + (z[mask] - (l2 + 1.0))
    mapped_z[mask] = l2 + radius + xmax - x[mask]
    mask = (z >= l2) & (z <= l2 + 1.0)
    angle = np.pi - 0.5 * np.pi * (z[mask] - l2)
    radial = xmax + radius - x[mask]
    mapped_x[mask] = xmax + radius + radial * np.cos(angle)
    mapped_z[mask] = l2 + radial * np.sin(angle)
    mask = (z >= -1.0) & (z <= 0.0)
    angle = 1.5 * np.pi - 0.5 * np.pi * (z[mask] + 1.0)
    radial = xmax + radius - x[mask]
    mapped_x[mask] = xmax + radius + radial * np.cos(angle)
    mapped_z[mask] = radial * np.sin(angle)
    mask = (z >= -l1 - 1.0) & (z <= -1.0)
    mapped_x[mask] = xmax + radius - (z[mask] + 1.0)
    mapped_z[mask] = -radius - xmax + x[mask]
    return mapped_x, mapped_z


def make_seawall_mesh(pde):
    geometry = str(Path(__file__).with_name("seawall"))
    points2d, elements2d = exasim.Mesh.gmshcall(pde, geometry, 2, 1)[0:2]
    elements2d = elements2d[::-1, :]
    l1, radius, l2, l3 = 41.0, 4.0, 210.0, 41.0
    counts = [80, 20, 400, 20, 80]
    pieces = [
        np.linspace(0.0, l1, counts[0]),
        np.linspace(l1, l1 + 1.0, counts[1]),
        np.linspace(l1 + 1.0, l1 + 1.0 + l2, counts[2]),
        np.linspace(l1 + 1.0 + l2, l1 + 2.0 + l2, counts[3]),
        np.linspace(l1 + 2.0 + l2, l1 + 2.0 + l2 + l3, counts[4]),
    ]
    z = np.concatenate([pieces[0]] + [piece[1:] for piece in pieces[1:]]) - (l1 + 1.0)
    points, elements = _extrude_quads(points2d, elements2d, z)
    dgnodes = createdgnodes(
        points, elements, np.zeros((6, elements.shape[1]), dtype=np.int64),
        [], [], pde["porder"]
    )
    x0, z0 = points[0, :].copy(), points[2, :].copy()
    xmax = np.max(x0)
    points[0, :], points[2, :] = _map_xz(x0, z0, xmax, l1, radius, l2, l3)
    dgnodes[:, 0, :], dgnodes[:, 2, :] = _map_xz(
        dgnodes[:, 0, :], dgnodes[:, 2, :], xmax, l1, radius, l2, l3
    )
    return points, elements, dgnodes
