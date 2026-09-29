"""Two-block ISOQ mesh shared by the Python verification driver."""

from __future__ import annotations

import numpy as np

from exasim.Preprocessing.createdgnodes import createdgnodes
from exasim.Preprocessing.facenumbering import facenumbering
from exasim.Preprocessing.masternodes import masternodes


def _loginc(values, alpha):
    lower, upper = np.min(values), np.max(values)
    if upper == lower:
        return np.array(values, copy=True)
    return lower + (upper - lower) * (
        np.exp(alpha * (values - lower) / (upper - lower)) - 1.0
    ) / (np.exp(alpha) - 1.0)


def _logdec(values, alpha):
    lower, upper = np.min(values), np.max(values)
    if upper == lower:
        return np.array(values, copy=True)
    return lower + (upper - lower) * (
        1.0 - np.exp(-alpha * (values - lower) / (upper - lower))
    ) / (1.0 - np.exp(-alpha))


def _rectangular_quad_mesh(nx, ny, porder):
    x = np.tile(np.linspace(0.0, 1.0, nx + 1), ny + 1)
    y = np.repeat(np.linspace(0.0, 1.0, ny + 1), nx + 1)
    points = np.vstack((x, y))
    elements = np.empty((4, nx * ny), dtype=np.int64)
    element = 0
    for iy in range(ny):
        for ix in range(nx):
            lower_left = ix + iy * (nx + 1)
            elements[:, element] = [
                lower_left,
                lower_left + 1,
                lower_left + nx + 2,
                lower_left + nx + 1,
            ]
            element += 1
    dgnodes = createdgnodes(
        points, elements, np.zeros((4, elements.shape[1]), dtype=np.int64),
        [], [], porder,
    )
    return points, elements, dgnodes


def _inverse_bilinear(point, coefficients, initial):
    xi, eta = initial
    for _ in range(100):
        basis = np.array([1.0, xi, eta, xi * eta])
        residual = basis @ coefficients - point
        jacobian = np.array(
            [
                [coefficients[1, 0] + coefficients[3, 0] * eta,
                 coefficients[2, 0] + coefficients[3, 0] * xi],
                [coefficients[1, 1] + coefficients[3, 1] * eta,
                 coefficients[2, 1] + coefficients[3, 1] * xi],
            ]
        )
        update = np.linalg.solve(jacobian, -residual)
        xi += update[0]
        eta += update[1]
        if np.linalg.norm(update) < 1.0e-10:
            return np.array([xi, eta])
    raise RuntimeError("inverse bilinear map did not converge")


def _forward_bilinear(reference_points, coefficients):
    xi, eta = reference_points[:, 0], reference_points[:, 1]
    basis = np.column_stack((np.ones(xi.size), xi, eta, xi * eta))
    return basis @ coefficients


def _surfmesh2d(
    lower_curve, upper_curve, nx, ny, porder,
    streamwise_scaling=(2.0, 1.5), normal_scaling=(3.0, 1.0e-8),
):
    matrix = np.array(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0],
         [1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0]]
    )
    corners = np.vstack(
        (lower_curve[0], lower_curve[-1], upper_curve[-1], upper_curve[0])
    )
    coefficients = np.linalg.solve(matrix, corners)

    lower_reference = np.vstack(
        [
            _inverse_bilinear(point, coefficients, (i / (len(lower_curve) - 1), 0.0))
            for i, point in enumerate(lower_curve)
        ]
    )
    upper_reference = np.vstack(
        [
            _inverse_bilinear(point, coefficients, (i / (len(upper_curve) - 1), 1.0))
            for i, point in enumerate(upper_curve)
        ]
    )
    lower_fit = np.polyfit(lower_reference[:, 0], lower_reference[:, 1],
                           min(len(lower_curve) - 1, 12))
    upper_fit = np.polyfit(upper_reference[:, 0], upper_reference[:, 1],
                           min(len(upper_curve) - 1, 12))

    points, elements, dgnodes = _rectangular_quad_mesh(nx, ny, porder)
    vertex_reference = points.T.copy()
    node_reference = np.column_stack(
        (
            dgnodes[:, 0, :].reshape(-1, order="F"),
            dgnodes[:, 1, :].reshape(-1, order="F"),
        )
    )
    sides = {
        "vertex_left": np.abs(vertex_reference[:, 0]) < 1.0e-6,
        "vertex_right": np.abs(vertex_reference[:, 0] - 1.0) < 1.0e-6,
        "node_left": np.abs(node_reference[:, 0]) < 1.0e-6,
        "node_right": np.abs(node_reference[:, 0] - 1.0) < 1.0e-6,
    }

    for reference in (vertex_reference, node_reference):
        reference[:, 0] = _logdec(
            _loginc(reference[:, 0], streamwise_scaling[0]),
            max(streamwise_scaling[1], 1.0e-8),
        )
        reference[:, 1] = _logdec(
            _loginc(reference[:, 1], normal_scaling[0]),
            max(normal_scaling[1], 1.0e-8),
        )
        lower = np.polyval(lower_fit, reference[:, 0])
        upper = np.polyval(upper_fit, reference[:, 0])
        endpoints = (np.abs(reference[:, 0]) < 1.0e-6) | (
            np.abs(reference[:, 0] - 1.0) < 1.0e-6
        )
        lower[endpoints] = 0.0
        upper[endpoints] = 1.0
        reference[:, 1] = lower + (upper - lower) * reference[:, 1]

    physical_points = _forward_bilinear(vertex_reference, coefficients)
    physical_nodes = _forward_bilinear(node_reference, coefficients)
    dgnodes[:, 0, :] = physical_nodes[:, 0].reshape(
        dgnodes.shape[0], dgnodes.shape[2], order="F"
    )
    dgnodes[:, 1, :] = physical_nodes[:, 1].reshape(
        dgnodes.shape[0], dgnodes.shape[2], order="F"
    )
    return physical_points.T, elements, dgnodes, sides


def _isoq_curves():
    nose_radius = 0.102
    shoulder_radius = nose_radius / 16.0
    end_x = 0.06
    nose_angle = 27.82 * np.pi / 180.0

    theta = np.linspace(0.0, nose_angle, 481)
    nose = np.column_stack(
        (nose_radius * (1.0 - np.cos(theta)), nose_radius * np.sin(theta))
    )
    initial = np.pi - nose_angle
    center = nose[-1] - shoulder_radius * np.array([np.cos(initial), np.sin(initial)])
    theta = np.linspace(initial, np.pi / 2.0, 49)
    shoulder = center + shoulder_radius * np.column_stack((np.cos(theta), np.sin(theta)))
    afterbody = np.column_stack(
        (np.linspace(shoulder[-1, 0], end_x, 240),
         np.full(240, shoulder[-1, 1]))
    )
    lower = np.vstack((nose[:-1], shoulder[:-1], afterbody))

    theta = np.linspace(np.pi, np.pi / 2.0, 2000)
    upper = np.column_stack(
        (3.7 * end_x + 4.4 * end_x * np.cos(theta),
         3.6 * end_x * np.sin(theta))
    )
    upper = upper[upper[:, 0] <= end_x]
    upper[-1, 0] = end_x
    return lower, upper


def _connect_blocks(first, second, tolerance=1.0e-5):
    p1, t1, dgnodes1, side1 = first
    p2, t2, dgnodes2, side2 = second
    average_vertices = 0.5 * (
        p1[:, side1["vertex_right"]].T + p2[:, side2["vertex_left"]].T
    )
    p1[:, side1["vertex_right"]] = average_vertices.T
    p2[:, side2["vertex_left"]] = average_vertices.T

    flat1 = np.column_stack(
        (dgnodes1[:, 0, :].reshape(-1, order="F"),
         dgnodes1[:, 1, :].reshape(-1, order="F"))
    )
    flat2 = np.column_stack(
        (dgnodes2[:, 0, :].reshape(-1, order="F"),
         dgnodes2[:, 1, :].reshape(-1, order="F"))
    )
    average_nodes = 0.5 * (
        flat1[side1["node_right"]] + flat2[side2["node_left"]]
    )
    flat1[side1["node_right"]] = average_nodes
    flat2[side2["node_left"]] = average_nodes
    for flat, nodes in ((flat1, dgnodes1), (flat2, dgnodes2)):
        nodes[:, 0, :] = flat[:, 0].reshape(nodes.shape[0], nodes.shape[2], order="F")
        nodes[:, 1, :] = flat[:, 1].reshape(nodes.shape[0], nodes.shape[2], order="F")

    mapping = np.empty(p2.shape[1], dtype=np.int64)
    keep = np.ones(p2.shape[1], dtype=bool)
    next_index = p1.shape[1]
    for index in range(p2.shape[1]):
        squared_distance = np.sum((p1.T - p2[:, index]) ** 2, axis=1)
        nearest = int(np.argmin(squared_distance))
        if squared_distance[nearest] < tolerance**2:
            mapping[index] = nearest
            keep[index] = False
        else:
            mapping[index] = next_index
            next_index += 1
    points = np.hstack((p1, p2[:, keep]))
    elements = np.hstack((t1, mapping[t2]))
    dgnodes = np.concatenate((dgnodes1, dgnodes2), axis=2)
    return points, elements, dgnodes


def make_isoq_mesh(porder, radial_shift=5.0e-4):
    lower, upper = _isoq_curves()
    upper_first = upper[upper[:, 0] <= -0.02]
    upper_second = np.vstack((upper_first[-1], upper[upper[:, 0] > -0.02]))
    lower_first = lower[lower[:, 0] <= 0.013]
    lower_second = np.vstack((lower_first[-1], lower[lower[:, 0] > 0.013]))

    first = _surfmesh2d(lower_first, upper_first, 30, 60, porder)
    second = _surfmesh2d(lower_second, upper_second, 25, 60, porder)
    points, elements, dgnodes = _connect_blocks(first, second)
    points[1, :] += radial_shift
    dgnodes[:, 1, :] += radial_shift

    lower_y = np.min(points[1, :])
    right_x = np.max(points[0, :])
    boundaryexpr = [
        lambda p: np.abs(p[1, :] - lower_y) < 1.0e-6,
        lambda p: p[0, :] > right_x - 1.0e-4,
        lambda p: (p[0, :] < -1.0e-3) | (p[1, :] > 0.1),
        lambda p: np.abs(p[0, :]) < 20.0 + 1.0e-6,
    ]
    faces = facenumbering(points, elements, 1, boundaryexpr, [])[0]
    return {
        "p": points,
        "t": elements,
        "f": faces,
        "dgnodes": dgnodes,
        "udg": [],
        "vdg": [],
        "wdg": [],
        "tprd": [],
        "boundaryexpr": boundaryexpr,
        "boundarycondition": np.array([5, 2, 1, 3], dtype=np.int64),
        "periodicexpr": [],
        "curvedboundary": np.array([], dtype=np.int64),
        "curvedboundaryexpr": [],
        "periodicboundary": np.array([], dtype=np.int64),
    }


def wall_distance(mesh, porder):
    perm = np.asarray(masternodes(porder, 2, 1)[4], dtype=np.int64) - 1
    local_faces, elements = np.nonzero(mesh["f"] == 4)
    wall_nodes = np.concatenate(
        [mesh["dgnodes"][perm[:, face], :, elem]
         for face, elem in zip(local_faces, elements)], axis=0
    )
    npe, _, ne = mesh["dgnodes"].shape
    distance = np.empty((npe, 1, ne), dtype=np.float64, order="F")
    for elem in range(ne):
        delta = mesh["dgnodes"][:, :, elem, None] - wall_nodes.T[None, :, :]
        distance[:, 0, elem] = np.sqrt(np.sum(delta * delta, axis=1)).min(axis=1)
    return distance
