"""Two-block Sharp-B mesh for the Python mesh-adaptivity driver."""

from __future__ import annotations

import os
import sys

import numpy as np

_ISOQ_DIRECTORY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "isoq2d_idealgas"
)
if _ISOQ_DIRECTORY not in sys.path:
    sys.path.append(_ISOQ_DIRECTORY)
from isoq_mesh import _connect_blocks, _surfmesh2d, wall_distance  # noqa: E402

from exasim.Preprocessing.facenumbering import facenumbering  # noqa: E402


def _circle_tangent_point(center_x, center_y, radius, point_x, point_y):
    vector = np.array([point_x - center_x, point_y - center_y], dtype=float)
    distance = np.linalg.norm(vector)
    unit = vector / distance
    perpendicular = np.array([-unit[1], unit[0]])
    ratio = radius / distance
    return np.array([center_x, center_y]) + radius * (
        ratio * unit + np.sqrt(1.0 - ratio**2) * perpendicular
    )


def _sharpb2_curves():
    radius, nose_radius, length = 0.27, 0.0415, 1.81
    tangent = _circle_tangent_point(nose_radius, 0.0, nose_radius, length, radius)
    theta = np.linspace(0.0, np.arcsin(tangent[1] / nose_radius), 400)
    nose = np.column_stack(
        (nose_radius * (1.0 - np.cos(theta)), nose_radius * np.sin(theta))
    )
    cone_x = np.linspace(tangent[0], length, 400)
    cone_y = tangent[1] + (radius - tangent[1]) * (
        cone_x - tangent[0]
    ) / (length - tangent[0])
    lower = np.vstack((nose, np.column_stack((cone_x[1:], cone_y[1:]))))

    outer_radius = 0.17
    theta = np.linspace(np.pi, np.pi / 1.75, 400)
    outer_nose = np.column_stack(
        (0.14 + outer_radius * np.cos(theta), outer_radius * np.sin(theta))
    )
    outer_x = np.linspace(outer_nose[-1, 0], length, 400)
    outer_y = outer_nose[-1, 1] + (2.0 * radius - outer_nose[-1, 1]) * (
        outer_x - outer_nose[-1, 0]
    ) / (length - outer_nose[-1, 0])
    upper = np.vstack((outer_nose, np.column_stack((outer_x, outer_y))))
    return lower, upper, tangent


def make_sharpb2_mesh(porder, radial_shift=1.0e-4):
    lower, upper, tangent = _sharpb2_curves()
    lower_first = lower[lower[:, 0] <= tangent[0]]
    lower_second = np.vstack(
        (lower_first[-1], lower[(lower[:, 0] > tangent[0]) & (lower[:, 0] <= 1.25)])
    )
    upper_first = upper[upper[:, 0] <= 0.0]
    upper_second = np.vstack(
        (upper_first[-1], upper[(upper[:, 0] > 0.0) & (upper[:, 0] <= 1.22)])
    )

    first = _surfmesh2d(
        lower_first, upper_first, 36, 100, porder,
        streamwise_scaling=(2.0, 1.2), normal_scaling=(5.5, 0.0),
    )
    second = _surfmesh2d(
        lower_second, upper_second, 120, 100, porder,
        streamwise_scaling=(3.5, 1.5), normal_scaling=(5.5, 0.0),
    )
    points, elements, dgnodes = _connect_blocks(first, second)
    points[1, :] += radial_shift
    dgnodes[:, 1, :] += radial_shift

    minimum_x, minimum_y = np.min(points[0, :]), np.min(points[1, :])
    maximum_y_index = int(np.argmax(points[1, :]))
    x_at_maximum_y, maximum_y = (
        points[0, maximum_y_index], points[1, maximum_y_index]
    )
    maximum_x_index = int(np.argmax(points[0, :]))
    y_at_maximum_x = points[1, maximum_x_index]
    transition_x, transition_y = 0.0, 0.04
    boundaryexpr = [
        lambda p: np.abs(p[1, :] - minimum_y) < 1.0e-6,
        lambda p: p[1, :] > minimum_y + (transition_y - minimum_y) /
            (transition_x - minimum_x) * (p[0, :] - minimum_x) - 1.0e-4,
        lambda p: p[1, :] > transition_y + (maximum_y - transition_y) /
            (x_at_maximum_y - transition_x) * (p[0, :] - transition_x) - 1.0e-4,
        lambda p: p[1, :] < y_at_maximum_x + 1.0e-4,
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
        "boundarycondition": np.array([5, 1, 1, 3, 2], dtype=np.int64),
        "periodicexpr": [],
        "curvedboundary": np.array([], dtype=np.int64),
        "curvedboundaryexpr": [],
        "periodicboundary": np.array([], dtype=np.int64),
    }


__all__ = ["make_sharpb2_mesh", "wall_distance"]
