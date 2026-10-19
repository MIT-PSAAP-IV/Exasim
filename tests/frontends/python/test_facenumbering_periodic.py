"""Regression checks for Python periodic face numbering."""

import os
import sys

try:
    import numpy as np
except ImportError:
    print("SKIP: numpy unavailable")
    sys.exit(77)


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "frontends", "Python"))

from exasim.Mesh.squaremesh import SquareMesh  # noqa: E402
from exasim.Preprocessing.facenumbering import facenumbering  # noqa: E402
from exasim.Preprocessing.getelemface import getelemface  # noqa: E402
from exasim.Preprocessing.mkf2e import mkf2e  # noqa: E402


BOUNDARY_EXPR = [
    lambda p: p[1, :] < 1.0e-12,
    lambda p: p[0, :] > 1.0 - 1.0e-12,
    lambda p: p[1, :] > 1.0 - 1.0e-12,
    lambda p: p[0, :] < 1.0e-12,
]
PERIODIC_EXPR = [
    [2, lambda p: p[1, :], 4, lambda p: p[1, :]],
    [1, lambda p: p[0, :], 3, lambda p: p[0, :]],
]


def face_data(p, t, elemtype):
    face = getelemface(2, elemtype)
    nodes = np.reshape(
        t[face.flatten(order="F"), :],
        (face.shape[0], face.shape[1], t.shape[1]),
        order="F",
    )
    points = np.reshape(
        p[:, nodes.flatten(order="F")],
        (p.shape[0], face.shape[0], face.shape[1], t.shape[1]),
        order="F",
    )
    return nodes, points, np.mean(points, axis=1)


def boundary_labels(points):
    labels = np.zeros(points.shape[2:], dtype=int, order="F")
    for local_face in range(points.shape[2]):
        for element in range(points.shape[3]):
            vertices = points[:, :, local_face, element]
            for boundary, expression in enumerate(BOUNDARY_EXPR, start=1):
                if np.all(expression(vertices)):
                    labels[local_face, element] = boundary
                    break
    return labels


def check_nonperiodic(elemtype, n):
    p, t = SquareMesh(n, n, elemtype)
    _, points, centers = face_data(p, t, elemtype)
    labels = boundary_labels(points)
    f, tprd, t2t = facenumbering(p, t, elemtype, BOUNDARY_EXPR, [])
    expected_t2t = mkf2e(t, elemtype, 2)[1]
    assert np.array_equal(f, labels)
    assert np.array_equal(tprd, t)
    assert np.array_equal(t2t, expected_t2t)
    assert centers.shape[1:] == f.shape


def check_periodic(elemtype, n):
    p, t = SquareMesh(n, n, elemtype)
    nodes, points, centers = face_data(p, t, elemtype)
    labels = boundary_labels(points)
    f, tprd, t2t = facenumbering(
        p, t, elemtype, BOUNDARY_EXPR, PERIODIC_EXPR
    )

    expected_f = labels.copy()
    expected_f[expected_f > 0] *= -1
    assert np.array_equal(f, expected_f)

    for local_face, element in zip(*np.nonzero(labels)):
        boundary = labels[local_face, element]
        opposite = {1: 3, 2: 4, 3: 1, 4: 2}[boundary]
        tangent = 0 if boundary in (1, 3) else 1
        candidates = np.argwhere(labels == opposite)
        differences = [
            abs(centers[tangent, lf, elem] - centers[tangent, local_face, element])
            for lf, elem in candidates
        ]
        match = candidates[int(np.argmin(differences))]
        assert min(differences) < 1.0e-12
        assert t2t[local_face, element] == match[1]

    representatives = {}
    for element in range(t.shape[1]):
        for local_node in range(t.shape[0]):
            original = t[local_node, element]
            representative = tprd[local_node, element]
            delta = p[:, original] - p[:, representative]
            assert np.allclose(delta, np.round(delta), rtol=0.0, atol=1.0e-12)
            key = tuple(np.round(np.mod(p[:, original], 1.0), 12))
            representatives.setdefault(key, set()).add(int(representative))
    assert all(len(values) == 1 for values in representatives.values())


def main():
    for elemtype in (0, 1):
        for n in (1, 2, 8):
            check_nonperiodic(elemtype, n)
            check_periodic(elemtype, n)
    print("PASS: test_facenumbering_periodic.py")


if __name__ == "__main__":
    main()
