import numpy as np


def make_plate_mesh(nx=120, ny=48, growth=1.2):
    x = np.linspace(-0.5, 0.5, nx + 1)
    layers = growth ** np.arange(ny + 1, dtype=float)
    y = 0.2 * (layers - 1.0) / (layers[-1] - 1.0)
    points = np.array([[xx, yy] for yy in y for xx in x], dtype=float).T
    elements = []
    for j in range(ny):
        for i in range(nx):
            lower_left = i + j * (nx + 1)
            elements.append([lower_left, lower_left + 1, lower_left + nx + 2, lower_left + nx + 1])
    return points, np.asarray(elements, dtype=int).T
