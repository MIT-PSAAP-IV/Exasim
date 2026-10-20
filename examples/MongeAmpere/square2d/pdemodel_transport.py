"""Transport stage for the analytic square monitor function."""

import numpy as np
from sympy import cosh, sinh, sqrt


def _monitor(x, mu):
    radius = sqrt(x[0] ** 2 + x[1] ** 2)
    argument = mu[3] * (radius**2 - mu[4] ** 2)
    density = 1.0 + mu[2] / cosh(argument)
    scale = -2.0 * mu[2] * mu[3] * sinh(argument) / cosh(argument) ** 2
    return density, scale * x[0], scale * x[1]


def mass(u, q, w, v, x, t, mu, eta):
    return np.array([1.0])


def flux(u, q, w, v, x, t, mu, eta):
    density, _, _ = _monitor(x, mu)
    monitor = density / mu[1]
    denominator = t + (1.0 - t) * monitor
    return np.array([[v[0] * u[0] / denominator + mu[0] * (density - 1.0) * q[0],
                      v[1] * u[0] / denominator + mu[0] * (density - 1.0) * q[1]]])


def source(u, q, w, v, x, t, mu, eta):
    density, density_x, density_y = _monitor(x, mu)
    monitor = density / mu[1]
    denominator = t + (1.0 - t) * monitor
    velocity_x, velocity_y = v[:2]
    derivative = (velocity_x * density_x + velocity_y * density_y) / mu[1]
    return np.array([u[0] * (monitor - 1.0 - (1.0 - t) * derivative / denominator) / denominator])


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return np.array([np.dot(flux(u, q, w, v, x, t, mu, eta)[0, :], n) + tau[0] * (u[0] - uhat[0])])


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return np.array([0.0])


def fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return np.array([[u[0] - uhat[0]]])


def initu(x, mu, eta):
    return np.array([x[0]])
