"""Transport stage for the density-equalizing mesh map."""

import numpy as np


def mass(u, q, w, v, x, t, mu, eta):
    return np.array([1.0])


def flux(u, q, w, v, x, t, mu, eta):
    density, velocity_x, velocity_y = v[0], v[3], v[4]
    denominator = t + (1.0 - t) * density
    diffusion = mu[0] * (mu[1] * density - 1.0)
    return np.array([[velocity_x * u[0] / denominator + diffusion * q[0],
                      velocity_y * u[0] / denominator + diffusion * q[1]]])


def source(u, q, w, v, x, t, mu, eta):
    density, density_x, density_y = v[:3]
    velocity_x, velocity_y = v[3:5]
    denominator = t + (1.0 - t) * density
    divergence_term = (velocity_x * density_x + velocity_y * density_y) / denominator
    return np.array([u[0] * (density - 1.0 - (1.0 - t) * divergence_term) / denominator])


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    f = flux(u, q, w, v, x, t, mu, eta)
    return np.array([np.dot(f[0, :], n) + tau[0] * (u[0] - uhat[0])])


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return np.array([0.0])


def fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return np.array([[u[0] - uhat[0], x[0] - uhat[0], x[1] - uhat[0]]])


def initu(x, mu, eta):
    return np.array([x[0]])
