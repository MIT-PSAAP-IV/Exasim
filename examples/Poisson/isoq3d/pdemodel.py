"""Three-dimensional Laplace model used by the ISOQ example."""

import numpy as np


def flux(u, q, w, v, x, t, mu, eta):
    return mu[0] * q


def source(u, q, w, v, x, t, mu, eta):
    return np.array([0.0 * u[0]])


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return np.array([0.0 * u[0]])


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return np.array([0.0 * u[0]])


def initu(x, mu, eta):
    return np.array([0.0 * x[0]])


def fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    dirichlet_zero = tau[0] * (mu[1] - uhat[0])
    dirichlet_one = tau[0] * (mu[2] - uhat[0])
    normal_flux = np.dot(flux(u, q, w, v, x, t, mu, eta), n)
    natural = normal_flux + tau[0] * (u[0] - uhat[0])
    return np.array([[dirichlet_zero, dirichlet_one, natural]])
