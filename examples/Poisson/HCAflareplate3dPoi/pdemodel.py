"""Three-dimensional Laplace model on the HCA flared-plate mesh."""

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
    zero = -uhat[0]
    one = 1.0 - uhat[0]
    normal_flux = np.dot(flux(uhat, q, w, v, x, t, mu, eta), n)
    natural = normal_flux + tau[0] * (u[0] - uhat[0])
    return np.array([[zero, one, natural]])
