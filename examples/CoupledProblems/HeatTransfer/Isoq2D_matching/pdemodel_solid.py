"""Solid model (model 2): axisymmetric heat equation, y is the radial coordinate."""

import numpy as np


def mass(u, q, w, v, x, t, mu, eta):
    return np.ones(1)


def flux(u, q, w, v, x, t, mu, eta):
    return mu[0] * np.reshape(q, (1, 2), order="F")


def source(u, q, w, v, x, t, mu, eta):
    return np.array([-mu[0] * q[1] / x[1]])


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return np.array([0.0 * u[0]])


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return np.array([0.0 * u[0]])


def fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    f = flux(u, q, w, v, x, t, mu, eta)
    numerical_flux = f[0, 0] * n[0] + f[0, 1] * n[1] + tau[0] * (u[0] - uhat[0])
    fixed_temperature = tau[0] * (mu[1] - uhat[0])
    return np.array([[fixed_temperature, numerical_flux, numerical_flux]])


def fint(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return np.array([-uhat[0]])


def initu(x, mu, eta):
    return np.array([mu[1]])
