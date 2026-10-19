"""Poisson stage for the analytic square monitor function."""

import numpy as np
from sympy import cosh, sqrt


def flux(u, q, w, v, x, t, mu, eta):
    return mu[0] * q


def source(u, q, w, v, x, t, mu, eta):
    radius = sqrt(x[0] ** 2 + x[1] ** 2)
    density = 1.0 + mu[2] / cosh(mu[3] * (radius**2 - mu[4] ** 2))
    return np.array([density / mu[1] - 1.0])


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    f = flux(u, q, w, v, x, t, mu, eta)
    return np.array([np.dot(f, n) + tau[0] * u[0]])


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return np.array([0.0])


def fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    f = flux(u, q, w, v, x, t, mu, eta)
    return np.array([[np.dot(f, n) + tau[0] * (u[0] - uhat[0])]])


def initu(x, mu, eta):
    return np.array([0.0])
