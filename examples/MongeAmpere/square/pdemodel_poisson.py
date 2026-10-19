"""Poisson stage for the density-equalizing mesh map."""

import numpy as np


def flux(u, q, w, v, x, t, mu, eta):
    return mu[0] * q


def source(u, q, w, v, x, t, mu, eta):
    return np.array([v[0] - 1.0])


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
