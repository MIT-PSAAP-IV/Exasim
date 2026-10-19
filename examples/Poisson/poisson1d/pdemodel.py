import numpy as np


def flux(u, q, w, v, x, t, mu, eta):
    return mu[0] * q


def source(u, q, w, v, x, t, mu, eta):
    return np.array([-4.0 * (x[0] - 0.5) ** 2 + 1.0])


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return np.array([[0.0, mu[0] * q[0]]])


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return np.array([[0.0, u[0]]])


def fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    exact = x[0] ** 4 / 3.0 - 2.0 * x[0] ** 3 / 3.0
    return np.array([tau[0] * (exact - uhat[0])])


def initu(x, mu, eta):
    return np.array([0.0 * x[0] ** 2])
