import numpy as np


def flux(u, q, w, v, x, t, mu, eta):
    inviscid = np.array([u[0], 0.0, 0.0, u[1]])
    viscous = np.array([mu[0] * q[0], q[1], mu[0] * q[2], q[3]])
    return np.reshape(inviscid + viscous, (2, 2), order="F")


def source(u, q, w, v, x, t, mu, eta):
    return np.array([1.0, u[0]])


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    f = flux(u, q, w, v, x, t, mu, eta)
    return f[:, 0] * n[0] + f[:, 1] * n[1] + tau[0] * (u - uhat)


def fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return -tau[0] * uhat


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return np.zeros(2)


def initu(x, mu, eta):
    return np.zeros(2)
