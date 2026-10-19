# This model is intentionally identical to ../concentration1d/pdemodel.py.
import numpy as np


def mass(u, q, w, v, x, t, mu, eta):
    return 1.0


def flux(u, q, w, v, x, t, mu, eta):
    return mu[0] * q


def source(u, q, w, v, x, t, mu, eta):
    return -mu[1] * u


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return 0.0


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return 0.0


def initu(x, mu, eta):
    return 1.0e-7


def fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    prescribed = tau[0] * (mu[2] - uhat[0])
    natural = mu[0] * q[0] + tau[0] * (u[0] - uhat[0])
    return np.array([[prescribed, natural]])
