import numpy as np
from sympy import exp


def flux(u, q, w, v, x, t, mu, eta):
    gradient = np.reshape(q, (2, 2), order="F")
    return mu[0] * (gradient + gradient.T) + mu[1] * np.trace(gradient) * np.eye(2)


def source(u, q, w, v, x, t, mu, eta):
    return 0.0 * x


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return 0.0 * x


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return 0.0 * x


def initu(x, mu, eta):
    return 0.0 * x


def fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    prescribed = 0.0 * uhat
    prescribed[1] = mu[2] * exp(-((x[0] - mu[3]) ** 2) / (2.0 * mu[4] ** 2))
    fixed = -tau[0] * uhat
    bottom = tau[0] * (prescribed - uhat)
    return np.column_stack((fixed, bottom))
