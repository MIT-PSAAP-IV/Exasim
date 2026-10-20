import numpy as np


def flux(u, q, w, v, x, t, mu, eta):
    gradient = np.reshape(q, (3, 3), order="F")
    return mu[0] * (gradient + gradient.T) + mu[1] * np.trace(gradient) * np.eye(3)


def source(u, q, w, v, x, t, mu, eta):
    return 0.0 * x


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return 0.0 * x


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return 0.0 * x


def initu(x, mu, eta):
    return 0.0 * x


def fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    stress = flux(u, q, w, v, x, t, mu, eta)
    numerical_traction = stress @ n + tau[0] * (u - uhat)
    fixed_displacement = -uhat
    applied_traction = numerical_traction + mu[2:5]
    return np.column_stack((numerical_traction, fixed_displacement, applied_traction))
