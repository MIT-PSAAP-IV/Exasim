import numpy as np
from sympy import exp, log


def mass(u, q, w, v, x, t, mu, eta):
    return np.array([1.0])


def flux(u, q, w, v, x, t, mu, eta):
    return np.array([mu[13] * q[0]])


def source(u, q, w, v, x, t, mu, eta):
    concentration = (exp(mu[16] * x[0]) + exp(2.0 * mu[16] - mu[16] * x[0])) / (
        1.0 + exp(2.0 * mu[16])
    )
    temperature = u[0] * mu[2]
    chemical_potential = mu[7] + mu[3] * temperature * log(concentration)
    rate = mu[11] * (mu[5] * mu[6] / mu[14] + mu[8] * mu[5] * chemical_potential / mu[14]) / mu[2]
    return np.array([rate * concentration])


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return 0.0


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return 0.0


def initu(x, mu, eta):
    return np.array([mu[1] / mu[2]])


def fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    physical_flux = flux(u, q, w, v, x, t, mu, eta)[0]
    applied_flux = physical_flux * n[0] + tau[0] * (u[0] - uhat[0]) + mu[15]
    prescribed = tau[0] * (mu[1] / mu[2] - uhat[0])
    return np.array([[applied_flux, prescribed]])
