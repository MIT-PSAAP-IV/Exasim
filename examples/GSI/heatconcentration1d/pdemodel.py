import numpy as np
from sympy import atan, exp, log, pi


def _lmax(value, alpha):
    return value * (atan(alpha * value) / pi + 0.5) - atan(alpha) / pi + 0.5


def mass(u, q, w, v, x, t, mu, eta):
    return np.array([1.0, 1.0])


def flux(u, q, w, v, x, t, mu, eta):
    temperature = u[0] * mu[2]
    diffusivity = mu[9] * exp(-mu[10] / (mu[4] * temperature))
    concentration_diffusivity = mu[11] * diffusivity / (mu[0] * mu[0])
    return np.array([[mu[13] * q[0]], [concentration_diffusivity * q[1]]])


def source(u, q, w, v, x, t, mu, eta):
    concentration = _lmax(u[1], 1.0e3)
    temperature = u[0] * mu[2]
    chemical_potential = mu[7] + mu[3] * temperature * log(concentration)
    heat_rate = mu[11] * (
        mu[5] * mu[6] / mu[14] + mu[8] * mu[5] * chemical_potential / mu[14]
    ) / mu[2]
    reaction_rate = mu[5] * mu[11]
    return np.array([heat_rate * concentration, -reaction_rate * concentration])


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return np.zeros(2)


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return np.zeros(2)


def initu(x, mu, eta):
    return np.array([mu[1] / mu[2], exp(-1.0e4 * x[0] * x[0])])


def fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    physical_flux = flux(u, q, w, v, x, t, mu, eta)
    left = np.array([
        physical_flux[0, 0] * n[0] + tau[0] * (u[0] - uhat[0]) + mu[15],
        tau[0] * (1.0 - uhat[1]),
    ])
    right = np.array([
        tau[0] * (mu[1] / mu[2] - uhat[0]),
        physical_flux[1, 0] * n[0] + tau[0] * (u[1] - uhat[1]),
    ])
    return np.column_stack((left, right))
