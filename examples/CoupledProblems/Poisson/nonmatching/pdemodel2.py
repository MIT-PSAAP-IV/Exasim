"""Model 2 of the non-matching coupled Poisson example (pdemodel2.m)."""

from numpy import array
from sympy import pi, sin


def flux(u, q, w, v, x, t, mu, eta):
    return mu[0] * q


def source(u, q, w, v, x, t, mu, eta):
    return array([(2 * pi * pi) * sin(pi * x[0]) * sin(pi * x[1])])


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    f = flux(u, q, w, v, x, t, mu, eta)
    return array([f[0] * n[0] + f[1] * n[1] + tau[0] * (u[0] - 0.0)])


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return array([0.0])


def initu(x, mu, eta):
    return array([0.0])


def fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    # Column 1: boundary condition 1 (u = 0); column 2: the coupled interface.
    return array([0 - uhat[0], uhat[0]])


def fint(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    f = flux(u, q, w, v, x, t, mu, eta)
    return array([f[0] * n[0] + f[1] * n[1] + tau[0] * (u[0] - uhat[0])])
