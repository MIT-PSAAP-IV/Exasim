"""Model 1 of the 3D coupled Poisson examples (pdemodel1.m)."""

from numpy import array
from sympy import exp, pi, sin


def flux(u, q, w, v, x, t, mu, eta):
    return mu[0] * q


def source(u, q, w, v, x, t, mu, eta):
    return array([(2 * pi * pi - 1) * exp(x[0]) * sin(pi * x[1]) * sin(pi * x[2])])


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    f = flux(u, q, w, v, x, t, mu, eta)
    return array([f[0] * n[0] + f[1] * n[1] + f[2] * n[2] + tau[0] * (u[0] - uhat[0])])


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return array([exp(x[0]) * sin(pi * x[1]) * sin(pi * x[2])])


def initu(x, mu, eta):
    return array([0.0])


def fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    # Column 1: Dirichlet boundary (condition 1); column 2: the coupled interface.
    ub = exp(x[0]) * sin(pi * x[1]) * sin(pi * x[2])
    f = flux(u, q, w, v, x, t, mu, eta)
    fi = f[0] * n[0] + f[1] * n[1] + f[2] * n[2] + tau[0] * (u[0] - uhat[0])
    return array([ub - uhat[0], fi])


def fint(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    # Model 1 sends its trace; model 2 enforces uhat2 = uhat1.
    return array([-uhat[0]])
