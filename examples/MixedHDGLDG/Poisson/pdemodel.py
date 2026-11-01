"""Unit-square Poisson model; q = -grad(u), physicsparam = [kappa, ub]."""
from numpy import array
from sympy import pi, sin


def flux(u, q, w, v, x, t, mu, eta):
    return mu[0] * q


def source(u, q, w, v, x, t, mu, eta):
    return array([2 * pi**2 * sin(pi*x[0]) * sin(pi*x[1])])


def mass(u, q, w, v, x, t, mu, eta):
    return array([1.0])


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return array([mu[1]])


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    f = flux(u, q, w, v, x, t, mu, eta)
    return array([f[0]*n[0] + f[1]*n[1] + tau[0]*(u[0]-mu[1])])


def fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return array([tau[0]*(mu[1]-uhat[0])])


def initu(x, mu, eta):
    return array([0.0])
