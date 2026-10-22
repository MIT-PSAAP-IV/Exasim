"""Poisson model shared by all four physical domains."""
from numpy import array
from sympy import pi, sin


def flux(u, q, w, v, x, t, mu, eta):
    return mu[0] * q


def source(u, q, w, v, x, t, mu, eta):
    return array([3*pi*pi*sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[2])])


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    f = flux(u, q, w, v, x, t, mu, eta)
    return array([f[0]*n[0] + f[1]*n[1] + f[2]*n[2] + tau[0]*(u[0]-uhat[0])])


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return array([sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[2])])


def initu(x, mu, eta):
    return array([0.0])


def fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    f = flux(u, q, w, v, x, t, mu, eta)
    normal_flux = f[0]*n[0] + f[1]*n[1] + f[2]*n[2] + tau[0]*(u[0]-uhat[0])
    boundary_value = sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[2])
    interface_equation = mu[1]*normal_flux + (1-mu[1])*uhat[0]
    return array([boundary_value-uhat[0], interface_equation, interface_equation])


def fint(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    """Columns 1 and 2 impose solution continuity and normal-flux balance."""
    f = flux(u, q, w, v, x, t, mu, eta)
    normal_flux = f[0]*n[0] + f[1]*n[1] + f[2]*n[2] + tau[0]*(u[0]-uhat[0])
    return array([-uhat[0], normal_flux])
