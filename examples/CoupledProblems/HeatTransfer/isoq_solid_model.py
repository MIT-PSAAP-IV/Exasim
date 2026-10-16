"""Axisymmetric solid heat equation shared by the ISOQ examples."""

from solid_heat_model import fbou, fbouhdg, fint, flux, initu, mass, ubou
import numpy as np


def source(u, q, w, v, x, t, mu, eta):
    return np.array([-mu[0] * q[1] / x[1]])
