"""Compressible Navier--Stokes model for the periodic Taylor--Green vortex."""

import numpy as np
from sympy import cos, pi, sin


def mass(u, q, w, v, x, t, mu, eta):
    return np.ones(5)


def flux(u, q, w, v, x, t, mu, eta):
    gam, reynolds, prandtl, mach = mu[0:4]
    gam1 = gam - 1.0
    density, mx, my, mz, energy = u
    ux, uy, uz = mx / density, my / density, mz / density
    kinetic = 0.5 * (ux * ux + uy * uy + uz * uz)
    pressure = gam1 * (energy - density * kinetic)
    enthalpy = (energy + pressure) / density
    inviscid = np.array([
        mx, mx * ux + pressure, my * ux, mz * ux, mx * enthalpy,
        my, mx * uy, my * uy + pressure, mz * uy, my * enthalpy,
        mz, mx * uz, my * uz, mz * uz + pressure, mz * enthalpy,
    ])

    gradients = np.reshape(q, (5, 3), order="F")
    velocity_gradient = []
    temperature_gradient = []
    for direction in range(3):
        dr, dmx, dmy, dmz, de = gradients[:, direction]
        du = (dmx - dr * ux) / density
        dv = (dmy - dr * uy) / density
        dw = (dmz - dr * uz) / density
        dk = ux * du + uy * dv + uz * dw
        dp = gam1 * (de - dr * kinetic - density * dk)
        dtemperature = gam * mach**2 * (dp * density - pressure * dr) / density**2
        velocity_gradient.append((du, dv, dw))
        temperature_gradient.append(dtemperature)

    dux, dvx, dwx = velocity_gradient[0]
    duy, dvy, dwy = velocity_gradient[1]
    duz, dvz, dwz = velocity_gradient[2]
    inverse_reynolds = 1.0 / reynolds
    two_thirds = 2.0 / 3.0
    txx = inverse_reynolds * two_thirds * (2.0 * dux - dvy - dwz)
    txy = inverse_reynolds * (duy + dvx)
    txz = inverse_reynolds * (duz + dwx)
    tyy = inverse_reynolds * two_thirds * (2.0 * dvy - dux - dwz)
    tyz = inverse_reynolds * (dvz + dwy)
    tzz = inverse_reynolds * two_thirds * (2.0 * dwz - dux - dvy)
    conductivity = 1.0 / (gam1 * mach**2 * reynolds * prandtl)
    tx, ty, tz = temperature_gradient
    viscous = np.array([
        0.0, txx, txy, txz, ux * txx + uy * txy + uz * txz + conductivity * tx,
        0.0, txy, tyy, tyz, ux * txy + uy * tyy + uz * tyz + conductivity * ty,
        0.0, txz, tyz, tzz, ux * txz + uy * tyz + uz * tzz + conductivity * tz,
    ])
    return np.reshape(inviscid + viscous, (5, 3), order="F")


def source(u, q, w, v, x, t, mu, eta):
    return np.zeros(5)


def _normal_flux(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    physical_flux = flux(u, q, w, v, x, t, mu, eta)
    return physical_flux @ n + tau[0] * (u - uhat)


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return _normal_flux(u, q, w, v, x, t, mu, eta, uhat, n, tau)


def fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return _normal_flux(u, q, w, v, x, t, mu, eta, uhat, n, tau)


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return np.zeros(5)


def initu(x, mu, eta):
    gam, mach = mu[0], mu[3]
    x1, x2, x3 = x
    ux = sin(x1) * cos(x2) * cos(x3)
    uy = -cos(x1) * sin(x2) * cos(x3)
    uz = 0.0
    pressure = 1.0 / (gam * mach**2) + (cos(2.0 * x1) + cos(2.0 * x2)) * (cos(2.0 * x3) + 2.0) / 16.0
    density = 1.0
    return np.array([density, ux, uy, uz, 0.5 * (ux * ux + uy * uy) + pressure / (gam - 1.0)])
