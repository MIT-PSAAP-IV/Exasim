"""Compressible Navier-Stokes model for the Mach-8 CHT examples."""

import numpy as np
from sympy import atan, pi, tanh


def _lmax(value, alpha):
    return value * (atan(alpha * value) / pi + 0.5) - atan(alpha) / pi + 0.5


def _viscosity(mu_ref, tref, temperature):
    return mu_ref * (temperature / tref) ** 1.5 * (tref + 110.4) / (temperature + 110.4)


def mass(u, q, w, v, x, t, mu, eta):
    return np.ones(4)


def flux(u, q, w, v, x, t, mu, eta):
    gam, reynolds, prandtl, mach_inf = mu[0:4]
    gam1 = gam - 1.0
    tref = mu[9]
    tinf = 1.0 / (gam * gam1 * mach_inf**2)
    alpha, rmin, pmin = 1.0e3, 1.0e-2, 1.0e-3
    r, ru, rv, rE = u
    rx, rux, rvx, rEx, ry, ruy, rvy, rEy = q

    r = rmin + _lmax(r - rmin, alpha)
    dr = atan(alpha * (r - rmin)) / pi + alpha * (r - rmin) / (
        pi * (alpha**2 * (r - rmin) ** 2 + 1.0)
    ) + 0.5
    rx, ry = rx * dr, ry * dr
    r1 = 1.0 / r
    ux, uy = ru * r1, rv * r1
    kinetic = 0.5 * (ux**2 + uy**2)
    pressure = gam1 * (rE - r * kinetic)
    pressure = pmin + _lmax(pressure - pmin, alpha)
    dp = atan(alpha * (pressure - pmin)) / pi + alpha * (pressure - pmin) / (
        pi * (alpha**2 * (pressure - pmin) ** 2 + 1.0)
    ) + 0.5
    enthalpy = (rE + pressure) * r1
    inviscid = np.array([
        ru, ru * ux + pressure, rv * ux, ru * enthalpy,
        rv, ru * uy, rv * uy + pressure, rv * enthalpy,
    ])

    du_dx = (rux - rx * ux) * r1
    dv_dx = (rvx - rx * uy) * r1
    dp_dx = gam1 * (rEx - rx * kinetic - r * (ux * du_dx + uy * dv_dx)) * dp
    dt_dx = (dp_dx * r - pressure * rx) * r1**2 / gam1
    du_dy = (ruy - ry * ux) * r1
    dv_dy = (rvy - ry * uy) * r1
    dp_dy = gam1 * (rEy - ry * kinetic - r * (ux * du_dy + uy * dv_dy)) * dp
    dt_dy = (dp_dy * r - pressure * ry) * r1**2 / gam1

    temperature = pressure / (gam1 * r)
    dynamic_viscosity = _viscosity(1.0 / reynolds, tref, tref * temperature / tinf)
    conductivity = dynamic_viscosity * gam / prandtl
    tau_xx = dynamic_viscosity * (2.0 / 3.0) * (2.0 * du_dx - dv_dy)
    tau_xy = dynamic_viscosity * (du_dy + dv_dx)
    tau_yy = dynamic_viscosity * (2.0 / 3.0) * (2.0 * dv_dy - du_dx)
    viscous = np.array([
        0.0, tau_xx, tau_xy, ux * tau_xx + uy * tau_xy + conductivity * dt_dx,
        0.0, tau_xy, tau_yy, ux * tau_xy + uy * tau_yy + conductivity * dt_dy,
    ])
    # artificial viscosity c*tanh(k*d): wall distance d in v[0], c = mu[11], k = mu[12]
    av = mu[11] * tanh(mu[12] * v[0])
    artificial = av * np.array([rx, rux, rvx, rEx, ry, ruy, rvy, rEy])
    return np.reshape(inviscid + viscous + artificial, (4, 2), order="F")


def source(u, q, w, v, x, t, mu, eta):
    return np.zeros(4)


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return np.zeros(4)


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return np.zeros(4)


def fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    t_wall = mu[10] / mu[9] * mu[8]
    uinf = np.array(mu[4:8])
    inflow = uinf - uhat
    outflow = u - uhat
    isothermal = 0 * u
    isothermal[0] = u[0] - uhat[0]
    isothermal[1] = -uhat[1]
    isothermal[2] = -uhat[2]
    isothermal[3] = -uhat[3] + uhat[0] * t_wall
    f = flux(uhat, q, w, v, x, t, mu, eta)
    adiabatic = 0 * u
    adiabatic[0:3] = isothermal[0:3]
    adiabatic[3] = f[3, 0] * n[0] + f[3, 1] * n[1] + tau[0] * (u[3] - uhat[3])
    temperature = 0 * u
    temperature[0:3] = isothermal[0:3]
    temperature[3] = uhat[3] / uhat[0]
    return np.column_stack((inflow, outflow, isothermal, adiabatic, temperature))


def fint(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    f = flux(uhat, q, w, v, x, t, mu, eta)
    return np.array([f[3, 0] * n[0] + f[3, 1] * n[1] + tau[0] * (u[3] - uhat[3])])


def initu(x, mu, eta):
    return np.array(mu[4:8])


def initv(x, mu, eta):
    return np.zeros(2)
