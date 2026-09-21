"""Python translation of the MATLAB Mach-8 cylinder PDE model."""

from __future__ import annotations

import numpy as np
from sympy import atan, pi, sqrt, tanh


def _lmax(value, alpha):
    return value * (atan(alpha * value) / pi + 0.5) - atan(alpha) / pi + 0.5


def _lmin(value, alpha):
    return value - _lmax(value, alpha)


def _limiting(value, lower, upper, alpha, beta):
    limited = lower + _lmax(value - beta, alpha)
    return _lmin(limited - upper, alpha) + upper


def _viscosity(mu_ref, tref, temperature):
    sutherland_temperature = 110.4
    return (
        mu_ref
        * (temperature / tref) ** 1.5
        * (tref + sutherland_temperature)
        / (temperature + sutherland_temperature)
    )


def mass(u, q, w, v, x, t, mu, eta):
    return np.array([1.0, 1.0, 1.0, 1.0])


def flux(u, q, w, v, x, t, mu, eta):
    gam = mu[0]
    gam1 = gam - 1.0
    reynolds = mu[1]
    prandtl = mu[2]
    mach_inf = mu[3]
    tref = mu[9]
    mu_ref = 1.0 / reynolds
    tinf = 1.0 / (gam * gam1 * mach_inf**2)
    c23 = 2.0 / 3.0

    alpha = 1.0e3
    rmin = 1.0e-2
    pmin = 1.0e-3
    av = (mu[-2] + mu[-1] * v[1]) * tanh(mu[-3] * v[0])

    r, ru, rv, rE = u
    rx, rux, rvx, rEx, ry, ruy, rvy, rEy = q

    r = rmin + _lmax(r - rmin, alpha)
    dr = (
        atan(alpha * (r - rmin)) / pi
        + alpha * (r - rmin) / (pi * (alpha**2 * (r - rmin) ** 2 + 1.0))
        + 0.5
    )
    rx = rx * dr
    ry = ry * dr
    r1 = 1.0 / r
    ux_velocity = ru * r1
    uy_velocity = rv * r1
    total_energy = rE * r1
    kinetic_energy = 0.5 * (ux_velocity**2 + uy_velocity**2)
    pressure = gam1 * (rE - r * kinetic_energy)
    pressure = pmin + _lmax(pressure - pmin, alpha)
    dp = (
        atan(alpha * (pressure - pmin)) / pi
        + alpha * (pressure - pmin)
        / (pi * (alpha**2 * (pressure - pmin) ** 2 + 1.0))
        + 0.5
    )
    enthalpy = total_energy + pressure * r1
    inviscid = np.array(
        [
            ru,
            ru * ux_velocity + pressure,
            rv * ux_velocity,
            ru * enthalpy,
            rv,
            ru * uy_velocity,
            rv * uy_velocity + pressure,
            rv * enthalpy,
        ]
    )

    du_dx = (rux - rx * ux_velocity) * r1
    dv_dx = (rvx - rx * uy_velocity) * r1
    dke_dx = ux_velocity * du_dx + uy_velocity * dv_dx
    dp_dx = gam1 * (rEx - rx * kinetic_energy - r * dke_dx) * dp
    dt_dx = (dp_dx * r - pressure * rx) * r1**2 / gam1

    du_dy = (ruy - ry * ux_velocity) * r1
    dv_dy = (rvy - ry * uy_velocity) * r1
    dke_dy = ux_velocity * du_dy + uy_velocity * dv_dy
    dp_dy = gam1 * (rEy - ry * kinetic_energy - r * dke_dy) * dp
    dt_dy = (dp_dy * r - pressure * ry) * r1**2 / gam1

    temperature = pressure / (gam1 * r)
    physical_temperature = tref * temperature / tinf
    dynamic_viscosity = _viscosity(mu_ref, tref, physical_temperature)
    conductivity = dynamic_viscosity * gam / prandtl
    tau_xx = dynamic_viscosity * c23 * (2.0 * du_dx - dv_dy)
    tau_xy = dynamic_viscosity * (du_dy + dv_dx)
    tau_yy = dynamic_viscosity * c23 * (2.0 * dv_dy - du_dx)
    viscous = np.array(
        [
            0.0,
            tau_xx,
            tau_xy,
            ux_velocity * tau_xx + uy_velocity * tau_xy + conductivity * dt_dx,
            0.0,
            tau_xy,
            tau_yy,
            ux_velocity * tau_xy + uy_velocity * tau_yy + conductivity * dt_dy,
        ]
    )
    artificial = np.array(
        [
            av * rx,
            av * rux,
            av * rvx,
            av * rEx,
            av * ry,
            av * ruy,
            av * rvy,
            av * rEy,
        ]
    )
    return np.reshape(inviscid + viscous + artificial, (4, 2), order="F")


def avfield(u, q, w, v, x, t, mu, eta):
    r, ru, rv = u[0:3]
    rx, rux, ry, rvy = q[0], q[1], q[4], q[6]
    velocity_x = ru / r
    velocity_y = rv / r
    du_dx = (rux - rx * velocity_x) / r
    dv_dy = (rvy - ry * velocity_y) / r
    divergence = du_dx + dv_dy
    return np.array(
        [_limiting(divergence * tanh(mu[-3] * v[0]), 0.0, mu[-4], 1.0e3, 0.0)]
    )


def visscalars(u, q, w, v, x, t, mu, eta):
    gam = mu[0]
    density = u[0]
    velocity_x = u[1] / density
    velocity_y = u[2] / density
    signed_pressure = (gam - 1.0) * (
        u[3] - 0.5 * (u[1] * velocity_x + u[2] * velocity_y)
    )
    pressure = sqrt(signed_pressure * signed_pressure)
    mach = sqrt(velocity_x**2 + velocity_y**2) / sqrt(gam * pressure / density)
    artificial_viscosity = mu[-1] * v[1] * tanh(mu[-3] * v[0])
    return np.array([mach, artificial_viscosity])


def source(u, q, w, v, x, t, mu, eta):
    return np.array([0.0 * u[0], 0.0 * u[0], 0.0 * u[0], 0.0 * u[0]])


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    f = flux(uhat, q, w, v, x, t, mu, eta)
    freestream = f[:, 0] * n[0] + f[:, 1] * n[1] + tau[0] * (u - uhat)
    adiabatic_wall = np.array(freestream, copy=True)
    adiabatic_wall[0] = 0.0
    adiabatic_wall[-1] = 0.0
    thermal_wall = np.array(freestream, copy=True)
    thermal_wall[0] = 0.0
    return np.column_stack(
        (freestream, adiabatic_wall, thermal_wall, adiabatic_wall, freestream, freestream)
    )


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    uinf = np.array(mu[4:8]).reshape(4)
    t_iso_wall = mu[10] / mu[9] * mu[8]
    isothermal = np.array(u, copy=True)
    isothermal[1:3] = 0.0
    isothermal[3] = u[0] * t_iso_wall
    slip = np.array(u, copy=True)
    normal_momentum = u[1] * n[0] + u[2] * n[1]
    slip[1] = u[1] - n[0] * normal_momentum
    slip[2] = u[2] - n[1] * normal_momentum
    return np.column_stack((uinf, uinf, isothermal, slip, uinf, u))


def fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    t_iso_wall = mu[10] / mu[9] * mu[8]
    uinf = np.array(mu[4:8]).reshape(4)
    outflow = u - uhat
    inflow = uinf - uhat
    wall = 0 * u
    wall[0] = u[0] - uhat[0]
    wall[1] = -uhat[1]
    wall[2] = -uhat[2]
    wall[3] = -uhat[3] + uhat[0] * t_iso_wall
    return np.column_stack((inflow, inflow, wall, wall, inflow, outflow))


def initu(x, mu, eta):
    return np.array(mu[4:8])


def initv(x, mu, eta):
    return np.array([0.0, 0.0])
