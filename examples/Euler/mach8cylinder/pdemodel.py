"""Python translation of the legacy Mach-8.98 cylinder model."""

import numpy as np
from sympy import atan, pi, sqrt, tanh


def _lmax(value, alpha):
    return value * (atan(alpha * value) / pi + 0.5) - atan(alpha) / pi + 0.5


def _lmin(value, alpha):
    return value - _lmax(value, alpha)


def _limiting(value, lower, upper, alpha, beta):
    value = lower + _lmax(value - beta, alpha)
    return _lmin(value - upper, alpha) + upper


def _viscosity(mu_ref, tref, temperature):
    sutherland = 110.4
    return mu_ref * (temperature / tref) ** 1.5 * (tref + sutherland) / (temperature + sutherland)


def mass(u, q, w, v, x, t, mu, eta):
    return np.ones(4)


def flux(u, q, w, v, x, t, mu, eta):
    gamma, reynolds, prandtl, mach = mu[0:4]
    gamma1 = gamma - 1.0
    alpha, rmin, pmin = 1.0e3, 1.0e-2, 1.0e-3
    density, momentum_x, momentum_y, total_energy = u
    rx, rux, rvx, rEx, ry, ruy, rvy, rEy = q
    density = rmin + _lmax(density - rmin, alpha)
    dr = atan(alpha * (density - rmin)) / pi + alpha * (density - rmin) / (
        pi * (alpha**2 * (density - rmin) ** 2 + 1.0)
    ) + 0.5
    rx, ry = rx * dr, ry * dr
    inverse_density = 1.0 / density
    velocity_x, velocity_y = momentum_x * inverse_density, momentum_y * inverse_density
    energy = total_energy * inverse_density
    kinetic = 0.5 * (velocity_x**2 + velocity_y**2)
    pressure = gamma1 * (total_energy - density * kinetic)
    pressure = pmin + _lmax(pressure - pmin, alpha)
    dp = atan(alpha * (pressure - pmin)) / pi + alpha * (pressure - pmin) / (
        pi * (alpha**2 * (pressure - pmin) ** 2 + 1.0)
    ) + 0.5
    enthalpy = energy + pressure * inverse_density
    inviscid = np.array([
        momentum_x, momentum_x * velocity_x + pressure, momentum_y * velocity_x,
        momentum_x * enthalpy, momentum_y, momentum_x * velocity_y,
        momentum_y * velocity_y + pressure, momentum_y * enthalpy,
    ])
    ux = (rux - rx * velocity_x) * inverse_density
    vx = (rvx - rx * velocity_y) * inverse_density
    uy = (ruy - ry * velocity_x) * inverse_density
    vy = (rvy - ry * velocity_y) * inverse_density
    px = gamma1 * (rEx - rx * kinetic - density * (velocity_x * ux + velocity_y * vx)) * dp
    py = gamma1 * (rEy - ry * kinetic - density * (velocity_x * uy + velocity_y * vy)) * dp
    tx = (px * density - pressure * rx) * inverse_density**2 / gamma1
    ty = (py * density - pressure * ry) * inverse_density**2 / gamma1
    tinf = 1.0 / (gamma * gamma1 * mach**2)
    temperature = pressure / (gamma1 * density)
    dynamic = _viscosity(1.0 / reynolds, mu[9], mu[9] * temperature / tinf) + mu[13] * v[0]
    conductivity = dynamic * gamma / prandtl
    bulk = v[0]
    tau_xx = dynamic * (2.0 / 3.0) * (2.0 * ux - vy) + bulk * (ux + vy)
    tau_xy = dynamic * (uy + vx)
    tau_yy = dynamic * (2.0 / 3.0) * (2.0 * vy - ux) + bulk * (ux + vy)
    viscous = np.array([
        0.0, tau_xx, tau_xy,
        velocity_x * tau_xx + velocity_y * tau_xy + (conductivity + mu[12] * v[0]) * tx,
        0.0, tau_xy, tau_yy,
        velocity_x * tau_xy + velocity_y * tau_yy + (conductivity + mu[12] * v[0]) * ty,
    ])
    return np.reshape(inviscid + viscous, (4, 2), order="F")


def avfield(u, q, w, v, x, t, mu, eta):
    gamma, porder = mu[0], mu[14]
    density, momentum_x, momentum_y, total_energy = u
    rx, rux, rvx, _, ry, ruy, rvy, _ = q
    alpha, rmin, hmin = 1.0e3, 1.0e-2, 1.0e-4
    density = rmin + _lmax(density - rmin, alpha)
    dr = atan(alpha * (density - rmin)) / pi + alpha * (density - rmin) / (
        pi * (alpha**2 * (density - rmin) ** 2 + 1.0)
    ) + 0.5
    rx, ry = rx * dr, ry * dr
    velocity_x, velocity_y = momentum_x / density, momentum_y / density
    ux, vx = (rux - rx * velocity_x) / density, (rvx - rx * velocity_y) / density
    uy, vy = (ruy - ry * velocity_x) / density, (rvy - ry * velocity_y) / density
    kinetic = 0.5 * (velocity_x**2 + velocity_y**2)
    enthalpy = hmin + _lmax(gamma * total_energy / density - (gamma - 1.0) * kinetic - hmin, alpha)
    sound = sqrt(2.0 * (gamma - 1.0) * enthalpy / (gamma + 1.0))
    divergence = _limiting(-(ux + vy), -1.0e4, 1.0e4, alpha, -1.0e4)
    vorticity = _limiting(sqrt((vx - uy) ** 2), 0.0, 1.0e4, alpha, 0.0)
    ratio = divergence**2 / (divergence**2 + vorticity**2 + 1.0e-16)
    sensor = -(v[1] / porder) * divergence / sound * ratio
    sensor = _limiting(sensor, 0.0, mu[16] / sqrt(gamma**2 - 1.0), alpha, mu[15])
    artificial = density * mu[11] * v[1] / porder
    artificial *= sqrt(velocity_x**2 + velocity_y**2 + sound**2) * sensor
    return np.array([artificial])


def source(u, q, w, v, x, t, mu, eta):
    return np.zeros(4)


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    value = flux(uhat, q, w, v, x, t, mu, eta) @ n + tau[0] * (u - uhat)
    adiabatic = np.array(value, copy=True); adiabatic[[0, 3]] = 0.0
    thermal = np.array(value, copy=True); thermal[0] = 0.0
    return np.column_stack((value, adiabatic, thermal, adiabatic, value, value))


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    gamma = mu[0]
    gamma1 = gamma - 1.0
    density, momentum_x, momentum_y, total_energy = u
    nx, ny = n
    velocity_x = momentum_x / density
    velocity_y = momentum_y / density
    energy = total_energy / density
    pressure = gamma1 * (
        total_energy - 0.5 * density * (velocity_x**2 + velocity_y**2)
    )
    enthalpy = energy + pressure / density
    sound = sqrt(gamma * pressure / density)
    normal_velocity = velocity_x * nx + velocity_y * ny
    tangent_velocity = -velocity_x * ny + velocity_y * nx

    right = np.array([
        [1.0, 1.0, 0.0, 1.0],
        [normal_velocity - sound, normal_velocity, 0.0, normal_velocity + sound],
        [tangent_velocity, tangent_velocity, 1.0, tangent_velocity],
        [enthalpy - normal_velocity * sound,
         0.5 * (normal_velocity**2 + tangent_velocity**2),
         tangent_velocity, enthalpy + normal_velocity * sound],
    ], dtype=object)
    right_inverse = gamma1 / (2.0 * sound**2) * np.array([
        [enthalpy + sound * (normal_velocity - sound) / gamma1,
         -(normal_velocity + sound / gamma1), -tangent_velocity, 1.0],
        [-2.0 * enthalpy + 4.0 * sound**2 / gamma1,
         2.0 * normal_velocity, 2.0 * tangent_velocity, -2.0],
        [-2.0 * tangent_velocity * sound**2 / gamma1,
         0.0, 2.0 * sound**2 / gamma1, 0.0],
        [enthalpy - sound * (normal_velocity + sound) / gamma1,
         -normal_velocity + sound / gamma1, -tangent_velocity, 1.0],
    ], dtype=object)
    rotate = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, nx, ny, 0.0],
        [0.0, -ny, nx, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=object)
    rotate_inverse = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, nx, -ny, 0.0],
        [0.0, ny, nx, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=object)
    characteristic_sign = np.diag([
        tanh(1.0e3 * (normal_velocity - sound)),
        tanh(1.0e3 * normal_velocity),
        tanh(1.0e3 * normal_velocity),
        tanh(1.0e3 * (normal_velocity + sound)),
    ])
    normal_sign = rotate_inverse @ right @ characteristic_sign @ right_inverse @ rotate
    uinf = np.array(mu[4:8]).reshape(4)
    farfield = 0.5 * ((u + uinf) + normal_sign @ (u - uinf))
    adiabatic = np.array(u, copy=True); adiabatic[1:3] = 0.0
    thermal = np.array(adiabatic, copy=True); thermal[3] = u[0] * mu[10] / mu[9] * mu[8]
    slip = np.array(u, copy=True)
    normal_momentum = u[1] * n[0] + u[2] * n[1]
    slip[1:3] -= n * normal_momentum
    return np.column_stack((farfield, adiabatic, thermal, slip, uinf, u))


def initu(x, mu, eta):
    return np.array(mu[4:8])
