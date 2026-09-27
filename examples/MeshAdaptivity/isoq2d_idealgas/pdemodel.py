"""Python translation of the axisymmetric ideal-gas ISOQ model."""

from __future__ import annotations

import numpy as np
from sympy import atan, pi, sqrt, tanh


def _lmax(value, alpha):
    return value * (atan(alpha * value) / pi + 0.5) - atan(alpha) / pi + 0.5


def _lmin(value, alpha):
    return value - _lmax(value, alpha)


def _limiting(value, lower, upper, alpha, beta):
    value = lower + _lmax(value - beta, alpha)
    return _lmin(value - upper, alpha) + upper


def _viscosity(reference_viscosity, reference_temperature, temperature):
    sutherland_temperature = 110.4
    return reference_viscosity * (temperature / reference_temperature) ** 1.5 * (
        reference_temperature + sutherland_temperature
    ) / (temperature + sutherland_temperature)


def mass(u, q, w, v, x, t, mu, eta):
    return np.ones(4)


def flux(u, q, w, v, x, t, mu, eta):
    gam, reynolds, prandtl, mach_inf = mu[0:4]
    gam1 = gam - 1.0
    reference_temperature = mu[9]
    reference_viscosity = 1.0 / reynolds
    tinf = 1.0 / (gam * gam1 * mach_inf**2)
    two_thirds = 2.0 / 3.0
    alpha, density_minimum, pressure_minimum = 4.0e3, 5.0e-2, 2.0e-3
    artificial_viscosity = (mu[-2] + mu[-1] * v[1]) * tanh(mu[-3] * v[0])

    density, momentum_x, momentum_y, total_energy_density = u
    density_x, momentum_x_x, momentum_y_x, energy_x = q[0:4]
    density_y, momentum_x_y, momentum_y_y, energy_y = q[4:8]
    density = density_minimum + _lmax(density - density_minimum, alpha)
    density_derivative = (
        atan(alpha * (density - density_minimum)) / pi
        + alpha * (density - density_minimum)
        / (pi * (alpha**2 * (density - density_minimum) ** 2 + 1.0))
        + 0.5
    )
    density_x *= density_derivative
    density_y *= density_derivative
    inverse_density = 1.0 / density
    velocity_x = momentum_x * inverse_density
    velocity_y = momentum_y * inverse_density
    total_energy = total_energy_density * inverse_density
    kinetic_energy = 0.5 * (velocity_x**2 + velocity_y**2)
    pressure = gam1 * (total_energy_density - density * kinetic_energy)
    pressure = pressure_minimum + _lmax(pressure - pressure_minimum, alpha)
    pressure_derivative = (
        atan(alpha * (pressure - pressure_minimum)) / pi
        + alpha * (pressure - pressure_minimum)
        / (pi * (alpha**2 * (pressure - pressure_minimum) ** 2 + 1.0))
        + 0.5
    )
    enthalpy = total_energy + pressure * inverse_density
    inviscid = np.array(
        [
            momentum_x,
            momentum_x * velocity_x + pressure,
            momentum_y * velocity_x,
            momentum_x * enthalpy,
            momentum_y,
            momentum_x * velocity_y,
            momentum_y * velocity_y + pressure,
            momentum_y * enthalpy,
        ]
    )

    velocity_x_x = (momentum_x_x - density_x * velocity_x) * inverse_density
    velocity_y_x = (momentum_y_x - density_x * velocity_y) * inverse_density
    kinetic_x = velocity_x * velocity_x_x + velocity_y * velocity_y_x
    pressure_x = gam1 * (
        energy_x - density_x * kinetic_energy - density * kinetic_x
    ) * pressure_derivative
    temperature_x = (
        pressure_x * density - pressure * density_x
    ) * inverse_density**2 / gam1
    velocity_x_y = (momentum_x_y - density_y * velocity_x) * inverse_density
    velocity_y_y = (momentum_y_y - density_y * velocity_y) * inverse_density
    kinetic_y = velocity_x * velocity_x_y + velocity_y * velocity_y_y
    pressure_y = gam1 * (
        energy_y - density_y * kinetic_energy - density * kinetic_y
    ) * pressure_derivative
    temperature_y = (
        pressure_y * density - pressure * density_y
    ) * inverse_density**2 / gam1

    temperature = pressure / (gam1 * density)
    physical_temperature = reference_temperature * temperature / tinf
    dynamic_viscosity = _viscosity(
        reference_viscosity, reference_temperature, physical_temperature
    )
    conductivity = dynamic_viscosity * gam / prandtl
    radial_coordinate = x[1]
    stress_xx = dynamic_viscosity * two_thirds * (
        2.0 * velocity_x_x - velocity_y_y + velocity_y / radial_coordinate
    )
    stress_xy = dynamic_viscosity * (velocity_x_y + velocity_y_x)
    stress_yy = dynamic_viscosity * two_thirds * (
        2.0 * velocity_y_y - velocity_x_x + velocity_y / radial_coordinate
    )
    viscous = np.array(
        [
            0.0,
            stress_xx,
            stress_xy,
            velocity_x * stress_xx + velocity_y * stress_xy
            + conductivity * temperature_x,
            0.0,
            stress_xy,
            stress_yy,
            velocity_x * stress_xy + velocity_y * stress_yy
            + conductivity * temperature_y,
        ]
    )
    artificial = artificial_viscosity * np.array(q)
    return np.reshape(inviscid + viscous + artificial, (4, 2), order="F")


def avfield(u, q, w, v, x, t, mu, eta):
    density = u[0]
    velocity_x = u[1] / density
    velocity_y = u[2] / density
    velocity_x_x = (q[1] - q[0] * velocity_x) / density
    velocity_y_y = (q[6] - q[4] * velocity_y) / density
    compression = velocity_x_x + velocity_y_y + velocity_y / x[1]
    return np.array(
        [_limiting(compression * tanh(mu[-3] * v[0]), 0.0, mu[-4], 1.0e3, 0.0)]
    )


def source(u, q, w, v, x, t, mu, eta):
    gam, reynolds, prandtl, mach_inf = mu[0:4]
    gam1 = gam - 1.0
    reference_temperature = mu[9]
    reference_viscosity = 1.0 / reynolds
    tinf = 1.0 / (gam * gam1 * mach_inf**2)
    two_thirds = 2.0 / 3.0
    alpha, density_minimum, pressure_minimum = 4.0e3, 5.0e-2, 2.0e-3
    artificial_viscosity = (mu[-2] + mu[-1] * v[1]) * tanh(mu[-3] * v[0])

    density, momentum_x, momentum_y, total_energy_density = u
    density_x, momentum_x_x, momentum_y_x = q[0:3]
    density_y, momentum_x_y, momentum_y_y, energy_y = q[4:8]
    density = density_minimum + _lmax(density - density_minimum, alpha)
    density_derivative = (
        atan(alpha * (density - density_minimum)) / pi
        + alpha * (density - density_minimum)
        / (pi * (alpha**2 * (density - density_minimum) ** 2 + 1.0))
        + 0.5
    )
    density_x *= density_derivative
    density_y *= density_derivative
    inverse_density = 1.0 / density
    velocity_x = momentum_x * inverse_density
    velocity_y = momentum_y * inverse_density
    total_energy = total_energy_density * inverse_density
    kinetic_energy = 0.5 * (velocity_x**2 + velocity_y**2)
    pressure = gam1 * (total_energy_density - density * kinetic_energy)
    pressure = pressure_minimum + _lmax(pressure - pressure_minimum, alpha)
    pressure_derivative = (
        atan(alpha * (pressure - pressure_minimum)) / pi
        + alpha * (pressure - pressure_minimum)
        / (pi * (alpha**2 * (pressure - pressure_minimum) ** 2 + 1.0))
        + 0.5
    )
    enthalpy = total_energy + pressure * inverse_density
    inviscid_y = np.array(
        [
            momentum_y,
            momentum_x * velocity_y,
            momentum_y * velocity_y,
            momentum_y * enthalpy,
        ]
    )
    velocity_x_x = (momentum_x_x - density_x * velocity_x) * inverse_density
    velocity_y_x = (momentum_y_x - density_x * velocity_y) * inverse_density
    velocity_x_y = (momentum_x_y - density_y * velocity_x) * inverse_density
    velocity_y_y = (momentum_y_y - density_y * velocity_y) * inverse_density
    kinetic_y = velocity_x * velocity_x_y + velocity_y * velocity_y_y
    pressure_y = gam1 * (
        energy_y - density_y * kinetic_energy - density * kinetic_y
    ) * pressure_derivative
    temperature_y = (
        pressure_y * density - pressure * density_y
    ) * inverse_density**2 / gam1
    temperature = pressure / (gam1 * density)
    physical_temperature = reference_temperature * temperature / tinf
    dynamic_viscosity = _viscosity(
        reference_viscosity, reference_temperature, physical_temperature
    )
    conductivity = dynamic_viscosity * gam / prandtl
    radial_coordinate = x[1]
    stress_xy = dynamic_viscosity * (velocity_x_y + velocity_y_x)
    stress_yy = dynamic_viscosity * two_thirds * (
        2.0 * velocity_y_y - velocity_x_x + velocity_y / radial_coordinate
    )
    stress_theta = dynamic_viscosity * two_thirds * (
        -2.0 * velocity_y / radial_coordinate - velocity_x_x - velocity_y_y
    )
    viscous_y = np.array(
        [
            0.0,
            stress_xy,
            stress_yy - stress_theta,
            velocity_x * stress_xy + velocity_y * stress_yy
            + conductivity * temperature_y,
        ]
    )
    artificial_y = artificial_viscosity * np.array(q[4:8])
    return -(inviscid_y + viscous_y + artificial_y) / radial_coordinate


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return np.zeros(4)


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return np.zeros(4)


def fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    gam1 = mu[0] - 1.0
    isothermal_temperature = mu[10] / mu[9] * mu[8]
    freestream = np.array(mu[4:8]).reshape(4)
    inflow = freestream - uhat
    outflow = u - uhat
    isothermal = 0 * u
    isothermal[0] = u[0] - uhat[0]
    isothermal[1] = -uhat[1]
    isothermal[2] = -uhat[2]
    isothermal[3] = -uhat[3] + uhat[0] * isothermal_temperature

    density, momentum_x, momentum_y, total_energy_density = uhat
    inverse_density = 1.0 / density
    velocity_x = momentum_x * inverse_density
    velocity_y = momentum_y * inverse_density
    kinetic_energy = 0.5 * (velocity_x**2 + velocity_y**2)
    pressure = gam1 * (total_energy_density - density * kinetic_energy)
    density_y, momentum_x_y, momentum_y_y, energy_y = q[4:8]
    velocity_x_y = (momentum_x_y - density_y * velocity_x) * inverse_density
    velocity_y_y = (momentum_y_y - density_y * velocity_y) * inverse_density
    kinetic_y = velocity_x * velocity_x_y + velocity_y * velocity_y_y
    pressure_y = gam1 * (energy_y - density_y * kinetic_energy - density * kinetic_y)
    temperature_y = (
        pressure_y * density - pressure * density_y
    ) * inverse_density**2 / gam1
    symmetry = 0 * u
    symmetry[0] = density_y + tau[0] * (u[0] - uhat[0])
    symmetry[1] = velocity_x_y + tau[0] * (u[1] - uhat[1])
    symmetry[2] = velocity_y_y - tau[0] * uhat[2]
    symmetry[3] = temperature_y + tau[0] * (u[3] - uhat[3])

    normal_momentum = u[1] * n[0] + u[2] * n[1]
    slip_state = np.array(u, copy=True)
    slip_state[1] -= n[0] * normal_momentum
    slip_state[2] -= n[1] * normal_momentum
    slip = tau[0] * (slip_state - uhat)
    gradient = np.array(q[0:4]) * n[0] + np.array(q[4:8]) * n[1] + tau[0] * (u - uhat)
    auxiliary_wall = 0 * u
    auxiliary_wall[0] = u[0] - uhat[0]
    auxiliary_wall[1] = -uhat[1]
    auxiliary_wall[2] = -uhat[2]
    return np.column_stack(
        (inflow, outflow, isothermal, symmetry, slip, gradient, auxiliary_wall)
    )


def initu(x, mu, eta):
    return np.array(mu[4:8])


def initv(x, mu, eta):
    return np.array([0.0, 0.0])


def visscalars(u, q, w, v, x, t, mu, eta):
    gam, reynolds, _, mach_inf = mu[0:4]
    reference_temperature = mu[9]
    gas_constant, standard_temperature, sutherland_temperature = 287.0, 273.15, 110.4
    standard_viscosity = 1.716e-5
    density = u[0]
    velocity_x, velocity_y = u[1] / density, u[2] / density
    pressure = (gam - 1.0) * (
        u[3] - 0.5 * (u[1] * velocity_x + u[2] * velocity_y)
    )
    temperature = pressure / ((gam - 1.0) * density)
    tinf = 1.0 / (gam * (gam - 1.0) * mach_inf**2)
    physical_temperature = reference_temperature * temperature / tinf
    velocity_reference = mach_inf * sqrt(gam * gas_constant * reference_temperature)
    ratio = reference_temperature / standard_temperature
    physical_viscosity = standard_viscosity * sqrt(ratio**3) * (
        standard_temperature + sutherland_temperature
    ) / (reference_temperature + sutherland_temperature)
    density_reference = reynolds * physical_viscosity / velocity_reference
    physical_density = density_reference * density
    physical_pressure = physical_density * gas_constant * physical_temperature
    mach = velocity_reference * sqrt(velocity_x**2 + velocity_y**2) / sqrt(
        gam * gas_constant * physical_temperature
    )
    artificial_viscosity = (mu[-2] + mu[-1] * v[1]) * tanh(mu[-3] * v[0])
    return np.array(
        [physical_density, physical_pressure, physical_temperature, mach, artificial_viscosity]
    )


def visvectors(u, q, w, v, x, t, mu, eta):
    velocity_reference = mu[3] * sqrt(mu[0] * 287.0 * mu[9])
    return velocity_reference * np.array(u[1:3]) / u[0]
