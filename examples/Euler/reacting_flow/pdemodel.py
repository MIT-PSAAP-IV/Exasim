"""Python five-species reacting Euler model."""

import numpy as np

from reacting_kernels import MW, pressure, production_rates, temperature_residual

NS = 5
NCH = 8


def mass(u, q, w, v, x, t, mu, eta):
    return np.ones(NCH)


def flux(u, q, w, v, x, t, mu, eta):
    density_species = u[:NS]
    density = sum(density_species)
    momentum_x, momentum_y, total_energy = u[NS:NS + 3]
    velocity_x, velocity_y = momentum_x / density, momentum_y / density
    pressure_value = pressure(w[0] * eta[3], density_species * eta[0]) / eta[2]
    enthalpy = total_energy / density + pressure_value / density
    qx, qy = -q[:NCH], -q[NCH:2 * NCH]
    artificial = v[0]
    result = np.empty((NCH, 2), dtype=object)
    for species in range(NS):
        result[species, 0] = density_species[species] * velocity_x - artificial * qx[species]
        result[species, 1] = density_species[species] * velocity_y - artificial * qy[species]
    result[NS, 0] = momentum_x * velocity_x + pressure_value - artificial * qx[NS]
    result[NS + 1, 0] = momentum_y * velocity_x - artificial * qx[NS + 1]
    result[NS + 2, 0] = momentum_x * enthalpy - artificial * qx[NS + 2]
    result[NS, 1] = momentum_x * velocity_y - artificial * qy[NS]
    result[NS + 1, 1] = momentum_y * velocity_y + pressure_value - artificial * qy[NS + 1]
    result[NS + 2, 1] = momentum_y * enthalpy - artificial * qy[NS + 2]
    return result


def source(u, q, w, v, x, t, mu, eta):
    dimensional_density = u[:NS] * eta[0]
    dimensional_temperature = w[0] * eta[3]
    omega_scale = eta[0] * eta[1] / eta[7]
    return np.concatenate((production_rates(dimensional_density, dimensional_temperature) / omega_scale, np.zeros(3)))


def _boundary_states(u, uhat, n):
    inflow_state = np.array([0.00000065,0.22830098,0.01026010,0.75430704,0.00713123,0.996653279785810,0.0,0.607507866832124])
    inflow, outflow = inflow_state - uhat, u - uhat
    wall_state = np.array(u, copy=True)
    normal_momentum = u[NS] * n[0] + u[NS + 1] * n[1]
    wall_state[NS] -= n[0] * normal_momentum
    wall_state[NS + 1] -= n[1] * normal_momentum
    wall = wall_state - uhat
    return np.column_stack((inflow, inflow, wall, wall, inflow, outflow))


def fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return _boundary_states(u, uhat, n)


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return 0 * _boundary_states(u, uhat, n)


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return 0 * _boundary_states(u, uhat, n)


def initu(x, mu, eta):
    return np.concatenate((mu[:NS] / eta[0], mu[NS:NS + 2] / (eta[0] * eta[1]), [mu[NS + 2] / eta[2]]))


def initw(x, mu, eta):
    return np.array([1.0])


def eos(u, q, w, v, x, t, mu, eta):
    dimensional_density = u[:NS] * eta[0]
    momentum = u[NS:NS + 2] * (eta[0] * eta[1])
    energy = u[NS + 2] * eta[2]
    internal = energy - 0.5 * np.dot(momentum, momentum) / sum(dimensional_density)
    return temperature_residual(w[0] * eta[3], dimensional_density / MW, internal)


def sourcew(u, q, w, v, x, t, mu, eta):
    dimensional_density = u[:NS] * eta[0]
    momentum = u[NS:NS + 2] * (eta[0] * eta[1])
    energy = u[NS + 2] * eta[2]
    internal = mu[22] * (energy - 0.5 * np.dot(momentum, momentum) / sum(dimensional_density))
    return temperature_residual(w[0] * eta[3], dimensional_density / MW, internal)
