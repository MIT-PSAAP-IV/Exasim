"""Equilibrium five-species-air model for the Python frontend.

This is the two-dimensional counterpart of ``pdemodel.m`` in this directory.
The material database supplies pressure, temperature, transport properties,
sound speed, and thermodynamic derivatives through ``w``.
"""

from __future__ import annotations

import numpy as np
import sympy as sp


def _zeros(rows, columns, prototype):
    return np.full((rows, columns), 0 * prototype, dtype=object)


def _qmat(q):
    return np.reshape(np.asarray(q, dtype=object), (4, 2), order="F")


def _tau_entry(tau, index):
    return tau[0] if len(tau) == 1 else tau[index]


def mass(u, q, w, v, x, t, mu, eta):
    return np.ones(4)


def materialstate(u, q, w, v, x, t, mu, eta):
    rho = u[0]
    velocity_x = u[1] / rho
    velocity_y = u[2] / rho
    internal_energy = u[3] / rho - 0.5 * (
        velocity_x * velocity_x + velocity_y * velocity_y
    )
    return np.array([sp.log(mu[0] * rho), mu[3] * internal_energy])


def _state(u, w, mu):
    rho = u[0]
    velocity = np.array([u[1] / rho, u[2] / rho], dtype=object)
    rho_energy = u[3]
    pressure = w[0] / mu[2]
    enthalpy = (rho_energy + pressure) / rho
    return rho, velocity, rho_energy, pressure, enthalpy


def _grad_primitives(u, gradient_u, w, mu):
    rho = u[0]
    velocity = np.array([u[1] / rho, u[2] / rho], dtype=object)
    total_energy = u[3] / rho
    gradient_velocity = _zeros(2, 2, u[0])
    gradient_temperature = np.full(2, 0 * u[0], dtype=object)
    for direction in range(2):
        gradient_rho = gradient_u[0, direction]
        for component in range(2):
            gradient_velocity[component, direction] = (
                gradient_u[1 + component, direction]
                - gradient_rho * velocity[component]
            ) / rho
        gradient_energy = (
            gradient_u[3, direction] - gradient_rho * total_energy
        ) / rho
        gradient_internal_energy = gradient_energy
        for component in range(2):
            gradient_internal_energy -= (
                velocity[component] * gradient_velocity[component, direction]
            )
        gradient_xi = gradient_rho / rho
        gradient_temperature[direction] = (
            w[8] * gradient_xi + w[9] * mu[3] * gradient_internal_energy
        )
    return gradient_velocity, gradient_temperature


def _stress_heat(velocity, gradient_velocity, w, mu):
    viscosity = mu[5] * w[2] / (mu[0] * mu[1] * mu[4])
    conductivity = (
        mu[5] * (w[3] + mu[6] * w[4]) / (mu[0] * mu[1] ** 3 * mu[4])
    )
    divergence = gradient_velocity[0, 0] + gradient_velocity[1, 1]
    stress = _zeros(2, 2, velocity[0])
    for i in range(2):
        for j in range(2):
            delta = 1.0 if i == j else 0.0
            stress[i, j] = viscosity * (
                gradient_velocity[i, j]
                + gradient_velocity[j, i]
                - (2.0 / 3.0) * divergence * delta
            )
    return stress, conductivity


def _flux_cartesian(u, q, w, v, x, mu):
    rho, velocity, _, pressure, enthalpy = _state(u, w, mu)
    gradient_u = -_qmat(q)
    gradient_velocity, gradient_temperature = _grad_primitives(
        u, gradient_u, w, mu
    )
    stress, conductivity = _stress_heat(velocity, gradient_velocity, w, mu)
    artificial_viscosity = (mu[-2] + mu[-1] * v[1]) * sp.tanh(mu[-3] * v[0])

    result = _zeros(4, 2, u[0])
    for direction in range(2):
        result[0, direction] = (
            rho * velocity[direction] - artificial_viscosity * gradient_u[0, direction]
        )
        for component in range(2):
            result[1 + component, direction] = (
                rho * velocity[component] * velocity[direction]
                - artificial_viscosity * gradient_u[1 + component, direction]
            )
        result[1 + direction, direction] += pressure
        result[3, direction] = (
            rho * velocity[direction] * enthalpy
            - artificial_viscosity * gradient_u[3, direction]
        )

    for direction in range(2):
        heat_work = 0 * u[0]
        for component in range(2):
            result[1 + component, direction] -= stress[component, direction]
            heat_work += velocity[component] * stress[component, direction]
        result[3, direction] -= heat_work + conductivity * gradient_temperature[direction]
    return result


def flux(u, q, w, v, x, t, mu, eta):
    return _flux_cartesian(u, q, w, v, x, mu)


def source(u, q, w, v, x, t, mu, eta):
    return np.full(4, 0 * u[0], dtype=object)


def _normal_basis(normal):
    norm = sp.sqrt(normal[0] * normal[0] + normal[1] * normal[1])
    nx = normal[0] / norm
    ny = normal[1] / norm
    return np.array([[nx, ny], [-ny, nx]], dtype=object)


def _remove_normal_momentum(u, normal):
    basis = _normal_basis(normal)
    nh = basis[0, :]
    momentum = np.asarray(u[1:3], dtype=object)
    normal_momentum = momentum[0] * nh[0] + momentum[1] * nh[1]
    state = np.asarray(u, dtype=object).copy()
    state[1] = momentum[0] - normal_momentum * nh[0]
    state[2] = momentum[1] - normal_momentum * nh[1]
    return state


def _wall_state_temperature(u, w, mu):
    rho = u[0]
    velocity_x = u[1] / rho
    velocity_y = u[2] / rho
    internal_energy = u[3] / rho - 0.5 * (
        velocity_x * velocity_x + velocity_y * velocity_y
    )
    wall_energy = internal_energy + (mu[7] - w[1]) / (w[9] * mu[3])
    state = np.asarray(u, dtype=object).copy()
    state[1] = 0 * u[0]
    state[2] = 0 * u[0]
    state[3] = rho * wall_energy
    return state


def _outlet_pressure(interior_pressure, mu):
    return mu[8] if len(mu) >= 9 else interior_pressure


def _outlet_relaxation(mu):
    return mu[9] if len(mu) >= 10 else 1.0


def _outlet_state(u, w, normal, mu):
    rho = u[0]
    rho_energy = u[3]
    basis = _normal_basis(normal)
    local_momentum = basis @ np.asarray(u[1:3], dtype=object)
    local_velocity = local_momentum / rho
    normal_velocity = local_velocity[0]
    total_energy = rho_energy / rho
    sound_speed = w[5] / mu[1]
    pressure_rho = w[6] / (rho * mu[2])
    pressure_energy = mu[3] * w[7] / mu[2]
    acoustic_energy = total_energy + rho * (
        sound_speed * sound_speed - pressure_rho
    ) / pressure_energy
    pressure_delta = (
        _outlet_relaxation(mu) * (_outlet_pressure(w[0], mu) - w[0]) / mu[2]
    )
    density_delta = pressure_delta / (sound_speed * sound_speed)
    right_eigenvector = np.array(
        [
            1.0,
            normal_velocity - sound_speed,
            local_velocity[1],
            acoustic_energy - normal_velocity * sound_speed,
        ],
        dtype=object,
    )
    transform_inverse = _zeros(4, 4, u[0])
    transform_inverse[0, 0] = 1.0
    transform_inverse[1:3, 1:3] = basis.T
    transform_inverse[3, 3] = 1.0
    return np.asarray(u, dtype=object) + density_delta * (
        transform_inverse @ right_eigenvector
    )


def _boundary_states(u, q, w, x, mu, eta, normal):
    qn = _qmat(q) @ np.asarray(normal, dtype=object)
    inflow = np.asarray(eta[0:4], dtype=object)
    outflow = _outlet_state(u, w, normal, mu)
    isothermal = _wall_state_temperature(u, w, mu)
    adiabatic = np.asarray(u, dtype=object).copy()
    adiabatic[1:3] = 0 * u[0]
    symmetry = _remove_normal_momentum(u, normal)
    gradient = np.asarray(u, dtype=object).copy()
    states = np.column_stack(
        (inflow, outflow, isothermal, adiabatic, symmetry, gradient, inflow, outflow)
    )
    return states, qn


def _normal_flux(u, q, w, v, x, mu, normal):
    return _flux_cartesian(u, q, w, v, x, mu) @ np.asarray(normal, dtype=object)


def _sign_matrix(state, normal, w, mu):
    rho = state[0]
    rho_energy = state[3]
    basis = _normal_basis(normal)
    local_momentum = basis @ np.asarray(state[1:3], dtype=object)
    local_velocity = local_momentum / rho
    normal_velocity = local_velocity[0]
    total_energy = rho_energy / rho
    sound_speed = w[5] / mu[1]
    pressure_rho = w[6] / (rho * mu[2])
    pressure_energy = mu[3] * w[7] / mu[2]
    bvalue = total_energy + rho * (
        sound_speed * sound_speed - pressure_rho
    ) / pressure_energy
    cvalue = total_energy - rho * pressure_rho / pressure_energy

    eigenvectors = _zeros(4, 4, state[0])
    eigenvectors[:, 0] = [
        1.0,
        normal_velocity - sound_speed,
        local_velocity[1],
        bvalue - normal_velocity * sound_speed,
    ]
    eigenvectors[:, 1] = [1.0, normal_velocity, local_velocity[1], cvalue]
    eigenvectors[:, 2] = [0.0, 0.0, 1.0, local_velocity[1]]
    eigenvectors[:, 3] = [
        1.0,
        normal_velocity + sound_speed,
        local_velocity[1],
        bvalue + normal_velocity * sound_speed,
    ]

    transform = _zeros(4, 4, state[0])
    transform[0, 0] = 1.0
    transform[1:3, 1:3] = basis
    transform[3, 3] = 1.0
    transform_inverse = _zeros(4, 4, state[0])
    transform_inverse[0, 0] = 1.0
    transform_inverse[1:3, 1:3] = basis.T
    transform_inverse[3, 3] = 1.0
    eigenvalue_sign = np.diag(
        [
            sp.tanh(100 * (normal_velocity - sound_speed)),
            sp.tanh(100 * normal_velocity),
            sp.tanh(100 * normal_velocity),
            sp.tanh(100 * (normal_velocity + sound_speed)),
        ]
    )
    inverse_eigenvectors = np.asarray(
        sp.Matrix(eigenvectors.tolist()).inv().tolist(), dtype=object
    )
    return transform_inverse @ eigenvectors @ eigenvalue_sign @ inverse_eigenvectors @ transform


def _characteristic_state(u, exterior, linearization, normal, w, mu):
    sign_matrix = _sign_matrix(linearization, normal, w, mu)
    return 0.5 * (
        np.asarray(u, dtype=object)
        + np.asarray(exterior, dtype=object)
        + sign_matrix
        @ (np.asarray(u, dtype=object) - np.asarray(exterior, dtype=object))
    )


def _symmetry_residual(u, q, x, normal, tau, uhat):
    basis = _normal_basis(normal)
    nh = basis[0, :]
    qmatrix = _qmat(q)
    momentum_hat = np.asarray(uhat[1:3], dtype=object)
    normal_momentum = momentum_hat[0] * nh[0] + momentum_hat[1] * nh[1]
    normal_momentum_gradient = qmatrix[1:3, :] @ nh
    tangent_projection = np.eye(2, dtype=object) - np.outer(nh, nh)
    tangent_gradient = tangent_projection @ normal_momentum_gradient
    residual = np.full(4, 0 * u[0], dtype=object)
    residual[0] = qmatrix[0, :] @ nh + _tau_entry(tau, 0) * (u[0] - uhat[0])
    residual[1:3] = normal_momentum * nh + tangent_gradient
    residual[3] = qmatrix[3, :] @ nh + _tau_entry(tau, 3) * (u[3] - uhat[3])
    return residual


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    states, qn = _boundary_states(u, q, w, x, mu, eta, n)
    result = _zeros(4, 8, u[0])
    interior = states[:, 5]
    for column, state_column in ((0, 0), (1, 5), (2, 2)):
        state = states[:, state_column]
        result[:, column] = _normal_flux(state, q, w, v, x, mu, n) + tau * (
            state - uhat
        )
    result[:, 3] = _normal_flux(states[:, 3], q, w, v, x, mu, n) + tau * (
        states[:, 3] - uhat
    )
    result[0, 3] = 0 * u[0]
    result[3, 3] = 0 * u[0]
    result[:, 4] = _symmetry_residual(interior, q, x, n, tau, uhat)
    result[:, 5] = qn + tau * (interior - uhat)
    inflow = _characteristic_state(interior, states[:, 6], interior, n, w, mu)
    outflow = _characteristic_state(interior, states[:, 7], interior, n, w, mu)
    result[:, 6] = _normal_flux(interior, q, w, v, x, mu, n) + tau * (
        interior - inflow
    )
    result[:, 7] = _normal_flux(interior, q, w, v, x, mu, n) + tau * (
        interior - outflow
    )
    return result


def fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    states, qn = _boundary_states(u, q, w, x, mu, eta, n)
    result = tau * (states - np.reshape(uhat, (4, 1)))
    result[:, 0] = states[:, 0] - uhat
    result[:, 1] = states[:, 5] - uhat
    result[:, 6] = _characteristic_state(
        states[:, 5], states[:, 6], uhat, n, w, mu
    ) - uhat
    result[:, 7] = _characteristic_state(
        states[:, 5], states[:, 7], uhat, n, w, mu
    ) - uhat
    adiabatic = np.full(4, 0 * u[0], dtype=object)
    adiabatic[0] = states[0, 3] - uhat[0]
    adiabatic[1:3] = -np.asarray(uhat[1:3], dtype=object)
    normal_flux = _normal_flux(uhat, q, w, v, x, mu, n)
    adiabatic[3] = normal_flux[3] + _tau_entry(tau, 3) * (
        states[3, 3] - uhat[3]
    )
    result[:, 3] = adiabatic
    result[:, 4] = _symmetry_residual(states[:, 5], q, x, n, tau, uhat)
    result[:, 5] = qn + tau * (states[:, 5] - uhat)
    return result


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return _boundary_states(u, q, w, x, mu, eta, n)[0]


def initu(x, mu, eta):
    return np.asarray(eta[0:4], dtype=object)


def initv(x, mu, eta):
    return np.array([0.0, 0.0])


def initw(x, mu, eta):
    return np.array(
        [
            1.0,
            300.0,
            1.0e-5,
            1.0e-2,
            0.0,
            300.0,
            1.0,
            1.0e-3,
            0.0,
            1.0e-3,
            0.767,
            0.233,
            0.0,
            0.0,
            0.0,
        ]
    )


def visscalars(u, q, w, v, x, t, mu, eta):
    return np.array([w[0] / mu[2]])


def _limited_maximum(value, alpha):
    return value * (sp.atan(alpha * value) / sp.pi + 0.5) - sp.atan(alpha) / sp.pi + 0.5


def _limiting(value, lower, upper, alpha, beta):
    lower_limited = lower + _limited_maximum(value - beta, alpha)
    return lower_limited - _limited_maximum(lower_limited - upper, alpha)


def avfield(u, q, w, v, x, t, mu, eta):
    rho = u[0]
    velocity_x = u[1] / rho
    velocity_y = u[2] / rho
    gradient = _qmat(q)
    velocity_x_gradient = (gradient[1, 0] - gradient[0, 0] * velocity_x) / rho
    velocity_y_gradient = (gradient[2, 1] - gradient[0, 1] * velocity_y) / rho
    divergence = velocity_x_gradient + velocity_y_gradient
    return np.array(
        [_limiting(divergence * sp.tanh(mu[-3] * v[0]), 0.0, mu[-4], 1.0e3, 0.0)]
    )
