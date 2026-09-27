"""Python translation of the axisymmetric equilibrium-air ISOQ model."""

from __future__ import annotations

import numpy as np
import sympy as sp


def _zeros(rows, columns, prototype):
    return np.full((rows, columns), 0 * prototype, dtype=object)


def _qmat(q):
    return np.reshape(np.asarray(q, dtype=object), (4, 2), order="F")


def _tau_entry(tau, index):
    return tau[0] if len(tau) == 1 else tau[index]


def _limited_maximum(value, alpha):
    return value * (sp.atan(alpha * value) / sp.pi + 0.5) - sp.atan(alpha) / sp.pi + 0.5


def _limiting(value, lower, upper, alpha, beta):
    lower_limited = lower + _limited_maximum(value - beta, alpha)
    return lower_limited - _limited_maximum(lower_limited - upper, alpha)


def mass(u, q, w, v, x, t, mu, eta):
    return np.ones(4)


def materialstate(u, q, w, v, x, t, mu, eta):
    rho = u[0]
    velocity_z = u[1] / rho
    velocity_r = u[2] / rho
    internal_energy = u[3] / rho - 0.5 * (
        velocity_z * velocity_z + velocity_r * velocity_r
    )
    limited_rho = _limiting(rho, 0.0001 / mu[0], 20.0 / mu[0], 1.0e2, 0.0)
    limited_energy = _limiting(
        internal_energy, -150000.0 / mu[3], 20000000.0 / mu[3], 1.0e2, 0.0
    )
    return np.array([sp.log(mu[0] * limited_rho), mu[3] * limited_energy])


def _artificial_viscosity(v, mu):
    return (mu[-2] + mu[-1] * v[1]) * sp.tanh(mu[-3] * v[0])


def _flow_data(u, q, w, x, mu):
    rho = u[0]
    velocity = np.array([u[1] / rho, u[2] / rho], dtype=object)
    pressure = w[0] / mu[2]
    enthalpy = (u[3] + pressure) / rho
    gradient_u = -_qmat(q)

    gradient_velocity = _zeros(3, 3, u[0])
    gradient_temperature = np.full(2, 0 * u[0], dtype=object)
    total_energy = u[3] / rho
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
        gradient_temperature[direction] = (
            w[8] * gradient_rho / rho
            + w[9] * mu[3] * gradient_internal_energy
        )
    gradient_velocity[2, 2] = velocity[1] / x[1]

    viscosity = mu[5] * w[2] / (mu[0] * mu[1] * mu[4])
    conductivity = (
        mu[5] * (w[3] + mu[6] * w[4]) / (mu[0] * mu[1] ** 3 * mu[4])
    )
    divergence = sum(gradient_velocity[index, index] for index in range(3))
    stress = _zeros(3, 3, u[0])
    for i in range(3):
        for j in range(3):
            delta = 1.0 if i == j else 0.0
            stress[i, j] = viscosity * (
                gradient_velocity[i, j]
                + gradient_velocity[j, i]
                - (2.0 / 3.0) * divergence * delta
            )
    return (
        rho,
        velocity,
        pressure,
        enthalpy,
        gradient_u,
        stress,
        conductivity,
        gradient_temperature,
    )


def flux(u, q, w, v, x, t, mu, eta):
    rho, velocity, pressure, enthalpy, gradient_u, stress, conductivity, gradient_temperature = (
        _flow_data(u, q, w, x, mu)
    )
    artificial_viscosity = _artificial_viscosity(v, mu)
    result = _zeros(4, 2, u[0])
    for direction in range(2):
        result[0, direction] = (
            rho * velocity[direction]
            - artificial_viscosity * gradient_u[0, direction]
        )
        for component in range(2):
            result[1 + component, direction] = (
                rho * velocity[component] * velocity[direction]
                - artificial_viscosity * gradient_u[1 + component, direction]
                - stress[component, direction]
            )
        result[1 + direction, direction] += pressure
        viscous_work = sum(
            velocity[component] * stress[component, direction]
            for component in range(2)
        )
        result[3, direction] = (
            rho * velocity[direction] * enthalpy
            - artificial_viscosity * gradient_u[3, direction]
            - viscous_work
            - conductivity * gradient_temperature[direction]
        )
    return result


def source(u, q, w, v, x, t, mu, eta):
    flux_value = flux(u, q, w, v, x, t, mu, eta)
    _, _, pressure, _, _, stress, _, _ = _flow_data(u, q, w, x, mu)
    result = -flux_value[:, 1] / x[1]
    result[2] += (pressure - stress[2, 2]) / x[1]
    return result


def avfield(u, q, w, v, x, t, mu, eta):
    rho = u[0]
    velocity_z = u[1] / rho
    velocity_r = u[2] / rho
    compression_z = (q[1] - q[0] * velocity_z) / rho
    compression_r = (q[6] - q[4] * velocity_r) / rho
    compression = compression_z + compression_r + velocity_r / x[1]
    return np.array(
        [_limiting(compression * sp.tanh(mu[-3] * v[0]), 0.0, mu[-4], 1.0e3, 0.0)]
    )


def _remove_normal_momentum(u, normal):
    nh = np.asarray(normal, dtype=object)
    nh = nh / sp.sqrt(nh @ nh)
    momentum = np.asarray(u[1:3], dtype=object)
    result = np.asarray(u, dtype=object).copy()
    result[1:3] = momentum - (momentum @ nh) * nh
    return result


def _wall_specific_energy(u, w, mu):
    rho = u[0]
    velocity_z = u[1] / rho
    velocity_r = u[2] / rho
    internal_energy = u[3] / rho - 0.5 * (
        velocity_z * velocity_z + velocity_r * velocity_r
    )
    return internal_energy + (mu[7] - w[1]) / (w[9] * mu[3])


def _wall_state(u, w, mu):
    result = np.asarray(u, dtype=object).copy()
    result[1:3] = 0 * u[0]
    result[3] = u[0] * _wall_specific_energy(u, w, mu)
    return result


def _boundary_states(u, q, w, mu, eta, normal):
    qn = _qmat(q) @ np.asarray(normal, dtype=object)
    inflow = np.asarray(eta[0:4], dtype=object)
    interior = np.asarray(u, dtype=object)
    isothermal = _wall_state(u, w, mu)
    adiabatic = interior.copy()
    adiabatic[1:3] = 0 * u[0]
    symmetry = _remove_normal_momentum(u, normal)
    return np.column_stack(
        (inflow, interior, isothermal, adiabatic, symmetry, interior)
    ), qn


def _normal_flux(u, q, w, v, x, mu, normal):
    return flux(u, q, w, v, x, 0, mu, 0) @ np.asarray(normal, dtype=object)


def _stabilized(state, uhat, tau):
    return np.array(
        [_tau_entry(tau, i) * (state[i] - uhat[i]) for i in range(4)],
        dtype=object,
    )


def _symmetry_residual(u, q, normal, tau, uhat):
    nh = np.asarray(normal, dtype=object)
    nh = nh / sp.sqrt(nh @ nh)
    qmatrix = _qmat(q)
    momentum_hat = np.asarray(uhat[1:3], dtype=object)
    normal_momentum = momentum_hat @ nh
    tangent_projection = np.eye(2, dtype=object) - np.outer(nh, nh)
    result = np.full(4, 0 * u[0], dtype=object)
    result[0] = qmatrix[0, :] @ nh + _tau_entry(tau, 0) * (u[0] - uhat[0])
    result[1:3] = normal_momentum * nh + tangent_projection @ (qmatrix[1:3, :] @ nh)
    result[3] = qmatrix[3, :] @ nh + _tau_entry(tau, 3) * (u[3] - uhat[3])
    return result


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    states, qn = _boundary_states(u, q, w, mu, eta, n)
    result = _zeros(4, 6, u[0])
    for column, state_column in ((0, 0), (1, 1), (2, 2), (3, 3)):
        state = states[:, state_column]
        result[:, column] = _normal_flux(state, q, w, v, x, mu, n) + _stabilized(
            state, uhat, tau
        )
    result[0, 3] = 0 * u[0]
    result[3, 3] = 0 * u[0]
    result[:, 4] = _symmetry_residual(u, q, n, tau, uhat)
    result[:, 5] = qn + _stabilized(u, uhat, tau)
    return result


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return _boundary_states(u, q, w, mu, eta, n)[0]


def fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    inflow = np.asarray(eta[0:4], dtype=object)
    interior = np.asarray(u, dtype=object)
    isothermal = np.full(4, 0 * u[0], dtype=object)
    isothermal[0] = u[0] - uhat[0]
    isothermal[1:3] = -np.asarray(uhat[1:3], dtype=object)
    isothermal[3] = uhat[0] * _wall_specific_energy(u, w, mu) - uhat[3]
    slip = _remove_normal_momentum(u, n)
    qn = _qmat(q) @ np.asarray(n, dtype=object)
    no_slip = np.full(4, 0 * u[0], dtype=object)
    no_slip[0] = u[0] - uhat[0]
    no_slip[1:3] = -np.asarray(uhat[1:3], dtype=object)
    symmetry = qn + _stabilized(interior, uhat, tau)
    symmetry[1:3] = _stabilized(slip, uhat, tau)[1:3]
    return np.column_stack(
        (
            inflow - uhat,
            interior - uhat,
            isothermal,
            symmetry,
            _stabilized(slip, uhat, tau),
            qn + _stabilized(interior, uhat, tau),
            no_slip,
        )
    )


def initu(x, mu, eta):
    return np.asarray(eta[0:4], dtype=object)


def initv(x, mu, eta):
    return np.array([0.0, 0.0])


def initw(x, mu, eta):
    return np.array(
        [
            1.0, 300.0, 1.0e-5, 1.0e-2, 0.0, 300.0, 1.0, 1.0e-3,
            0.0, 1.0e-3, 0.767, 0.233, 0.0, 0.0, 0.0,
        ]
    )


def visscalars(u, q, w, v, x, t, mu, eta):
    rho = u[0]
    velocity = np.asarray(u[1:3], dtype=object) / rho
    speed = mu[1] * sp.sqrt(velocity @ velocity)
    mach = speed / w[5]
    return np.array(
        [
            mu[0] * rho, w[0], w[1], mach, _artificial_viscosity(v, mu),
            w[13], w[14], w[12], w[10], w[11],
        ],
        dtype=object,
    )


def visvectors(u, q, w, v, x, t, mu, eta):
    return mu[1] * np.asarray(u[1:3], dtype=object) / u[0]
