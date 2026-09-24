"""Two-dimensional equilibrium-air model matching pdemodel.m."""

using LinearAlgebra

zero_vector(n, prototype) = fill(0 * prototype, n)
zero_matrix(m, n, prototype) = fill(0 * prototype, m, n)
q_matrix(q) = reshape(collect(q), 4, 2)
tau_entry(tau, i) = length(tau) == 1 ? tau[1] : tau[i]

mass(u, q, w, v, x, t, mu, eta) = ones(4)

function materialstate(u, q, w, v, x, t, mu, eta)
    rho = u[1]
    velocity_x = u[2] / rho
    velocity_y = u[3] / rho
    internal_energy = u[4] / rho - 0.5 * (velocity_x^2 + velocity_y^2)
    return [log(mu[1] * rho), mu[4] * internal_energy]
end

function state_values(u, w, mu)
    rho = u[1]
    velocity = [u[2] / rho, u[3] / rho]
    rho_energy = u[4]
    pressure = w[1] / mu[3]
    enthalpy = (rho_energy + pressure) / rho
    return rho, velocity, rho_energy, pressure, enthalpy
end

function gradient_primitives(u, gradient_u, w, mu)
    rho = u[1]
    velocity = [u[2] / rho, u[3] / rho]
    total_energy = u[4] / rho
    gradient_velocity = zero_matrix(2, 2, u[1])
    gradient_temperature = zero_vector(2, u[1])
    for direction in 1:2
        gradient_rho = gradient_u[1, direction]
        for component in 1:2
            gradient_velocity[component, direction] =
                (gradient_u[1 + component, direction] -
                 gradient_rho * velocity[component]) / rho
        end
        gradient_energy =
            (gradient_u[4, direction] - gradient_rho * total_energy) / rho
        gradient_internal_energy = gradient_energy
        for component in 1:2
            gradient_internal_energy -=
                velocity[component] * gradient_velocity[component, direction]
        end
        gradient_xi = gradient_rho / rho
        gradient_temperature[direction] =
            w[9] * gradient_xi + w[10] * mu[4] * gradient_internal_energy
    end
    return gradient_velocity, gradient_temperature
end

function stress_heat(velocity, gradient_velocity, w, mu)
    viscosity = mu[6] * w[3] / (mu[1] * mu[2] * mu[5])
    conductivity =
        mu[6] * (w[4] + mu[7] * w[5]) / (mu[1] * mu[2]^3 * mu[5])
    divergence = gradient_velocity[1, 1] + gradient_velocity[2, 2]
    stress = zero_matrix(2, 2, velocity[1])
    for i in 1:2, j in 1:2
        delta = i == j ? 1.0 : 0.0
        stress[i, j] = viscosity *
            (gradient_velocity[i, j] + gradient_velocity[j, i] -
             (2.0 / 3.0) * divergence * delta)
    end
    return stress, conductivity
end

function flux_cartesian(u, q, w, v, x, mu)
    rho, velocity, _, pressure, enthalpy = state_values(u, w, mu)
    gradient_u = -q_matrix(q)
    gradient_velocity, gradient_temperature =
        gradient_primitives(u, gradient_u, w, mu)
    stress, conductivity = stress_heat(velocity, gradient_velocity, w, mu)
    artificial_viscosity =
        (mu[end-1] + mu[end] * v[2]) * tanh(mu[end-2] * v[1])

    result = zero_matrix(4, 2, u[1])
    for direction in 1:2
        result[1, direction] = rho * velocity[direction] -
                               artificial_viscosity * gradient_u[1, direction]
        for component in 1:2
            result[1 + component, direction] =
                rho * velocity[component] * velocity[direction] -
                artificial_viscosity * gradient_u[1 + component, direction]
        end
        result[1 + direction, direction] += pressure
        result[4, direction] = rho * velocity[direction] * enthalpy -
                               artificial_viscosity * gradient_u[4, direction]
    end
    for direction in 1:2
        heat_work = 0 * u[1]
        for component in 1:2
            result[1 + component, direction] -= stress[component, direction]
            heat_work += velocity[component] * stress[component, direction]
        end
        result[4, direction] -=
            heat_work + conductivity * gradient_temperature[direction]
    end
    return result
end

flux(u, q, w, v, x, t, mu, eta) = flux_cartesian(u, q, w, v, x, mu)
source(u, q, w, v, x, t, mu, eta) = zero_vector(4, u[1])

function normal_basis(normal)
    norm = sqrt(normal[1]^2 + normal[2]^2)
    nx = normal[1] / norm
    ny = normal[2] / norm
    return [nx ny; -ny nx]
end

function remove_normal_momentum(u, normal)
    basis = normal_basis(normal)
    nh = basis[1, :]
    momentum = collect(u[2:3])
    normal_momentum = sum(momentum .* nh)
    result = collect(u)
    result[2:3] .= momentum .- normal_momentum .* nh
    return result
end

function wall_state_temperature(u, w, mu)
    rho = u[1]
    velocity_x = u[2] / rho
    velocity_y = u[3] / rho
    internal_energy = u[4] / rho - 0.5 * (velocity_x^2 + velocity_y^2)
    wall_energy = internal_energy + (mu[8] - w[2]) / (w[10] * mu[4])
    result = collect(u)
    result[2] = 0 * u[1]
    result[3] = 0 * u[1]
    result[4] = rho * wall_energy
    return result
end

outlet_pressure(interior_pressure, mu) = length(mu) >= 9 ? mu[9] : interior_pressure
outlet_relaxation(mu) = length(mu) >= 10 ? mu[10] : 1.0

function outlet_state(u, w, normal, mu)
    rho = u[1]
    basis = normal_basis(normal)
    local_momentum = basis * collect(u[2:3])
    local_velocity = local_momentum / rho
    normal_velocity = local_velocity[1]
    total_energy = u[4] / rho
    sound_speed = w[6] / mu[2]
    pressure_rho = w[7] / (rho * mu[3])
    pressure_energy = mu[4] * w[8] / mu[3]
    acoustic_energy = total_energy +
        rho * (sound_speed^2 - pressure_rho) / pressure_energy
    pressure_delta = outlet_relaxation(mu) *
        (outlet_pressure(w[1], mu) - w[1]) / mu[3]
    density_delta = pressure_delta / sound_speed^2
    right_eigenvector = [
        1.0,
        normal_velocity - sound_speed,
        local_velocity[2],
        acoustic_energy - normal_velocity * sound_speed,
    ]
    transform_inverse = zero_matrix(4, 4, u[1])
    transform_inverse[1, 1] = 1.0
    transform_inverse[2:3, 2:3] = transpose(basis)
    transform_inverse[4, 4] = 1.0
    return collect(u) + density_delta * transform_inverse * right_eigenvector
end

function boundary_states(u, q, w, x, mu, eta, normal)
    qn = q_matrix(q) * collect(normal)
    inflow = collect(eta[1:4])
    outflow = outlet_state(u, w, normal, mu)
    isothermal = wall_state_temperature(u, w, mu)
    adiabatic = collect(u)
    adiabatic[2:3] .= 0 * u[1]
    symmetry = remove_normal_momentum(u, normal)
    gradient = collect(u)
    return hcat(
        inflow, outflow, isothermal, adiabatic, symmetry, gradient, inflow, outflow
    ), qn
end

normal_flux(u, q, w, v, x, mu, normal) =
    flux_cartesian(u, q, w, v, x, mu) * collect(normal)

function sign_matrix(state, normal, w, mu)
    rho = state[1]
    basis = normal_basis(normal)
    local_momentum = basis * collect(state[2:3])
    local_velocity = local_momentum / rho
    normal_velocity = local_velocity[1]
    total_energy = state[4] / rho
    sound_speed = w[6] / mu[2]
    pressure_rho = w[7] / (rho * mu[3])
    pressure_energy = mu[4] * w[8] / mu[3]
    bvalue = total_energy + rho * (sound_speed^2 - pressure_rho) / pressure_energy
    cvalue = total_energy - rho * pressure_rho / pressure_energy

    eigenvectors = zero_matrix(4, 4, state[1])
    eigenvectors[:, 1] = [
        1.0,
        normal_velocity - sound_speed,
        local_velocity[2],
        bvalue - normal_velocity * sound_speed,
    ]
    eigenvectors[:, 2] = [1.0, normal_velocity, local_velocity[2], cvalue]
    eigenvectors[:, 3] = [0.0, 0.0, 1.0, local_velocity[2]]
    eigenvectors[:, 4] = [
        1.0,
        normal_velocity + sound_speed,
        local_velocity[2],
        bvalue + normal_velocity * sound_speed,
    ]

    transform = zero_matrix(4, 4, state[1])
    transform[1, 1] = 1.0
    transform[2:3, 2:3] = basis
    transform[4, 4] = 1.0
    transform_inverse = zero_matrix(4, 4, state[1])
    transform_inverse[1, 1] = 1.0
    transform_inverse[2:3, 2:3] = transpose(basis)
    transform_inverse[4, 4] = 1.0
    eigenvalue_sign = zero_matrix(4, 4, state[1])
    eigenvalue_sign[1, 1] = tanh(100 * (normal_velocity - sound_speed))
    eigenvalue_sign[2, 2] = tanh(100 * normal_velocity)
    eigenvalue_sign[3, 3] = tanh(100 * normal_velocity)
    eigenvalue_sign[4, 4] = tanh(100 * (normal_velocity + sound_speed))
    return transform_inverse * eigenvectors * eigenvalue_sign * inv(eigenvectors) * transform
end

function characteristic_state(u, exterior, linearization, normal, w, mu)
    sign = sign_matrix(linearization, normal, w, mu)
    return 0.5 .* (collect(u) .+ collect(exterior) .+
                   sign * (collect(u) .- collect(exterior)))
end

function symmetry_residual(u, q, x, normal, tau, uhat)
    basis = normal_basis(normal)
    nh = basis[1, :]
    qmatrix = q_matrix(q)
    momentum_hat = collect(uhat[2:3])
    normal_momentum = sum(momentum_hat .* nh)
    normal_momentum_gradient = qmatrix[2:3, :] * nh
    tangent_projection = [1.0 0.0; 0.0 1.0] - nh * transpose(nh)
    tangent_gradient = tangent_projection * normal_momentum_gradient
    residual = zero_vector(4, u[1])
    residual[1] = sum(qmatrix[1, :] .* nh) + tau_entry(tau, 1) * (u[1] - uhat[1])
    residual[2:3] .= normal_momentum .* nh .+ tangent_gradient
    residual[4] = sum(qmatrix[4, :] .* nh) + tau_entry(tau, 4) * (u[4] - uhat[4])
    return residual
end

function stabilized(state, uhat, tau)
    return [tau_entry(tau, i) * (state[i] - uhat[i]) for i in 1:4]
end

function fbou(u, q, w, v, x, t, mu, eta, uhat, normal, tau)
    states, qn = boundary_states(u, q, w, x, mu, eta, normal)
    result = zero_matrix(4, 8, u[1])
    interior = states[:, 6]
    for (column, state_column) in ((1, 1), (2, 6), (3, 3))
        state = states[:, state_column]
        result[:, column] = normal_flux(state, q, w, v, x, mu, normal) +
                            stabilized(state, uhat, tau)
    end
    result[:, 4] = normal_flux(states[:, 4], q, w, v, x, mu, normal) +
                    stabilized(states[:, 4], uhat, tau)
    result[1, 4] = 0 * u[1]
    result[4, 4] = 0 * u[1]
    result[:, 5] = symmetry_residual(interior, q, x, normal, tau, uhat)
    result[:, 6] = qn + stabilized(interior, uhat, tau)
    inflow = characteristic_state(interior, states[:, 7], interior, normal, w, mu)
    outflow = characteristic_state(interior, states[:, 8], interior, normal, w, mu)
    result[:, 7] = normal_flux(interior, q, w, v, x, mu, normal) +
                    stabilized(interior, inflow, tau)
    result[:, 8] = normal_flux(interior, q, w, v, x, mu, normal) +
                    stabilized(interior, outflow, tau)
    return result
end

function fbouhdg(u, q, w, v, x, t, mu, eta, uhat, normal, tau)
    states, qn = boundary_states(u, q, w, x, mu, eta, normal)
    result = zero_matrix(4, 8, u[1])
    for column in 1:8
        result[:, column] = stabilized(states[:, column], uhat, tau)
    end
    result[:, 1] = states[:, 1] - uhat
    result[:, 2] = states[:, 6] - uhat
    result[:, 7] = characteristic_state(
        states[:, 6], states[:, 7], uhat, normal, w, mu
    ) - uhat
    result[:, 8] = characteristic_state(
        states[:, 6], states[:, 8], uhat, normal, w, mu
    ) - uhat
    adiabatic = zero_vector(4, u[1])
    adiabatic[1] = states[1, 4] - uhat[1]
    adiabatic[2:3] = -collect(uhat[2:3])
    fn_hat = normal_flux(uhat, q, w, v, x, mu, normal)
    adiabatic[4] = fn_hat[4] + tau_entry(tau, 4) * (states[4, 4] - uhat[4])
    result[:, 4] = adiabatic
    result[:, 5] = symmetry_residual(states[:, 6], q, x, normal, tau, uhat)
    result[:, 6] = qn + stabilized(states[:, 6], uhat, tau)
    return result
end

ubou(u, q, w, v, x, t, mu, eta, uhat, normal, tau) =
    first(boundary_states(u, q, w, x, mu, eta, normal))
initu(x, mu, eta) = collect(eta[1:4])
initv(x, mu, eta) = [0.0, 0.0]

function initw(x, mu, eta)
    return [
        1.0, 300.0, 1.0e-5, 1.0e-2, 0.0, 300.0, 1.0, 1.0e-3,
        0.0, 1.0e-3, 0.767, 0.233, 0.0, 0.0, 0.0,
    ]
end

visscalars(u, q, w, v, x, t, mu, eta) = [w[1] / mu[3]]

limited_maximum(value, alpha) =
    value * (atan(alpha * value) / pi + 0.5) - atan(alpha) / pi + 0.5

function limiting_value(value, lower, upper, alpha, beta)
    lower_limited = lower + limited_maximum(value - beta, alpha)
    return lower_limited - limited_maximum(lower_limited - upper, alpha)
end

function avfield(u, q, w, v, x, t, mu, eta)
    rho = u[1]
    velocity_x = u[2] / rho
    velocity_y = u[3] / rho
    gradient = q_matrix(q)
    velocity_x_gradient = (gradient[2, 1] - gradient[1, 1] * velocity_x) / rho
    velocity_y_gradient = (gradient[3, 2] - gradient[1, 2] * velocity_y) / rho
    divergence = velocity_x_gradient + velocity_y_gradient
    return [limiting_value(
        divergence * tanh(mu[end-2] * v[1]), 0.0, mu[end-3], 1.0e3, 0.0
    )]
end
