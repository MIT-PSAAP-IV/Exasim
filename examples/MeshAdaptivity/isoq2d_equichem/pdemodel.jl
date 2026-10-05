"""Julia translation of the axisymmetric equilibrium-air ISOQ model."""

using LinearAlgebra

zero_vector(n, prototype) = fill(0 * prototype, n)
zero_matrix(m, n, prototype) = fill(0 * prototype, m, n)
q_matrix(q) = reshape(collect(q), 4, 2)
tau_entry(tau, i) = length(tau) == 1 ? tau[1] : tau[i]

limited_maximum(value, alpha) =
    value * (atan(alpha * value) / pi + 0.5) - atan(alpha) / pi + 0.5

function limiting_value(value, lower, upper, alpha, beta)
    lower_limited = lower + limited_maximum(value - beta, alpha)
    return lower_limited - limited_maximum(lower_limited - upper, alpha)
end

mass(u, q, w, v, x, t, mu, eta) = ones(4)

function materialstate(u, q, w, v, x, t, mu, eta)
    rho = u[1]
    velocity_z = u[2] / rho
    velocity_r = u[3] / rho
    internal_energy = u[4] / rho - 0.5 * (velocity_z^2 + velocity_r^2)
    limited_rho = limiting_value(rho, 0.0001 / mu[1], 20.0 / mu[1], 1.0e2, 0.0001 / mu[1])
    limited_energy = limiting_value(
        internal_energy, -150000.0 / mu[4], 20000000.0 / mu[4], 1.0e2, -150000.0 / mu[4]
    )
    return [log(mu[1] * limited_rho), mu[4] * limited_energy]
end

artificial_viscosity(v, mu) =
    (mu[end-1] + mu[end] * v[2]) * tanh(mu[end-2] * v[1])

function flow_data(u, q, w, x, mu)
    rho = u[1]
    velocity = [u[2] / rho, u[3] / rho]
    pressure = w[1] / mu[3]
    enthalpy = (u[4] + pressure) / rho
    gradient_u = -q_matrix(q)

    gradient_velocity = zero_matrix(3, 3, u[1])
    gradient_temperature = zero_vector(2, u[1])
    total_energy = u[4] / rho
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
        gradient_temperature[direction] =
            w[9] * (gradient_rho / rho) +
            w[10] * mu[4] * gradient_internal_energy
    end
    gradient_velocity[3, 3] = velocity[2] / x[2]

    viscosity = mu[6] * w[3] / (mu[1] * mu[2] * mu[5])
    conductivity =
        mu[6] * (w[4] + mu[7] * w[5]) / (mu[1] * mu[2]^3 * mu[5])
    divergence = sum(gradient_velocity[index, index] for index in 1:3)
    stress = zero_matrix(3, 3, u[1])
    for i in 1:3, j in 1:3
        delta = i == j ? 1.0 : 0.0
        stress[i, j] = viscosity * (
            gradient_velocity[i, j] + gradient_velocity[j, i] -
            (2.0 / 3.0) * divergence * delta
        )
    end
    return rho, velocity, pressure, enthalpy, gradient_u, stress,
           conductivity, gradient_temperature
end

function flux(u, q, w, v, x, t, mu, eta)
    rho, velocity, pressure, enthalpy, gradient_u, stress,
        conductivity, gradient_temperature = flow_data(u, q, w, x, mu)
    av = artificial_viscosity(v, mu)
    result = zero_matrix(4, 2, u[1])
    for direction in 1:2
        result[1, direction] =
            rho * velocity[direction] - av * gradient_u[1, direction]
        for component in 1:2
            result[1 + component, direction] =
                rho * velocity[component] * velocity[direction] -
                av * gradient_u[1 + component, direction] -
                stress[component, direction]
        end
        result[1 + direction, direction] += pressure
        viscous_work = sum(
            velocity[component] * stress[component, direction]
            for component in 1:2
        )
        result[4, direction] =
            rho * velocity[direction] * enthalpy -
            av * gradient_u[4, direction] - viscous_work -
            conductivity * gradient_temperature[direction]
    end
    return result
end

function source(u, q, w, v, x, t, mu, eta)
    flux_value = flux(u, q, w, v, x, t, mu, eta)
    _, _, pressure, _, _, stress, _, _ = flow_data(u, q, w, x, mu)
    result = -flux_value[:, 2] / x[2]
    result[3] += (pressure - stress[3, 3]) / x[2]
    return result
end

function avfield(u, q, w, v, x, t, mu, eta)
    rho = u[1]
    velocity_z = u[2] / rho
    velocity_r = u[3] / rho
    compression_z = (q[2] - q[1] * velocity_z) / rho
    compression_r = (q[7] - q[5] * velocity_r) / rho
    compression = compression_z + compression_r + velocity_r / x[2]
    return [
        limiting_value(
            compression * tanh(mu[end-2] * v[1]), 0.0, mu[end-3], 1.0e3, 0.0
        ),
        w[1],
    ]
end

function remove_normal_momentum(u, normal)
    nh = collect(normal) / sqrt(sum(value^2 for value in normal))
    momentum = collect(u[2:3])
    result = collect(u)
    result[2:3] .= momentum .- sum(momentum .* nh) .* nh
    return result
end

function wall_specific_energy(u, w, mu)
    rho = u[1]
    velocity_z = u[2] / rho
    velocity_r = u[3] / rho
    internal_energy = u[4] / rho - 0.5 * (velocity_z^2 + velocity_r^2)
    return internal_energy + (mu[8] - w[2]) / (w[10] * mu[4])
end

function wall_state(u, w, mu)
    result = collect(u)
    result[2:3] .= 0 * u[1]
    result[4] = u[1] * wall_specific_energy(u, w, mu)
    return result
end

function boundary_states(u, q, w, mu, eta, normal)
    qn = q_matrix(q) * collect(normal)
    inflow = collect(eta[1:4])
    interior = collect(u)
    isothermal = wall_state(u, w, mu)
    adiabatic = collect(u)
    adiabatic[2:3] .= 0 * u[1]
    symmetry = remove_normal_momentum(u, normal)
    return hcat(inflow, interior, isothermal, adiabatic, symmetry, interior), qn
end

normal_flux(u, q, w, v, x, mu, normal) =
    flux(u, q, w, v, x, 0, mu, 0) * collect(normal)

stabilized(state, uhat, tau) =
    [tau_entry(tau, i) * (state[i] - uhat[i]) for i in 1:4]

function symmetry_residual(u, q, normal, tau, uhat)
    nh = collect(normal) / sqrt(sum(value^2 for value in normal))
    qmatrix = q_matrix(q)
    momentum_hat = collect(uhat[2:3])
    normal_momentum = sum(momentum_hat .* nh)
    tangent_projection = [1.0 0.0; 0.0 1.0] - nh * transpose(nh)
    result = zero_vector(4, u[1])
    result[1] = sum(qmatrix[1, :] .* nh) +
                tau_entry(tau, 1) * (u[1] - uhat[1])
    result[2:3] .= normal_momentum .* nh .+
                     tangent_projection * (qmatrix[2:3, :] * nh)
    result[4] = sum(qmatrix[4, :] .* nh) +
                tau_entry(tau, 4) * (u[4] - uhat[4])
    return result
end

function fbou(u, q, w, v, x, t, mu, eta, uhat, normal, tau)
    states, qn = boundary_states(u, q, w, mu, eta, normal)
    result = zero_matrix(4, 6, u[1])
    for column in 1:4
        state = states[:, column]
        result[:, column] = normal_flux(state, q, w, v, x, mu, normal) +
                            stabilized(state, uhat, tau)
    end
    result[1, 4] = 0 * u[1]
    result[4, 4] = 0 * u[1]
    result[:, 5] = symmetry_residual(u, q, normal, tau, uhat)
    result[:, 6] = qn + stabilized(u, uhat, tau)
    return result
end

ubou(u, q, w, v, x, t, mu, eta, uhat, normal, tau) =
    first(boundary_states(u, q, w, mu, eta, normal))

function fbouhdg(u, q, w, v, x, t, mu, eta, uhat, normal, tau)
    inflow = collect(eta[1:4])
    interior = collect(u)
    isothermal = zero_vector(4, u[1])
    isothermal[1] = u[1] - uhat[1]
    isothermal[2:3] .= -collect(uhat[2:3])
    isothermal[4] = uhat[1] * wall_specific_energy(u, w, mu) - uhat[4]
    slip = remove_normal_momentum(u, normal)
    qn = q_matrix(q) * collect(normal)
    no_slip = zero_vector(4, u[1])
    no_slip[1] = u[1] - uhat[1]
    no_slip[2:3] .= -collect(uhat[2:3])
    symmetry = qn + stabilized(interior, uhat, tau)
    symmetry[2:3] .= stabilized(slip, uhat, tau)[2:3]
    return hcat(
        inflow - uhat,
        interior - uhat,
        isothermal,
        symmetry,
        stabilized(slip, uhat, tau),
        qn + stabilized(interior, uhat, tau),
        no_slip,
    )
end

initu(x, mu, eta) = collect(eta[1:4])
initv(x, mu, eta) = [0.0, 0.0]

function initw(x, mu, eta)
    return [
        1.0, 300.0, 1.0e-5, 1.0e-2, 0.0, 300.0, 1.0, 1.0e-3,
        0.0, 1.0e-3, 0.767, 0.233, 0.0, 0.0, 0.0,
    ]
end

function visscalars(u, q, w, v, x, t, mu, eta)
    rho = u[1]
    velocity = collect(u[2:3]) / rho
    speed = mu[2] * sqrt(sum(value^2 for value in velocity))
    mach = speed / w[6]
    return [
        mu[1] * rho, w[1], w[2], mach, artificial_viscosity(v, mu),
        w[14], w[15], w[13], w[11], w[12],
    ]
end

visvectors(u, q, w, v, x, t, mu, eta) = mu[2] .* collect(u[2:3]) ./ u[1]

function surfacequantities(u, q, w, v, x, t, mu, eta, uhat, normal, tau)
    _, _, pressure, _, _, stress, conductivity, temperature_gradient =
        flow_data(uhat, q, w, x, mu)

    density_inf = eta[1]
    velocity_inf = collect(eta[2:3]) ./ density_inf
    speed_inf2 = sum(velocity_inf .* velocity_inf)
    dynamic_pressure = 0.5 * density_inf * speed_inf2
    heat_flux_reference = density_inf * speed_inf2 * sqrt(speed_inf2)
    pressure_inf = mu[9] / mu[3]

    nvec = collect(normal)
    tangent = [-nvec[2], nvec[1]]
    traction = stress[1:2, 1:2] * nvec
    traction[1] += tau_entry(tau, 2) * (u[2] - uhat[2])
    traction[2] += tau_entry(tau, 3) * (u[3] - uhat[3])
    heat_flux = conductivity * sum(temperature_gradient .* nvec) +
                tau_entry(tau, 4) * (u[4] - uhat[4])

    [
        (pressure - pressure_inf) / dynamic_pressure,
        sum(tangent .* traction) / dynamic_pressure,
        heat_flux / heat_flux_reference,
    ]
end
