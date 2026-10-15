"""Julia translation of the axisymmetric ideal-gas ISOQ model."""

lmax_value(value, alpha) = value * (atan(alpha * value) / pi + 0.5) - atan(alpha) / pi + 0.5
lmin_value(value, alpha) = value - lmax_value(value, alpha)

function limiting_value(value, lower, upper, alpha, beta)
    value = lower + lmax_value(value - beta, alpha)
    lmin_value(value - upper, alpha) + upper
end

function viscosity_value(reference_viscosity, reference_temperature, temperature)
    sutherland_temperature = 110.4
    reference_viscosity * (temperature / reference_temperature)^1.5 *
    (reference_temperature + sutherland_temperature) /
    (temperature + sutherland_temperature)
end

mass(u, q, w, v, x, t, mu, eta) = ones(4)

function flow_quantities(u, q, mu)
    gam1 = mu[1] - 1.0
    alpha, density_minimum, pressure_minimum = 4.0e3, 5.0e-2, 2.0e-3
    density, momentum_x, momentum_y, total_energy_density = u
    density_x, momentum_x_x, momentum_y_x, energy_x,
    density_y, momentum_x_y, momentum_y_y, energy_y = q
    density = density_minimum + lmax_value(density - density_minimum, alpha)
    density_derivative = atan(alpha * (density - density_minimum)) / pi +
        alpha * (density - density_minimum) /
        (pi * (alpha^2 * (density - density_minimum)^2 + 1.0)) + 0.5
    density_x *= density_derivative
    density_y *= density_derivative
    inverse_density = 1.0 / density
    velocity_x = momentum_x * inverse_density
    velocity_y = momentum_y * inverse_density
    total_energy = total_energy_density * inverse_density
    kinetic_energy = 0.5 * (velocity_x^2 + velocity_y^2)
    pressure = gam1 * (total_energy_density - density * kinetic_energy)
    pressure = pressure_minimum + lmax_value(pressure - pressure_minimum, alpha)
    pressure_derivative = atan(alpha * (pressure - pressure_minimum)) / pi +
        alpha * (pressure - pressure_minimum) /
        (pi * (alpha^2 * (pressure - pressure_minimum)^2 + 1.0)) + 0.5
    velocity_x_x = (momentum_x_x - density_x * velocity_x) * inverse_density
    velocity_y_x = (momentum_y_x - density_x * velocity_y) * inverse_density
    velocity_x_y = (momentum_x_y - density_y * velocity_x) * inverse_density
    velocity_y_y = (momentum_y_y - density_y * velocity_y) * inverse_density
    kinetic_x = velocity_x * velocity_x_x + velocity_y * velocity_y_x
    kinetic_y = velocity_x * velocity_x_y + velocity_y * velocity_y_y
    pressure_x = gam1 * (energy_x - density_x * kinetic_energy - density * kinetic_x) * pressure_derivative
    pressure_y = gam1 * (energy_y - density_y * kinetic_energy - density * kinetic_y) * pressure_derivative
    temperature_x = (pressure_x * density - pressure * density_x) * inverse_density^2 / gam1
    temperature_y = (pressure_y * density - pressure * density_y) * inverse_density^2 / gam1
    return (
        density, momentum_x, momentum_y, total_energy_density, density_x, density_y,
        velocity_x, velocity_y, total_energy, kinetic_energy, pressure,
        velocity_x_x, velocity_y_x, velocity_x_y, velocity_y_y,
        temperature_x, temperature_y,
    )
end

function flux(u, q, w, v, x, t, mu, eta)
    gam, reynolds, prandtl, mach_inf = mu[1:4]
    gam1 = gam - 1.0
    reference_temperature = mu[10]
    tinf = 1.0 / (gam * gam1 * mach_inf^2)
    artificial_viscosity = (mu[end-1] + mu[end] * v[2]) * tanh(mu[end-2] * v[1])
    density, momentum_x, momentum_y, _, density_x, density_y,
    velocity_x, velocity_y, total_energy, _, pressure,
    velocity_x_x, velocity_y_x, velocity_x_y, velocity_y_y,
    temperature_x, temperature_y = flow_quantities(u, q, mu)
    enthalpy = total_energy + pressure / density
    inviscid = [
        momentum_x,
        momentum_x * velocity_x + pressure,
        momentum_y * velocity_x,
        momentum_x * enthalpy,
        momentum_y,
        momentum_x * velocity_y,
        momentum_y * velocity_y + pressure,
        momentum_y * enthalpy,
    ]
    temperature = pressure / (gam1 * density)
    physical_temperature = reference_temperature * temperature / tinf
    dynamic_viscosity = viscosity_value(1.0 / reynolds, reference_temperature, physical_temperature)
    conductivity = dynamic_viscosity * gam / prandtl
    radial_coordinate = x[2]
    stress_xx = dynamic_viscosity * (2.0 / 3.0) *
        (2.0 * velocity_x_x - velocity_y_y + velocity_y / radial_coordinate)
    stress_xy = dynamic_viscosity * (velocity_x_y + velocity_y_x)
    stress_yy = dynamic_viscosity * (2.0 / 3.0) *
        (2.0 * velocity_y_y - velocity_x_x + velocity_y / radial_coordinate)
    viscous = [
        0.0,
        stress_xx,
        stress_xy,
        velocity_x * stress_xx + velocity_y * stress_xy + conductivity * temperature_x,
        0.0,
        stress_xy,
        stress_yy,
        velocity_x * stress_xy + velocity_y * stress_yy + conductivity * temperature_y,
    ]
    reshape(inviscid + viscous + artificial_viscosity .* q, 4, 2)
end

function avfield(u, q, w, v, x, t, mu, eta)
    density = u[1]
    velocity_x, velocity_y = u[2] / density, u[3] / density
    velocity_x_x = (q[2] - q[1] * velocity_x) / density
    velocity_y_y = (q[7] - q[5] * velocity_y) / density
    compression = velocity_x_x + velocity_y_y + velocity_y / x[2]
    gam, reynolds, mach_inf, reference_temperature = mu[1], mu[2], mu[4], mu[10]
    gas_constant, standard_temperature, sutherland_temperature = 287.0, 273.15, 110.4
    standard_viscosity = 1.716e-5
    pressure = (gam - 1.0) *
        (u[4] - 0.5 * (u[2] * velocity_x + u[3] * velocity_y))
    temperature = pressure / ((gam - 1.0) * density)
    tinf = 1.0 / (gam * (gam - 1.0) * mach_inf^2)
    physical_temperature = reference_temperature * temperature / tinf
    velocity_reference = mach_inf * sqrt(gam * gas_constant * reference_temperature)
    ratio = reference_temperature / standard_temperature
    physical_viscosity = standard_viscosity * sqrt(ratio^3) *
        (standard_temperature + sutherland_temperature) /
        (reference_temperature + sutherland_temperature)
    density_reference = reynolds * physical_viscosity / velocity_reference
    physical_density = density_reference * density
    physical_pressure = physical_density * gas_constant * physical_temperature
    [
        limiting_value(compression * tanh(mu[end-2] * v[1]), 0.0, mu[end-3], 1.0e3, 0.0),
        physical_pressure,
    ]
end

function source(u, q, w, v, x, t, mu, eta)
    gam, reynolds, prandtl, mach_inf = mu[1:4]
    gam1 = gam - 1.0
    reference_temperature = mu[10]
    tinf = 1.0 / (gam * gam1 * mach_inf^2)
    artificial_viscosity = (mu[end-1] + mu[end] * v[2]) * tanh(mu[end-2] * v[1])
    density, momentum_x, momentum_y, _, _, _, velocity_x, velocity_y,
    total_energy, _, pressure, velocity_x_x, velocity_y_x, velocity_x_y,
    velocity_y_y, _, temperature_y = flow_quantities(u, q, mu)
    enthalpy = total_energy + pressure / density
    inviscid_y = [
        momentum_y, momentum_x * velocity_y, momentum_y * velocity_y, momentum_y * enthalpy
    ]
    temperature = pressure / (gam1 * density)
    physical_temperature = reference_temperature * temperature / tinf
    dynamic_viscosity = viscosity_value(1.0 / reynolds, reference_temperature, physical_temperature)
    conductivity = dynamic_viscosity * gam / prandtl
    radial_coordinate = x[2]
    stress_xy = dynamic_viscosity * (velocity_x_y + velocity_y_x)
    stress_yy = dynamic_viscosity * (2.0 / 3.0) *
        (2.0 * velocity_y_y - velocity_x_x + velocity_y / radial_coordinate)
    stress_theta = dynamic_viscosity * (2.0 / 3.0) *
        (-2.0 * velocity_y / radial_coordinate - velocity_x_x - velocity_y_y)
    viscous_y = [
        0.0,
        stress_xy,
        stress_yy - stress_theta,
        velocity_x * stress_xy + velocity_y * stress_yy + conductivity * temperature_y,
    ]
    -(inviscid_y + viscous_y + artificial_viscosity .* q[5:8]) / radial_coordinate
end

fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = 0 .* u
ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = 0 .* u

function fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    gam1 = mu[1] - 1.0
    isothermal_temperature = mu[11] / mu[10] * mu[9]
    freestream = mu[5:8]
    inflow, outflow = freestream - uhat, u - uhat
    isothermal = 0 * u
    isothermal[1] = u[1] - uhat[1]
    isothermal[2] = -uhat[2]
    isothermal[3] = -uhat[3]
    isothermal[4] = -uhat[4] + uhat[1] * isothermal_temperature
    density, momentum_x, momentum_y, energy = uhat
    inverse_density = 1.0 / density
    velocity_x, velocity_y = momentum_x * inverse_density, momentum_y * inverse_density
    kinetic_energy = 0.5 * (velocity_x^2 + velocity_y^2)
    pressure = gam1 * (energy - density * kinetic_energy)
    density_y, momentum_x_y, momentum_y_y, energy_y = q[5:8]
    velocity_x_y = (momentum_x_y - density_y * velocity_x) * inverse_density
    velocity_y_y = (momentum_y_y - density_y * velocity_y) * inverse_density
    kinetic_y = velocity_x * velocity_x_y + velocity_y * velocity_y_y
    pressure_y = gam1 * (energy_y - density_y * kinetic_energy - density * kinetic_y)
    temperature_y = (pressure_y * density - pressure * density_y) * inverse_density^2 / gam1
    symmetry = 0 * u
    symmetry[1] = density_y + tau[1] * (u[1] - uhat[1])
    symmetry[2] = velocity_x_y + tau[1] * (u[2] - uhat[2])
    symmetry[3] = velocity_y_y - tau[1] * uhat[3]
    symmetry[4] = temperature_y + tau[1] * (u[4] - uhat[4])
    normal_momentum = u[2] * n[1] + u[3] * n[2]
    slip_state = copy(u)
    slip_state[2] -= n[1] * normal_momentum
    slip_state[3] -= n[2] * normal_momentum
    slip = tau[1] * (slip_state - uhat)
    gradient = q[1:4] * n[1] + q[5:8] * n[2] + tau[1] * (u - uhat)
    auxiliary_wall = 0 * u
    auxiliary_wall[1] = u[1] - uhat[1]
    auxiliary_wall[2] = -uhat[2]
    auxiliary_wall[3] = -uhat[3]
    [inflow outflow isothermal symmetry slip gradient auxiliary_wall]
end

initu(x, mu, eta) = mu[5:8]
initv(x, mu, eta) = [0.0, 0.0]

function visscalars(u, q, w, v, x, t, mu, eta)
    gam, reynolds, mach_inf, reference_temperature = mu[1], mu[2], mu[4], mu[10]
    gas_constant, standard_temperature, sutherland_temperature = 287.0, 273.15, 110.4
    standard_viscosity = 1.716e-5
    density = u[1]
    velocity_x, velocity_y = u[2] / density, u[3] / density
    pressure = (gam - 1.0) * (u[4] - 0.5 * (u[2] * velocity_x + u[3] * velocity_y))
    temperature = pressure / ((gam - 1.0) * density)
    tinf = 1.0 / (gam * (gam - 1.0) * mach_inf^2)
    physical_temperature = reference_temperature * temperature / tinf
    velocity_reference = mach_inf * sqrt(gam * gas_constant * reference_temperature)
    ratio = reference_temperature / standard_temperature
    physical_viscosity = standard_viscosity * sqrt(ratio^3) *
        (standard_temperature + sutherland_temperature) /
        (reference_temperature + sutherland_temperature)
    density_reference = reynolds * physical_viscosity / velocity_reference
    physical_density = density_reference * density
    physical_pressure = physical_density * gas_constant * physical_temperature
    mach = velocity_reference * sqrt(velocity_x^2 + velocity_y^2) /
        sqrt(gam * gas_constant * physical_temperature)
    artificial_viscosity = (mu[end-1] + mu[end] * v[2]) * tanh(mu[end-2] * v[1])
    [physical_density, physical_pressure, physical_temperature, mach, artificial_viscosity]
end

function surfacequantities(u, q, w, v, x, t, mu, eta, uhat, normal, tau)
    gam, reynolds, prandtl = mu[1:3]
    gam1 = gam - 1.0
    reference_temperature = mu[10]
    tinf = mu[9]

    density_inf = mu[5]
    momentum_x_inf = mu[6]
    momentum_y_inf = mu[7]
    energy_inf = mu[8]
    velocity_x_inf = momentum_x_inf / density_inf
    velocity_y_inf = momentum_y_inf / density_inf
    speed_inf2 = velocity_x_inf^2 + velocity_y_inf^2
    dynamic_pressure = 0.5 * density_inf * speed_inf2
    heat_flux_reference = density_inf * speed_inf2 * sqrt(speed_inf2)
    pressure_inf = gam1 * (
        energy_inf - 0.5 * (momentum_x_inf^2 + momentum_y_inf^2) / density_inf
    )

    density, momentum_x, momentum_y, _, _, _,
    velocity_x, velocity_y, _, _, pressure,
    velocity_x_x, velocity_y_x, velocity_x_y, velocity_y_y,
    temperature_x, temperature_y = flow_quantities(uhat, q, mu)

    temperature = pressure / (gam1 * density)
    physical_temperature = reference_temperature * temperature / tinf
    dynamic_viscosity = viscosity_value(1.0 / reynolds, reference_temperature, physical_temperature)
    conductivity = dynamic_viscosity * gam / prandtl
    radial_coordinate = x[2]
    stress_xx = dynamic_viscosity * (2.0 / 3.0) *
        (2.0 * velocity_x_x - velocity_y_y + velocity_y / radial_coordinate)
    stress_xy = dynamic_viscosity * (velocity_x_y + velocity_y_x)
    stress_yy = dynamic_viscosity * (2.0 / 3.0) *
        (2.0 * velocity_y_y - velocity_x_x + velocity_y / radial_coordinate)

    normal_x, normal_y = normal
    tangent_x, tangent_y = -normal_y, normal_x
    traction_x = stress_xx * normal_x + stress_xy * normal_y + tau[1] * (u[2] - uhat[2])
    traction_y = stress_xy * normal_x + stress_yy * normal_y + tau[1] * (u[3] - uhat[3])
    tangential_traction = tangent_x * traction_x + tangent_y * traction_y
    conductive_wall_flux = conductivity * (temperature_x * normal_x + temperature_y * normal_y) +
                           tau[1] * (u[4] - uhat[4])

    [
        (pressure - pressure_inf) / dynamic_pressure,
        tangential_traction / dynamic_pressure,
        conductive_wall_flux / heat_flux_reference,
    ]
end

function visvectors(u, q, w, v, x, t, mu, eta)
    velocity_reference = mu[4] * sqrt(mu[1] * 287.0 * mu[10])
    velocity_reference .* u[2:3] ./ u[1]
end
