using SymPy

function ale_mapping(x, t, mu)
    time_scale = sqrt(Sym(125))
    shape = sin(Sym(pi) * x[1] / 10) * sin(Sym(pi) * x[2] / 7.5)
    [x[1] + 2.0 * shape * sin(2 * Sym(pi) * t / time_scale),
     x[2] + 1.5 * shape * sin(4 * Sym(pi) * t / time_scale)]
end

function deformation_gradient(x, t, mu)
    time_scale = sqrt(Sym(125))
    cx, sx = cos(Sym(pi) * x[1] / 10), sin(Sym(pi) * x[1] / 10)
    cy, sy = cos(Sym(pi) * x[2] / 7.5), sin(Sym(pi) * x[2] / 7.5)
    s2, s4 = sin(2 * Sym(pi) * t / time_scale), sin(4 * Sym(pi) * t / time_scale)
    [1 + (Sym(pi) / 5) * cx * sy * s2 (2 * Sym(pi) / 7.5) * sx * cy * s2;
     (1.5 * Sym(pi) / 10) * cx * sy * s4 1 + (Sym(pi) / 5) * sx * cy * s4]
end

function mapping_jacobian(x, t, mu)
    gradient = deformation_gradient(x, t, mu)
    gradient[1, 1] * gradient[2, 2] - gradient[1, 2] * gradient[2, 1]
end

function mesh_velocity(x, t, mu)
    time_scale = sqrt(Sym(125))
    shape = sin(Sym(pi) * x[1] / 10) * sin(Sym(pi) * x[2] / 7.5)
    [4 * Sym(pi) * shape * cos(2 * Sym(pi) * t / time_scale) / time_scale,
     6 * Sym(pi) * shape * cos(4 * Sym(pi) * t / time_scale) / time_scale]
end

function jacobian_time_derivative(x, t, mu)
    time_scale = sqrt(Sym(125))
    cx, sx = cos(Sym(pi) * x[1] / 10), sin(Sym(pi) * x[1] / 10)
    cy, sy = cos(Sym(pi) * x[2] / 7.5), sin(Sym(pi) * x[2] / 7.5)
    s2, c2 = sin(2 * Sym(pi) * t / time_scale), cos(2 * Sym(pi) * t / time_scale)
    s4, c4 = sin(4 * Sym(pi) * t / time_scale), cos(4 * Sym(pi) * t / time_scale)
    g11 = 1 + (Sym(pi) / 5) * cx * sy * s2
    g12 = (2 * Sym(pi) / 7.5) * sx * cy * s2
    g21 = (1.5 * Sym(pi) / 10) * cx * sy * s4
    g22 = 1 + (Sym(pi) / 5) * sx * cy * s4
    w2, w4 = 2 * Sym(pi) / time_scale, 4 * Sym(pi) / time_scale
    dg11 = (Sym(pi) / 5) * cx * sy * w2 * c2
    dg12 = (2 * Sym(pi) / 7.5) * sx * cy * w2 * c2
    dg21 = (1.5 * Sym(pi) / 10) * cx * sy * w4 * c4
    dg22 = (Sym(pi) / 5) * sx * cy * w4 * c4
    dg11 * g22 + g11 * dg22 - dg12 * g21 - g12 * dg21
end

function mass(u, q, w, v, x, t, mu, eta)
    jacobian = mapping_jacobian(x, t, mu)
    [jacobian, jacobian, jacobian, jacobian]
end

function flux(u, q, w, v, x, t, mu, eta)
    gamma = mu[1]
    density, momentum_x, momentum_y, total_energy = u
    velocity_x, velocity_y = momentum_x / density, momentum_y / density
    pressure = (gamma - 1) * (total_energy - 0.5 * density * (velocity_x^2 + velocity_y^2))
    enthalpy = (total_energy + pressure) / density
    physical = reshape([momentum_x, momentum_x * velocity_x + pressure,
        momentum_y * velocity_x, momentum_x * enthalpy, momentum_y,
        momentum_x * velocity_y, momentum_y * velocity_y + pressure,
        momentum_y * enthalpy], 4, 2)
    gradient = deformation_gradient(x, t, mu)
    cofactor = [gradient[2, 2] -gradient[2, 1]; -gradient[1, 2] gradient[1, 1]]
    (physical - u * transpose(mesh_velocity(x, t, mu))) * cofactor
end

source(u, q, w, v, x, t, mu, eta) = -u .* jacobian_time_derivative(x, t, mu)

function fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    flux(u, q, w, v, x, t, mu, eta) * n + tau[1] .* (u - uhat)
end

ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = zeros(4)

function fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    value = tau[1] * (0.0 - uhat[1])
    [value, value, value, value]
end

function vortex(x, t, mu)
    physical = ale_mapping(x, t, mu)
    gamma, mach, strength, radius, center_x0, center_y0 = mu
    gamma_minus_one = gamma - 1
    direction_x, direction_y = 2 / sqrt(Sym(5)), 1 / sqrt(Sym(5))
    mean_x, mean_y = mach * direction_x, mach * direction_y
    dx = physical[1] - (center_x0 + mean_x * t)
    dy = physical[2] - (center_y0 + mean_y * t)
    profile = 1.0 - (dx^2 + dy^2) / radius^2
    temperature = 1.0 - gamma_minus_one * strength^2 * exp(profile) / (8 * Sym(pi)^2)
    density = temperature^(1 / gamma_minus_one)
    velocity_x = mean_x - strength * dy * exp(profile / 2) / (2 * Sym(pi) * radius)
    velocity_y = mean_y + strength * dx * exp(profile / 2) / (2 * Sym(pi) * radius)
    pressure = temperature^(gamma / gamma_minus_one) / gamma
    energy = pressure / gamma_minus_one + 0.5 * density * (velocity_x^2 + velocity_y^2)
    [density, density * velocity_x, density * velocity_y, energy]
end

initu(x, mu, eta) = vortex(x, Sym(0), mu)

function qoivolume(u, q, w, v, x, t, mu, eta)
    error = u - vortex(x, t, mu)
    mapping_jacobian(x, t, mu) * sum(error .* error)
end

function visscalars(u, q, w, v, x, t, mu, eta)
    density, momentum_x, momentum_y, total_energy = u
    pressure = (mu[1] - 1) * (total_energy - 0.5 * (momentum_x^2 + momentum_y^2) / density)
    [density, pressure, density - vortex(x, t, mu)[1]]
end

visvectors(u, q, w, v, x, t, mu, eta) = ale_mapping(x, t, mu) - x
