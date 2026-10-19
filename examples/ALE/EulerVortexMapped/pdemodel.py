import numpy as np
from sympy import cos, exp, pi, sin, sqrt


def ale_mapping(x, t, mu):
    time_scale = sqrt(125)
    shape = sin(pi * x[0] / 10) * sin(pi * x[1] / 7.5)
    return np.array([
        x[0] + 2.0 * shape * sin(2 * pi * t / time_scale),
        x[1] + 1.5 * shape * sin(4 * pi * t / time_scale),
    ])


def deformation_gradient(x, t, mu):
    time_scale = sqrt(125)
    cos_x = cos(pi * x[0] / 10)
    sin_x = sin(pi * x[0] / 10)
    cos_y = cos(pi * x[1] / 7.5)
    sin_y = sin(pi * x[1] / 7.5)
    s2 = sin(2 * pi * t / time_scale)
    s4 = sin(4 * pi * t / time_scale)
    return np.array([
        [1 + (pi / 5) * cos_x * sin_y * s2, (2 * pi / 7.5) * sin_x * cos_y * s2],
        [(1.5 * pi / 10) * cos_x * sin_y * s4, 1 + (pi / 5) * sin_x * cos_y * s4],
    ])


def mapping_jacobian(x, t, mu):
    gradient = deformation_gradient(x, t, mu)
    return gradient[0, 0] * gradient[1, 1] - gradient[0, 1] * gradient[1, 0]


def mesh_velocity(x, t, mu):
    time_scale = sqrt(125)
    shape = sin(pi * x[0] / 10) * sin(pi * x[1] / 7.5)
    return np.array([
        4 * pi * shape * cos(2 * pi * t / time_scale) / time_scale,
        6 * pi * shape * cos(4 * pi * t / time_scale) / time_scale,
    ])


def jacobian_time_derivative(x, t, mu):
    time_scale = sqrt(125)
    cos_x = cos(pi * x[0] / 10)
    sin_x = sin(pi * x[0] / 10)
    cos_y = cos(pi * x[1] / 7.5)
    sin_y = sin(pi * x[1] / 7.5)
    s2 = sin(2 * pi * t / time_scale)
    c2 = cos(2 * pi * t / time_scale)
    s4 = sin(4 * pi * t / time_scale)
    c4 = cos(4 * pi * t / time_scale)
    g11 = 1 + (pi / 5) * cos_x * sin_y * s2
    g12 = (2 * pi / 7.5) * sin_x * cos_y * s2
    g21 = (1.5 * pi / 10) * cos_x * sin_y * s4
    g22 = 1 + (pi / 5) * sin_x * cos_y * s4
    w2 = 2 * pi / time_scale
    w4 = 4 * pi / time_scale
    dg11 = (pi / 5) * cos_x * sin_y * w2 * c2
    dg12 = (2 * pi / 7.5) * sin_x * cos_y * w2 * c2
    dg21 = (1.5 * pi / 10) * cos_x * sin_y * w4 * c4
    dg22 = (pi / 5) * sin_x * cos_y * w4 * c4
    return dg11 * g22 + g11 * dg22 - dg12 * g21 - g12 * dg21


def mass(u, q, w, v, x, t, mu, eta):
    jacobian = mapping_jacobian(x, t, mu)
    return np.array([jacobian, jacobian, jacobian, jacobian])


def flux(u, q, w, v, x, t, mu, eta):
    gamma = mu[0]
    density, momentum_x, momentum_y, total_energy = u
    velocity_x = momentum_x / density
    velocity_y = momentum_y / density
    pressure = (gamma - 1) * (total_energy - 0.5 * density * (velocity_x**2 + velocity_y**2))
    enthalpy = (total_energy + pressure) / density
    physical = np.reshape(np.array([
        momentum_x, momentum_x * velocity_x + pressure, momentum_y * velocity_x,
        momentum_x * enthalpy, momentum_y, momentum_x * velocity_y,
        momentum_y * velocity_y + pressure, momentum_y * enthalpy,
    ]), (4, 2), order="F")
    gradient = deformation_gradient(x, t, mu)
    cofactor = np.array([[gradient[1, 1], -gradient[1, 0]], [-gradient[0, 1], gradient[0, 0]]])
    velocity = mesh_velocity(x, t, mu)
    return (physical - np.outer(u, velocity)) @ cofactor


def source(u, q, w, v, x, t, mu, eta):
    return -u * jacobian_time_derivative(x, t, mu)


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    physical_flux = flux(u, q, w, v, x, t, mu, eta)
    return physical_flux @ n + tau[0] * (u - uhat)


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return np.zeros(4)


def fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    value = tau[0] * (0.0 - uhat[0])
    return np.array([value, value, value, value])


def vortex(x, t, mu):
    physical = ale_mapping(x, t, mu)
    gamma, mach, strength, radius, center_x0, center_y0 = mu
    gamma_minus_one = gamma - 1
    direction_x = 2 / sqrt(5)
    direction_y = 1 / sqrt(5)
    mean_x = mach * direction_x
    mean_y = mach * direction_y
    dx = physical[0] - (center_x0 + mean_x * t)
    dy = physical[1] - (center_y0 + mean_y * t)
    profile = 1.0 - (dx**2 + dy**2) / radius**2
    temperature = 1.0 - gamma_minus_one * strength**2 * exp(profile) / (8 * pi**2)
    density = temperature ** (1 / gamma_minus_one)
    velocity_x = mean_x - strength * dy * exp(profile / 2) / (2 * pi * radius)
    velocity_y = mean_y + strength * dx * exp(profile / 2) / (2 * pi * radius)
    pressure = temperature ** (gamma / gamma_minus_one) / gamma
    energy = pressure / gamma_minus_one + 0.5 * density * (velocity_x**2 + velocity_y**2)
    return np.array([density, density * velocity_x, density * velocity_y, energy])


def initu(x, mu, eta):
    return vortex(x, 0, mu)


def qoivolume(u, q, w, v, x, t, mu, eta):
    error = u - vortex(x, t, mu)
    return mapping_jacobian(x, t, mu) * np.dot(error, error)


def visscalars(u, q, w, v, x, t, mu, eta):
    density, momentum_x, momentum_y, total_energy = u
    pressure = (mu[0] - 1) * (
        total_energy - 0.5 * (momentum_x**2 + momentum_y**2) / density
    )
    return np.array([density, pressure, density - vortex(x, t, mu)[0]])


def visvectors(u, q, w, v, x, t, mu, eta):
    return ale_mapping(x, t, mu) - x
