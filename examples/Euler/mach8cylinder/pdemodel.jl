"""Julia translation of the legacy Mach-8.98 cylinder model."""

lmax_value(x, a) = x * (atan(a * x) / pi + 0.5) - atan(a) / pi + 0.5
lmin_value(x, a) = x - lmax_value(x, a)
function limiting_value(x, lo, hi, a, beta)
    x = lo + lmax_value(x - beta, a)
    lmin_value(x - hi, a) + hi
end
viscosity_value(mu0, tref, temperature) = mu0 * (temperature / tref)^1.5 * (tref + 110.4) / (temperature + 110.4)

mass(u, q, w, v, x, t, mu, eta) = ones(4)

function flux(u, q, w, v, x, t, mu, eta)
    gamma, reynolds, prandtl, mach = mu[1:4]
    gamma1 = gamma - 1.0
    alpha, rmin, pmin = 1.0e3, 1.0e-2, 1.0e-3
    density, momentum_x, momentum_y, total_energy = u
    rx, rux, rvx, rEx, ry, ruy, rvy, rEy = q
    density = rmin + lmax_value(density - rmin, alpha)
    dr = atan(alpha * (density - rmin)) / pi + alpha * (density - rmin) / (pi * (alpha^2 * (density - rmin)^2 + 1.0)) + 0.5
    rx, ry = rx * dr, ry * dr
    velocity_x, velocity_y = momentum_x / density, momentum_y / density
    energy = total_energy / density
    kinetic = 0.5 * (velocity_x^2 + velocity_y^2)
    pressure = pmin + lmax_value(gamma1 * (total_energy - density * kinetic) - pmin, alpha)
    dp = atan(alpha * (pressure - pmin)) / pi + alpha * (pressure - pmin) / (pi * (alpha^2 * (pressure - pmin)^2 + 1.0)) + 0.5
    enthalpy = energy + pressure / density
    inviscid = [momentum_x, momentum_x * velocity_x + pressure, momentum_y * velocity_x,
        momentum_x * enthalpy, momentum_y, momentum_x * velocity_y,
        momentum_y * velocity_y + pressure, momentum_y * enthalpy]
    ux, vx = (rux - rx * velocity_x) / density, (rvx - rx * velocity_y) / density
    uy, vy = (ruy - ry * velocity_x) / density, (rvy - ry * velocity_y) / density
    px = gamma1 * (rEx - rx * kinetic - density * (velocity_x * ux + velocity_y * vx)) * dp
    py = gamma1 * (rEy - ry * kinetic - density * (velocity_x * uy + velocity_y * vy)) * dp
    tx = (px * density - pressure * rx) / (gamma1 * density^2)
    ty = (py * density - pressure * ry) / (gamma1 * density^2)
    tinf = 1.0 / (gamma * gamma1 * mach^2)
    temperature = pressure / (gamma1 * density)
    dynamic = viscosity_value(1.0 / reynolds, mu[10], mu[10] * temperature / tinf) + mu[14] * v[1]
    conductivity = dynamic * gamma / prandtl
    bulk = v[1]
    tau_xx = dynamic * (2.0 / 3.0) * (2.0 * ux - vy) + bulk * (ux + vy)
    tau_xy = dynamic * (uy + vx)
    tau_yy = dynamic * (2.0 / 3.0) * (2.0 * vy - ux) + bulk * (ux + vy)
    viscous = [0.0, tau_xx, tau_xy,
        velocity_x * tau_xx + velocity_y * tau_xy + (conductivity + mu[13] * v[1]) * tx,
        0.0, tau_xy, tau_yy,
        velocity_x * tau_xy + velocity_y * tau_yy + (conductivity + mu[13] * v[1]) * ty]
    reshape(inviscid + viscous, 4, 2)
end

function avfield(u, q, w, v, x, t, mu, eta)
    gamma, porder = mu[1], mu[15]
    density, momentum_x, momentum_y, total_energy = u
    rx, rux, rvx, _, ry, ruy, rvy, _ = q
    alpha, rmin, hmin = 1.0e3, 1.0e-2, 1.0e-4
    density = rmin + lmax_value(density - rmin, alpha)
    dr = atan(alpha * (density - rmin)) / pi + alpha * (density - rmin) / (pi * (alpha^2 * (density - rmin)^2 + 1.0)) + 0.5
    rx, ry = rx * dr, ry * dr
    velocity_x, velocity_y = momentum_x / density, momentum_y / density
    ux, vx = (rux - rx * velocity_x) / density, (rvx - rx * velocity_y) / density
    uy, vy = (ruy - ry * velocity_x) / density, (rvy - ry * velocity_y) / density
    kinetic = 0.5 * (velocity_x^2 + velocity_y^2)
    enthalpy = hmin + lmax_value(gamma * total_energy / density - (gamma - 1.0) * kinetic - hmin, alpha)
    sound = sqrt(2.0 * (gamma - 1.0) * enthalpy / (gamma + 1.0))
    divergence = limiting_value(-(ux + vy), -1.0e4, 1.0e4, alpha, -1.0e4)
    vorticity = limiting_value(sqrt((vx - uy)^2), 0.0, 1.0e4, alpha, 0.0)
    ratio = divergence^2 / (divergence^2 + vorticity^2 + 1.0e-16)
    sensor = limiting_value(-(v[2] / porder) * divergence / sound * ratio, 0.0,
        mu[17] / sqrt(gamma^2 - 1.0), alpha, mu[16])
    [density * mu[12] * v[2] / porder * sqrt(velocity_x^2 + velocity_y^2 + sound^2) * sensor]
end

source(u, q, w, v, x, t, mu, eta) = zeros(4)
function fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    value = flux(uhat, q, w, v, x, t, mu, eta) * n + tau[1] .* (u - uhat)
    adiabatic = copy(value); adiabatic[[1, 4]] .= 0.0
    thermal = copy(value); thermal[1] = 0.0
    [value adiabatic thermal adiabatic value value]
end
function ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    gamma = mu[1]
    gamma1 = gamma - 1.0
    density, momentum_x, momentum_y, total_energy = u
    nx, ny = n
    velocity_x, velocity_y = momentum_x / density, momentum_y / density
    energy = total_energy / density
    pressure = gamma1 * (total_energy - 0.5 * density * (velocity_x^2 + velocity_y^2))
    enthalpy = energy + pressure / density
    sound = sqrt(gamma * pressure / density)
    normal_velocity = velocity_x * nx + velocity_y * ny
    tangent_velocity = -velocity_x * ny + velocity_y * nx
    right = [
        1.0 1.0 0.0 1.0;
        normal_velocity-sound normal_velocity 0.0 normal_velocity+sound;
        tangent_velocity tangent_velocity 1.0 tangent_velocity;
        enthalpy-normal_velocity*sound 0.5*(normal_velocity^2+tangent_velocity^2) tangent_velocity enthalpy+normal_velocity*sound
    ]
    right_inverse = gamma1 / (2.0 * sound^2) .* [
        enthalpy+sound*(normal_velocity-sound)/gamma1 -(normal_velocity+sound/gamma1) -tangent_velocity 1.0;
        -2.0*enthalpy+4.0*sound^2/gamma1 2.0*normal_velocity 2.0*tangent_velocity -2.0;
        -2.0*tangent_velocity*sound^2/gamma1 0.0 2.0*sound^2/gamma1 0.0;
        enthalpy-sound*(normal_velocity+sound)/gamma1 -normal_velocity+sound/gamma1 -tangent_velocity 1.0
    ]
    rotate = [1.0 0.0 0.0 0.0; 0.0 nx ny 0.0; 0.0 -ny nx 0.0; 0.0 0.0 0.0 1.0]
    rotate_inverse = [1.0 0.0 0.0 0.0; 0.0 nx -ny 0.0; 0.0 ny nx 0.0; 0.0 0.0 0.0 1.0]
    characteristic_sign = [
        tanh(1.0e3 * (normal_velocity - sound)) 0.0 0.0 0.0;
        0.0 tanh(1.0e3 * normal_velocity) 0.0 0.0;
        0.0 0.0 tanh(1.0e3 * normal_velocity) 0.0;
        0.0 0.0 0.0 tanh(1.0e3 * (normal_velocity + sound))
    ]
    normal_sign = rotate_inverse * right * characteristic_sign * right_inverse * rotate
    uinf = mu[5:8]
    farfield = 0.5 .* ((u + uinf) + normal_sign * (u - uinf))
    adiabatic = copy(u); adiabatic[2:3] .= 0.0
    thermal = copy(adiabatic); thermal[4] = u[1] * mu[11] / mu[10] * mu[9]
    slip = copy(u); normal_momentum = u[2] * n[1] + u[3] * n[2]
    slip[2:3] .-= n .* normal_momentum
    [farfield adiabatic thermal slip uinf u]
end
initu(x, mu, eta) = mu[5:8]
