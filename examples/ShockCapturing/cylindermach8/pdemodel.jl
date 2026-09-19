"""Julia translation of the MATLAB Mach-8 cylinder PDE model."""

lmax_value(value, alpha) = value * (atan(alpha * value) / pi + 0.5) - atan(alpha) / pi + 0.5
lmin_value(value, alpha) = value - lmax_value(value, alpha)

function limiting_value(value, lower, upper, alpha, beta)
    limited = lower + lmax_value(value - beta, alpha)
    return lmin_value(limited - upper, alpha) + upper
end

function viscosity_value(mu_ref, tref, temperature)
    sutherland_temperature = 110.4
    return mu_ref * (temperature / tref)^1.5 *
           (tref + sutherland_temperature) / (temperature + sutherland_temperature)
end

mass(u, q, w, v, x, t, mu, eta) = [1.0, 1.0, 1.0, 1.0]

function flux(u, q, w, v, x, t, mu, eta)
    gam = mu[1]
    gam1 = gam - 1.0
    reynolds = mu[2]
    prandtl = mu[3]
    mach_inf = mu[4]
    tref = mu[10]
    mu_ref = 1.0 / reynolds
    tinf = 1.0 / (gam * gam1 * mach_inf^2)
    c23 = 2.0 / 3.0

    alpha = 1.0e3
    rmin = 1.0e-2
    pmin = 1.0e-3
    av = (mu[end-1] + mu[end] * v[2]) * tanh(mu[end-2] * v[1])

    r, ru, rv, rE = u
    rx, rux, rvx, rEx, ry, ruy, rvy, rEy = q

    r = rmin + lmax_value(r - rmin, alpha)
    dr = atan(alpha * (r - rmin)) / pi +
         alpha * (r - rmin) / (pi * (alpha^2 * (r - rmin)^2 + 1.0)) + 0.5
    rx = rx * dr
    ry = ry * dr
    r1 = 1.0 / r
    ux_velocity = ru * r1
    uy_velocity = rv * r1
    total_energy = rE * r1
    kinetic_energy = 0.5 * (ux_velocity^2 + uy_velocity^2)
    pressure = gam1 * (rE - r * kinetic_energy)
    pressure = pmin + lmax_value(pressure - pmin, alpha)
    dp = atan(alpha * (pressure - pmin)) / pi +
         alpha * (pressure - pmin) /
         (pi * (alpha^2 * (pressure - pmin)^2 + 1.0)) + 0.5
    enthalpy = total_energy + pressure * r1
    inviscid = [
        ru,
        ru * ux_velocity + pressure,
        rv * ux_velocity,
        ru * enthalpy,
        rv,
        ru * uy_velocity,
        rv * uy_velocity + pressure,
        rv * enthalpy,
    ]

    du_dx = (rux - rx * ux_velocity) * r1
    dv_dx = (rvx - rx * uy_velocity) * r1
    dke_dx = ux_velocity * du_dx + uy_velocity * dv_dx
    dp_dx = gam1 * (rEx - rx * kinetic_energy - r * dke_dx) * dp
    dt_dx = (dp_dx * r - pressure * rx) * r1^2 / gam1

    du_dy = (ruy - ry * ux_velocity) * r1
    dv_dy = (rvy - ry * uy_velocity) * r1
    dke_dy = ux_velocity * du_dy + uy_velocity * dv_dy
    dp_dy = gam1 * (rEy - ry * kinetic_energy - r * dke_dy) * dp
    dt_dy = (dp_dy * r - pressure * ry) * r1^2 / gam1

    temperature = pressure / (gam1 * r)
    physical_temperature = tref * temperature / tinf
    dynamic_viscosity = viscosity_value(mu_ref, tref, physical_temperature)
    conductivity = dynamic_viscosity * gam / prandtl
    tau_xx = dynamic_viscosity * c23 * (2.0 * du_dx - dv_dy)
    tau_xy = dynamic_viscosity * (du_dy + dv_dx)
    tau_yy = dynamic_viscosity * c23 * (2.0 * dv_dy - du_dx)
    viscous = [
        0.0,
        tau_xx,
        tau_xy,
        ux_velocity * tau_xx + uy_velocity * tau_xy + conductivity * dt_dx,
        0.0,
        tau_xy,
        tau_yy,
        ux_velocity * tau_xy + uy_velocity * tau_yy + conductivity * dt_dy,
    ]
    artificial = [
        av * rx,
        av * rux,
        av * rvx,
        av * rEx,
        av * ry,
        av * ruy,
        av * rvy,
        av * rEy,
    ]
    return reshape(inviscid + viscous + artificial, 4, 2)
end

function avfield(u, q, w, v, x, t, mu, eta)
    r, ru, rv = u[1:3]
    rx, rux, ry, rvy = q[1], q[2], q[5], q[7]
    velocity_x = ru / r
    velocity_y = rv / r
    du_dx = (rux - rx * velocity_x) / r
    dv_dy = (rvy - ry * velocity_y) / r
    divergence = du_dx + dv_dy
    return [limiting_value(divergence * tanh(mu[end-2] * v[1]), 0.0, mu[end-3], 1.0e3, 0.0)]
end

function visscalars(u, q, w, v, x, t, mu, eta)
    gam = mu[1]
    density = u[1]
    velocity_x = u[2] / density
    velocity_y = u[3] / density
    signed_pressure = (gam - 1.0) *
                      (u[4] - 0.5 * (u[2] * velocity_x + u[3] * velocity_y))
    pressure = sqrt(signed_pressure * signed_pressure)
    mach = sqrt(velocity_x^2 + velocity_y^2) / sqrt(gam * pressure / density)
    artificial_viscosity = mu[end] * v[2] * tanh(mu[end-2] * v[1])
    return [mach, artificial_viscosity]
end

source(u, q, w, v, x, t, mu, eta) = fill(0.0 * u[1], 4)

function fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    f = flux(uhat, q, w, v, x, t, mu, eta)
    freestream = f[:, 1] * n[1] + f[:, 2] * n[2] + tau[1] * (u - uhat)
    adiabatic_wall = copy(freestream)
    adiabatic_wall[1] = 0.0
    adiabatic_wall[end] = 0.0
    thermal_wall = copy(freestream)
    thermal_wall[1] = 0.0
    return [freestream adiabatic_wall thermal_wall adiabatic_wall freestream freestream]
end

function ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    uinf = mu[5:8]
    t_iso_wall = mu[11] / mu[10] * mu[9]
    isothermal = copy(u)
    isothermal[2:3] .= 0.0
    isothermal[4] = u[1] * t_iso_wall
    slip = copy(u)
    normal_momentum = u[2] * n[1] + u[3] * n[2]
    slip[2] = u[2] - n[1] * normal_momentum
    slip[3] = u[3] - n[2] * normal_momentum
    return [uinf uinf isothermal slip uinf u]
end

function fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    t_iso_wall = mu[11] / mu[10] * mu[9]
    uinf = mu[5:8]
    outflow = u - uhat
    inflow = uinf - uhat
    wall = 0 * u
    wall[1] = u[1] - uhat[1]
    wall[2] = -uhat[2]
    wall[3] = -uhat[3]
    wall[4] = -uhat[4] + uhat[1] * t_iso_wall
    return [inflow inflow wall wall inflow outflow]
end

initu(x, mu, eta) = mu[5:8]
initv(x, mu, eta) = [0.0, 0.0]
