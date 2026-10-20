mass(u, q, w, v, x, t, mu, eta) = ones(5)

function flux(u, q, w, v, x, t, mu, eta)
    ion_mass, kB = mu[1:2]
    density, momentum, pressure = u[1], u[2:4], u[5]
    kappa = 1.24e4 * (pressure / (kB * density))^2.5
    density_gradient = [q[1], q[6], q[11]]
    pressure_gradient = [q[5], q[10], q[15]]
    mass_flux = momentum / ion_mass
    identity3 = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    momentum_flux = momentum * transpose(momentum) / (ion_mass * density) + pressure * identity3
    pressure_flux = pressure * momentum / (ion_mass * density) -
        2 * kappa * (pressure * density_gradient - density * pressure_gradient) /
        (3 * kB * density^2)
    [transpose(mass_flux); momentum_flux; transpose(pressure_flux)]
end

function source(u, q, w, v, x, t, mu, eta)
    ion_mass, charge, gravity = mu[1], mu[3], mu[4]
    electric, magnetic, neutral_velocity = v[1:3], v[4:6], v[7:9]
    density, momentum, pressure = u[1], u[2:4], u[5]
    density_gradient = [q[1], q[6], q[11]]
    momentum_divergence = q[2] + q[8] + q[14]
    radius = sqrt(sum(x .* x))

    # Match the existing Python example's deliberately simplified source.
    collision_frequency = 0.0
    production = 0.0
    loss = 0.0
    heating = 0.0
    cooling = 0.0
    momentum_source = (production / density - loss) .* momentum +
        density * ion_mass * gravity .* x / radius + density * charge .* electric +
        charge .* cross(momentum, magnetic) / ion_mass +
        collision_frequency .* (density * ion_mass .* neutral_velocity - momentum)
    pressure_source = 2 * pressure *
        (density * momentum_divergence - dot(momentum, density_gradient)) /
        (3 * ion_mass * density) + pressure * (production / density - loss) +
        2 * (heating - cooling * pressure) / 3
    [production - loss * density; momentum_source; pressure_source]
end

function equilibrium_state(v, mu)
    ion_mass = mu[1]
    neutral_velocity = v[7:9]
    production, loss, heating, cooling = v[11:14]
    [production / loss;
     ion_mass * production .* neutral_velocity / loss;
     heating / cooling]
end

function ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    [equilibrium_state(v, mu) u]
end

function fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    f = flux(u, q, w, v, x, t, mu, eta)
    physical = f[:,1] * n[1] + f[:,2] * n[2] + f[:,3] * n[3] + tau[1] * (u - uhat)
    [physical zeros(5)]
end

function fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    f = flux(u, q, w, v, x, t, mu, eta)
    physical = f[:,1] * n[1] + f[:,2] * n[2] + f[:,3] * n[3] + tau[1] * (u - uhat)
    [tau[1] * (equilibrium_state(v, mu) - uhat) physical]
end

initu(x, mu, eta) = zeros(5)
