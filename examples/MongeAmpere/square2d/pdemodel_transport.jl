"""Transport stage for the analytic square monitor function."""

function monitor_values(x, mu)
    radius = sqrt(x[1]^2 + x[2]^2)
    argument = mu[4] * (radius^2 - mu[5]^2)
    density = 1.0 + mu[3] / cosh(argument)
    scale = -2.0 * mu[3] * mu[4] * sinh(argument) / cosh(argument)^2
    return density, scale * x[1], scale * x[2]
end

mass(u, q, w, v, x, t, mu, eta) = [1.0]
initu(x, mu, eta) = [x[1]]
ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = [0.0]

function flux(u, q, w, v, x, t, mu, eta)
    density, _, _ = monitor_values(x, mu)
    monitor = density / mu[2]
    denominator = t + (1.0 - t) * monitor
    return reshape([v[1] * u[1] / denominator + mu[1] * (density - 1.0) * q[1],
                    v[2] * u[1] / denominator + mu[1] * (density - 1.0) * q[2]], 1, 2)
end

function source(u, q, w, v, x, t, mu, eta)
    density, density_x, density_y = monitor_values(x, mu)
    monitor = density / mu[2]
    denominator = t + (1.0 - t) * monitor
    derivative = (v[1] * density_x + v[2] * density_y) / mu[2]
    return [u[1] * (monitor - 1.0 - (1.0 - t) * derivative / denominator) / denominator]
end

function fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    return [dot(flux(u, q, w, v, x, t, mu, eta)[1, :], n) + tau[1] * (u[1] - uhat[1])]
end

fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau) = reshape([u[1] - uhat[1]], 1, 1)
