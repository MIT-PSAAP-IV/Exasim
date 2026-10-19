"""Transport stage for the density-equalizing mesh map."""

mass(u, q, w, v, x, t, mu, eta) = [1.0]
initu(x, mu, eta) = [x[1]]
ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = [0.0]

function flux(u, q, w, v, x, t, mu, eta)
    density, velocity_x, velocity_y = v[1], v[4], v[5]
    denominator = t + (1.0 - t) * density
    diffusion = mu[1] * (mu[2] * density - 1.0)
    return reshape([velocity_x * u[1] / denominator + diffusion * q[1],
                    velocity_y * u[1] / denominator + diffusion * q[2]], 1, 2)
end

function source(u, q, w, v, x, t, mu, eta)
    density, density_x, density_y = v[1:3]
    velocity_x, velocity_y = v[4:5]
    denominator = t + (1.0 - t) * density
    divergence_term = (velocity_x * density_x + velocity_y * density_y) / denominator
    return [u[1] * (density - 1.0 - (1.0 - t) * divergence_term) / denominator]
end

function fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    return [dot(flux(u, q, w, v, x, t, mu, eta)[1, :], n) + tau[1] * (u[1] - uhat[1])]
end

fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau) =
    reshape([u[1] - uhat[1], x[1] - uhat[1], x[2] - uhat[1]], 1, 3)
