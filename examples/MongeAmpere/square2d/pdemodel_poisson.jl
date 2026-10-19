"""Poisson stage for the analytic square monitor function."""

flux(u, q, w, v, x, t, mu, eta) = mu[1] .* q
initu(x, mu, eta) = [0.0]
ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = [0.0]

function source(u, q, w, v, x, t, mu, eta)
    radius = sqrt(x[1]^2 + x[2]^2)
    density = 1.0 + mu[3] / cosh(mu[4] * (radius^2 - mu[5]^2))
    return [density / mu[2] - 1.0]
end

function fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    return [dot(flux(u, q, w, v, x, t, mu, eta), n) + tau[1] * u[1]]
end

function fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    return reshape([dot(flux(u, q, w, v, x, t, mu, eta), n) +
                    tau[1] * (u[1] - uhat[1])], 1, 1)
end
