"""Poisson stage for the density-equalizing mesh map."""

flux(u, q, w, v, x, t, mu, eta) = mu[1] .* q
source(u, q, w, v, x, t, mu, eta) = [v[1] - 1.0]
initu(x, mu, eta) = [0.0]
ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = [0.0]

function fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    return [dot(flux(u, q, w, v, x, t, mu, eta), n) + tau[1] * u[1]]
end

function fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    return reshape([dot(flux(u, q, w, v, x, t, mu, eta), n) +
                    tau[1] * (u[1] - uhat[1])], 1, 1)
end
