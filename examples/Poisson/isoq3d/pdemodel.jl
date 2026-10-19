"""Three-dimensional Laplace model used by the ISOQ example."""

flux(u, q, w, v, x, t, mu, eta) = mu[1] .* q
source(u, q, w, v, x, t, mu, eta) = [0.0 * u[1]]
fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = [0.0 * u[1]]
ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = [0.0 * u[1]]
initu(x, mu, eta) = [0.0 * x[1]]

function fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    dirichlet_zero = tau[1] * (mu[2] - uhat[1])
    dirichlet_one = tau[1] * (mu[3] - uhat[1])
    normal_flux = dot(flux(u, q, w, v, x, t, mu, eta), n)
    natural = normal_flux + tau[1] * (u[1] - uhat[1])
    return reshape([dirichlet_zero, dirichlet_one, natural], 1, 3)
end
