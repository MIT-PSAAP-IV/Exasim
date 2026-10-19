"""Three-dimensional Laplace model on the HCA flared-plate mesh."""

flux(u, q, w, v, x, t, mu, eta) = mu[1] .* q
source(u, q, w, v, x, t, mu, eta) = [0.0 * u[1]]
fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = [0.0 * u[1]]
ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = [0.0 * u[1]]
initu(x, mu, eta) = [0.0 * x[1]]

function fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    zero = -uhat[1]
    one = 1.0 - uhat[1]
    normal_flux = dot(flux(uhat, q, w, v, x, t, mu, eta), n)
    natural = normal_flux + tau[1] * (u[1] - uhat[1])
    return reshape([zero, one, natural], 1, 3)
end
