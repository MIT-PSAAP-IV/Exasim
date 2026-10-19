flux(u, q, w, v, x, t, mu, eta) = mu[1] .* q
source(u, q, w, v, x, t, mu, eta) = -4.0 * (x[1] - 0.5)^2 + 1.0
fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = reshape([0.0, mu[1] * q[1]], 1, 2)
ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = reshape([0.0, u[1]], 1, 2)

function fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    exact = x[1]^4 / 3.0 - 2.0 * x[1]^3 / 3.0
    tau[1] * (exact - uhat[1])
end

initu(x, mu, eta) = 0.0 * x[1]^2
