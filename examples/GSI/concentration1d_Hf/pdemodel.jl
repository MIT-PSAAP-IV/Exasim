# This model is intentionally identical to ../concentration1d/pdemodel.jl.
mass(u, q, w, v, x, t, mu, eta) = 1.0
flux(u, q, w, v, x, t, mu, eta) = mu[1] .* q
source(u, q, w, v, x, t, mu, eta) = -mu[2] .* u
fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = 0.0
ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = 0.0
initu(x, mu, eta) = 1.0e-7

function fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    prescribed = tau[1] * (mu[3] - uhat[1])
    natural = mu[1] * q[1] + tau[1] * (u[1] - uhat[1])
    reshape([prescribed, natural], 1, :)
end
