function flux(u, q, w, v, x, t, mu, eta)
    inviscid = [u[1], 0.0, 0.0, u[2]]
    viscous = [mu[1] * q[1], q[2], mu[1] * q[3], q[4]]
    reshape(inviscid + viscous, 2, 2)
end

source(u, q, w, v, x, t, mu, eta) = [1.0, u[1]]

function fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    f = flux(u, q, w, v, x, t, mu, eta)
    f[:, 1] * n[1] + f[:, 2] * n[2] + tau[1] * (u - uhat)
end

fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau) = -tau[1] .* uhat
ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = zeros(2)
initu(x, mu, eta) = zeros(2)
