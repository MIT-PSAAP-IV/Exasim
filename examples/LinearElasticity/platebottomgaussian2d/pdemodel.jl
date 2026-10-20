function flux(u, q, w, v, x, t, mu, eta)
    gradient = reshape(q, 2, 2)
    identity = [1.0 0.0; 0.0 1.0]
    mu[1] * (gradient + transpose(gradient)) +
        mu[2] * (gradient[1, 1] + gradient[2, 2]) * identity
end

source(u, q, w, v, x, t, mu, eta) = 0.0 .* x
fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = 0.0 .* x
ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = 0.0 .* x
initu(x, mu, eta) = 0.0 .* x

function fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    prescribed = 0.0 .* uhat
    prescribed[2] = mu[3] * exp(-((x[1] - mu[4])^2) / (2.0 * mu[5]^2))
    fixed = -tau[1] .* uhat
    bottom = tau[1] .* (prescribed - uhat)
    hcat(fixed, bottom)
end
