function flux(u, q, w, v, x, t, mu, eta)
    gradient = reshape(q, 3, 3)
    identity = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    trace_gradient = gradient[1, 1] + gradient[2, 2] + gradient[3, 3]
    mu[1] * (gradient + transpose(gradient)) + mu[2] * trace_gradient * identity
end

source(u, q, w, v, x, t, mu, eta) = 0.0 .* x
fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = 0.0 .* x
ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = 0.0 .* x

function fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    stress = flux(u, q, w, v, x, t, mu, eta)
    numerical_traction = stress * n + tau[1] * (u - uhat)
    fixed_displacement = -tau[1] .* uhat
    applied_traction = numerical_traction - mu[3:5]
    hcat(numerical_traction, fixed_displacement, applied_traction)
end

initu(x, mu, eta) = 0.0 .* x
