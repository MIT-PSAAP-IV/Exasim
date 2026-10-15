# Model 2 of the mixed triangle/quadrilateral Poisson example (pdemodel2.m).
module pdemodel2

function flux(u, q, w, v, x, t, mu, eta)
    return mu[1] * q
end

function source(u, q, w, v, x, t, mu, eta)
    return (2 * pi * pi) * sin(pi * x[1]) * sin(pi * x[2])
end

function fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    f = flux(u, q, w, v, x, t, mu, eta)
    return f[1] * n[1] + f[2] * n[2] + tau[1] * u[1]
end

function ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    return 0 * x[1]   # symbolic zero, like sym(0.0) in pdemodel2.m
end

function initu(x, mu, eta)
    return 0 * x[1] * x[2]
end

function fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    return [-uhat[1] uhat[1]]
end

function fint(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    f = flux(u, q, w, v, x, t, mu, eta)
    return f[1] * n[1] + f[2] * n[2] + tau[1] * (u[1] - uhat[1])
end

end
