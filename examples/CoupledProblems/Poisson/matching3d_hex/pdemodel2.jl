# Model 2 of the 3D coupled Poisson examples (pdemodel2.m).
module pdemodel2

function flux(u, q, w, v, x, t, mu, eta)
    return mu[1] * q
end

function source(u, q, w, v, x, t, mu, eta)
    return (2 * pi * pi - 1) * exp(x[1]) * sin(pi * x[2]) * sin(pi * x[3])
end

function fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    f = flux(u, q, w, v, x, t, mu, eta)
    return f[1] * n[1] + f[2] * n[2] + f[3] * n[3] + tau[1] * (u[1] - uhat[1])
end

function ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    return exp(x[1]) * sin(pi * x[2]) * sin(pi * x[3])
end

function initu(x, mu, eta)
    return 0 * x[1] * x[2] * x[3]
end

function fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    # Column 1: Dirichlet boundary (condition 1); column 2: the coupled interface.
    ub = exp(x[1]) * sin(pi * x[2]) * sin(pi * x[3])
    fi = uhat[1]
    return [ub - uhat[1] fi]
end

function fint(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    # Model 2 returns its normal flux to model 1.
    f = flux(u, q, w, v, x, t, mu, eta)
    return f[1] * n[1] + f[2] * n[2] + f[3] * n[3] + tau[1] * (u[1] - uhat[1])
end

end
