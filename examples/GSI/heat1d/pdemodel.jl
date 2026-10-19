mass(u, q, w, v, x, t, mu, eta) = [1.0]
flux(u, q, w, v, x, t, mu, eta) = [mu[14] * q[1]]

function source(u, q, w, v, x, t, mu, eta)
    concentration = (exp(mu[17] * x[1]) + exp(2.0 * mu[17] - mu[17] * x[1])) /
        (1.0 + exp(2.0 * mu[17]))
    temperature = u[1] * mu[3]
    chemical_potential = mu[8] + mu[4] * temperature * log(concentration)
    rate = mu[12] * (mu[6] * mu[7] / mu[15] + mu[9] * mu[6] * chemical_potential / mu[15]) / mu[3]
    [rate * concentration]
end

fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = 0.0
ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = 0.0
initu(x, mu, eta) = [mu[2] / mu[3]]

function fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    physical_flux = flux(u, q, w, v, x, t, mu, eta)[1]
    applied_flux = physical_flux * n[1] + tau[1] * (u[1] - uhat[1]) + mu[16]
    prescribed = tau[1] * (mu[2] / mu[3] - uhat[1])
    reshape([applied_flux, prescribed], 1, :)
end
