lmax(value, alpha) = value * (atan(alpha * value) / pi + 0.5) - atan(alpha) / pi + 0.5
mass(u, q, w, v, x, t, mu, eta) = [1.0, 1.0]

function flux(u, q, w, v, x, t, mu, eta)
    temperature = u[1] * mu[3]
    diffusivity = mu[10] * exp(-mu[11] / (mu[5] * temperature))
    concentration_diffusivity = mu[12] * diffusivity / (mu[1] * mu[1])
    reshape([mu[14] * q[1], concentration_diffusivity * q[2]], 2, 1)
end

function source(u, q, w, v, x, t, mu, eta)
    concentration = lmax(u[2], 1.0e3)
    temperature = u[1] * mu[3]
    chemical_potential = mu[8] + mu[4] * temperature * log(concentration)
    heat_rate = mu[12] * (mu[6] * mu[7] / mu[15] + mu[9] * mu[6] * chemical_potential / mu[15]) / mu[3]
    reaction_rate = mu[6] * mu[12]
    [heat_rate * concentration, -reaction_rate * concentration]
end

fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = zeros(2)
ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = zeros(2)
initu(x, mu, eta) = [mu[2] / mu[3], exp(-1.0e4 * x[1] * x[1])]

function fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    physical_flux = flux(u, q, w, v, x, t, mu, eta)
    left = [physical_flux[1] * n[1] + tau[1] * (u[1] - uhat[1]) + mu[16], tau[1] * (1.0 - uhat[2])]
    right = [tau[1] * (mu[2] / mu[3] - uhat[1]), physical_flux[2] * n[1] + tau[1] * (u[2] - uhat[2])]
    hcat(left, right)
end
