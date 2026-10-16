module SolidHeatModel

mass(u, q, w, v, x, t, mu, eta) = 1.0
flux(u, q, w, v, x, t, mu, eta) = reshape(mu[1] .* q, 1, 2)
source(u, q, w, v, x, t, mu, eta) = 0.0 * u[1]
fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = 0.0 * u[1]
ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = 0.0 * u[1]

function fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    f = flux(u, q, w, v, x, t, mu, eta)
    numerical_flux = f[1, 1] * n[1] + f[1, 2] * n[2] + tau[1] * (u[1] - uhat[1])
    fixed_temperature = tau[1] * (mu[2] - uhat[1])
    return reshape([fixed_temperature, numerical_flux, numerical_flux], 1, 3)
end

fint(u, q, w, v, x, t, mu, eta, uhat, n, tau) = -uhat[1]
initu(x, mu, eta) = [mu[2]]

end
