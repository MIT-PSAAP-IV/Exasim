"""Sharp-B equilibrium-air model with guarded material-table coordinates."""

include(joinpath(dirname(@__DIR__), "isoq2d_equichem", "pdemodel.jl"))

function materialstate(u, q, w, v, x, t, mu, eta)
    rho = u[1]
    velocity_z = u[2] / rho
    velocity_r = u[3] / rho
    internal_energy = u[4] / rho - 0.5 * (velocity_z^2 + velocity_r^2)

    rho_min = 0.0001 / mu[1]
    rho_max = 20.0 / mu[1]
    limited_rho = limiting_value(rho, rho_min, rho_max, 1.0e2, rho_min)

    energy_safety = 1000.0
    energy_min = -150000.0 + energy_safety
    energy_max = 20000000.0 - energy_safety
    limited_energy = limiting_value(
        mu[4] * internal_energy,
        energy_min,
        energy_max,
        1.0e2,
        energy_min,
    )
    return [log(mu[1] * limited_rho), limited_energy]
end
