mass(u, q, w, v, x, t, mu, eta) = ones(5)

function flux(u, q, w, v, x, t, mu, eta)
    gam, reynolds, prandtl, mach_inf = mu[1:4]
    gam1 = gam - 1.0
    inverse_reynolds = 1.0 / reynolds
    mach2 = mach_inf^2
    artificial_viscosity = v[1]
    density, momentum_x, momentum_y, momentum_z, energy_density = u
    density_x, momentum_x_x, momentum_y_x, momentum_z_x, energy_x,
    density_y, momentum_x_y, momentum_y_y, momentum_z_y, energy_y,
    density_z, momentum_x_z, momentum_y_z, momentum_z_z, energy_z = q

    inverse_density = 1.0 / density
    velocity_x = momentum_x * inverse_density
    velocity_y = momentum_y * inverse_density
    velocity_z = momentum_z * inverse_density
    total_energy = energy_density * inverse_density
    kinetic_energy = 0.5 * (velocity_x^2 + velocity_y^2 + velocity_z^2)
    pressure = gam1 * (energy_density - density * kinetic_energy)
    temperature = gam * mach2 * pressure * inverse_density
    viscosity = 1.4042 * temperature * sqrt(temperature) / (temperature + 0.40417)
    conductivity = viscosity / (gam1 * mach2 * reynolds * prandtl)
    enthalpy = total_energy + pressure * inverse_density
    inviscid = [
        momentum_x, momentum_x*velocity_x+pressure, momentum_y*velocity_x, momentum_z*velocity_x, momentum_x*enthalpy,
        momentum_y, momentum_x*velocity_y, momentum_y*velocity_y+pressure, momentum_z*velocity_y, momentum_y*enthalpy,
        momentum_z, momentum_x*velocity_z, momentum_y*velocity_z, momentum_z*velocity_z+pressure, momentum_z*enthalpy,
    ]

    velocity_x_x = (momentum_x_x-density_x*velocity_x)*inverse_density
    velocity_y_x = (momentum_y_x-density_x*velocity_y)*inverse_density
    velocity_z_x = (momentum_z_x-density_x*velocity_z)*inverse_density
    kinetic_x = velocity_x*velocity_x_x + velocity_y*velocity_y_x + velocity_z*velocity_z_x
    pressure_x = gam1*(energy_x-density_x*kinetic_energy-density*kinetic_x)
    temperature_x = gam*mach2*(pressure_x*density-pressure*density_x)*inverse_density^2
    velocity_x_y = (momentum_x_y-density_y*velocity_x)*inverse_density
    velocity_y_y = (momentum_y_y-density_y*velocity_y)*inverse_density
    velocity_z_y = (momentum_z_y-density_y*velocity_z)*inverse_density
    kinetic_y = velocity_x*velocity_x_y + velocity_y*velocity_y_y + velocity_z*velocity_z_y
    pressure_y = gam1*(energy_y-density_y*kinetic_energy-density*kinetic_y)
    temperature_y = gam*mach2*(pressure_y*density-pressure*density_y)*inverse_density^2
    velocity_x_z = (momentum_x_z-density_z*velocity_x)*inverse_density
    velocity_y_z = (momentum_y_z-density_z*velocity_y)*inverse_density
    velocity_z_z = (momentum_z_z-density_z*velocity_z)*inverse_density
    kinetic_z = velocity_x*velocity_x_z + velocity_y*velocity_y_z + velocity_z*velocity_z_z
    pressure_z = gam1*(energy_z-density_z*kinetic_energy-density*kinetic_z)
    temperature_z = gam*mach2*(pressure_z*density-pressure*density_z)*inverse_density^2

    two_thirds = 2.0/3.0
    stress_xx = viscosity*inverse_reynolds*two_thirds*(2*velocity_x_x-velocity_y_y-velocity_z_z)
    stress_xy = viscosity*inverse_reynolds*(velocity_x_y+velocity_y_x)
    stress_xz = viscosity*inverse_reynolds*(velocity_x_z+velocity_z_x)
    stress_yy = viscosity*inverse_reynolds*two_thirds*(2*velocity_y_y-velocity_x_x-velocity_z_z)
    stress_yz = viscosity*inverse_reynolds*(velocity_y_z+velocity_z_y)
    stress_zz = viscosity*inverse_reynolds*two_thirds*(2*velocity_z_z-velocity_x_x-velocity_y_y)
    viscous = [
        0.0, stress_xx, stress_xy, stress_xz,
        velocity_x*stress_xx+velocity_y*stress_xy+velocity_z*stress_xz+conductivity*temperature_x,
        0.0, stress_xy, stress_yy, stress_yz,
        velocity_x*stress_xy+velocity_y*stress_yy+velocity_z*stress_yz+conductivity*temperature_y,
        0.0, stress_xz, stress_yz, stress_zz,
        velocity_x*stress_xz+velocity_y*stress_yz+velocity_z*stress_zz+conductivity*temperature_z,
    ]
    reshape(inviscid + viscous + artificial_viscosity .* q, 5, 3)
end

source(u, q, w, v, x, t, mu, eta) = zeros(5)

function normal_flux(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    f = flux(u, q, w, v, x, t, mu, eta)
    f[:,1]*n[1] + f[:,2]*n[2] + f[:,3]*n[3] + tau[1]*(u-uhat)
end

fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = normal_flux(u, q, w, v, x, t, mu, eta, uhat, n, tau)
fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau) = normal_flux(u, q, w, v, x, t, mu, eta, uhat, n, tau)
ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = zeros(5)

function initu(x, mu, eta)
    length_scale = 1.0
    x1, x2, x3 = x
    gam, mach_inf = mu[1], mu[4]
    mach2 = mach_inf^2
    pressure0 = 1.0/(gam*mach2)
    velocity_x = sin(x1/length_scale)*cos(x2/length_scale)*cos(x3/length_scale)
    velocity_y = -cos(x1/length_scale)*sin(x2/length_scale)*cos(x3/length_scale)
    velocity_z = 0.0
    pressure = pressure0 + (1.0/16.0)*(cos(2*x1/length_scale)+cos(2*x2/length_scale))*(2.0+cos(2*x3/length_scale))
    density = gam*mach2*pressure
    [density, density*velocity_x, density*velocity_y, density*velocity_z,
     pressure/(gam-1.0) + 0.5*density*(velocity_x^2+velocity_y^2+velocity_z^2)]
end

initv(x, mu, eta) = [0.0]

function avfield(u, q, w, v, x, t, mu, eta)
    [getavfield3d(u, q, mu[6], mu[1], mu[7], mu[8])]
end

function qoivolume(u, q, w, v, x, t, mu, eta)
    gam, reynolds, mach_inf, reference_density = mu[1], mu[2], mu[4], mu[5]
    mach2 = mach_inf^2
    domain_volume = (2.0*pi)^3
    density, momentum_x, momentum_y, momentum_z, energy_density = u
    density_x, momentum_x_x, momentum_y_x, momentum_z_x = q[1:4]
    density_y, momentum_x_y, momentum_y_y, momentum_z_y = q[6:9]
    density_z, momentum_x_z, momentum_y_z, momentum_z_z = q[11:14]
    inverse_density = 1.0/density
    velocity_x = momentum_x*inverse_density
    velocity_y = momentum_y*inverse_density
    velocity_z = momentum_z*inverse_density
    kinetic_energy = 0.5*(velocity_x^2+velocity_y^2+velocity_z^2)
    pressure = (gam-1.0)*(energy_density-density*kinetic_energy)
    temperature = gam*mach2*pressure*inverse_density
    viscosity = 1.4042*temperature*sqrt(temperature)/(temperature+0.40417)
    velocity_x_x = (momentum_x_x-density_x*velocity_x)*inverse_density
    velocity_y_x = (momentum_y_x-density_x*velocity_y)*inverse_density
    velocity_z_x = (momentum_z_x-density_x*velocity_z)*inverse_density
    velocity_x_y = (momentum_x_y-density_y*velocity_x)*inverse_density
    velocity_y_y = (momentum_y_y-density_y*velocity_y)*inverse_density
    velocity_z_y = (momentum_z_y-density_y*velocity_z)*inverse_density
    velocity_x_z = (momentum_x_z-density_z*velocity_x)*inverse_density
    velocity_y_z = (momentum_y_z-density_z*velocity_y)*inverse_density
    velocity_z_z = (momentum_z_z-density_z*velocity_z)*inverse_density
    curl_squared = (velocity_z_y-velocity_y_z)^2 + (velocity_x_z-velocity_z_x)^2 + (velocity_y_x-velocity_x_y)^2
    divergence = velocity_x_x+velocity_y_y+velocity_z_z
    [0.5*density*(velocity_x^2+velocity_y^2+velocity_z^2)/(reference_density*domain_volume),
     viscosity*curl_squared/(reference_density*reynolds*domain_volume),
     (4.0/3.0)*viscosity*divergence^2/(reference_density*reynolds*domain_volume)]
end

function getavfield3d(u, q, mesh_size, gam, coefficient, polynomial_order)
    gam1 = gam-1.0
    alpha, density_minimum, enthalpy_minimum = 1.0e3, 1.0e-3, 1.0e-4
    divergence_maximum, sensor_maximum, sensor_threshold = 50.0, 4.0, 0.1
    density, momentum_x, momentum_y, momentum_z, energy_density = u
    density_x, momentum_x_x, momentum_y_x, momentum_z_x = q[1:4]
    density_y, momentum_x_y, momentum_y_y, momentum_z_y = q[6:9]
    density_z, momentum_x_z, momentum_y_z, momentum_z_z = q[11:14]
    density = density_minimum + lmax(density-density_minimum,alpha)
    inverse_density = 1.0/density
    velocity_x = momentum_x*inverse_density
    velocity_y = momentum_y*inverse_density
    velocity_z = momentum_z*inverse_density
    total_energy = energy_density*inverse_density
    kinetic_energy = 0.5*(velocity_x^2+velocity_y^2+velocity_z^2)
    enthalpy = enthalpy_minimum + lmax(gam*total_energy-gam1*kinetic_energy-enthalpy_minimum,alpha)
    acoustic_scale = sqrt(2.0*gam1*enthalpy/(gam+1.0))
    velocity_x_x = (momentum_x_x-density_x*velocity_x)*inverse_density
    velocity_y_x = (momentum_y_x-density_x*velocity_y)*inverse_density
    velocity_z_x = (momentum_z_x-density_x*velocity_z)*inverse_density
    velocity_x_y = (momentum_x_y-density_y*velocity_x)*inverse_density
    velocity_y_y = (momentum_y_y-density_y*velocity_y)*inverse_density
    velocity_z_y = (momentum_z_y-density_y*velocity_z)*inverse_density
    velocity_x_z = (momentum_x_z-density_z*velocity_x)*inverse_density
    velocity_y_z = (momentum_y_z-density_z*velocity_y)*inverse_density
    velocity_z_z = (momentum_z_z-density_z*velocity_z)*inverse_density
    divergence = limiting(velocity_x_x+velocity_y_y+velocity_z_z,0.0,divergence_maximum,alpha,5.0)
    vorticity_magnitude = sqrt((velocity_z_y-velocity_y_z)^2 +
        (velocity_x_z-velocity_z_x)^2 + (velocity_y_x-velocity_x_y)^2)
    vorticity = limiting(vorticity_magnitude,0.0,divergence_maximum,alpha,0.0)
    ducros = divergence^2/(divergence^2+vorticity^2+1.0e-16)
    sensor = sqrt(mesh_size/polynomial_order)*(divergence/acoustic_scale)*ducros
    coefficient*limiting(sensor,0.0,sensor_maximum,alpha,sensor_threshold)
end

lmax(value, alpha) = value*(atan(alpha*value)/pi+0.5)-atan(alpha)/pi+0.5
lmin(value, alpha) = value-lmax(value,alpha)
function limiting(value, lower, upper, alpha, offset)
    regularized = lower+lmax(value-offset,alpha)
    lmin(regularized-upper,alpha)+upper
end
