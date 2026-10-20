"""Compressible Navier--Stokes model for the periodic Taylor--Green vortex."""

mass(u, q, w, v, x, t, mu, eta) = ones(5)

function flux(u, q, w, v, x, t, mu, eta)
    gam, reynolds, prandtl, mach = mu[1:4]
    gam1 = gam - 1.0
    density, mx, my, mz, energy = u
    ux, uy, uz = mx / density, my / density, mz / density
    kinetic = 0.5 * (ux^2 + uy^2 + uz^2)
    pressure = gam1 * (energy - density * kinetic)
    enthalpy = (energy + pressure) / density
    inviscid = [
        mx, mx*ux+pressure, my*ux, mz*ux, mx*enthalpy,
        my, mx*uy, my*uy+pressure, mz*uy, my*enthalpy,
        mz, mx*uz, my*uz, mz*uz+pressure, mz*enthalpy,
    ]
    gradients = reshape(q, 5, 3)
    velocity_gradient = Vector{Any}(undef, 3)
    temperature_gradient = Vector{Any}(undef, 3)
    for direction in 1:3
        dr, dmx, dmy, dmz, de = gradients[:, direction]
        du = (dmx-dr*ux)/density
        dv = (dmy-dr*uy)/density
        dw = (dmz-dr*uz)/density
        dk = ux*du + uy*dv + uz*dw
        dp = gam1*(de-dr*kinetic-density*dk)
        velocity_gradient[direction] = (du, dv, dw)
        temperature_gradient[direction] = gam*mach^2*(dp*density-pressure*dr)/density^2
    end
    dux, dvx, dwx = velocity_gradient[1]
    duy, dvy, dwy = velocity_gradient[2]
    duz, dvz, dwz = velocity_gradient[3]
    inverse_reynolds, two_thirds = 1.0/reynolds, 2.0/3.0
    txx = inverse_reynolds*two_thirds*(2*dux-dvy-dwz)
    txy = inverse_reynolds*(duy+dvx)
    txz = inverse_reynolds*(duz+dwx)
    tyy = inverse_reynolds*two_thirds*(2*dvy-dux-dwz)
    tyz = inverse_reynolds*(dvz+dwy)
    tzz = inverse_reynolds*two_thirds*(2*dwz-dux-dvy)
    conductivity = 1.0/(gam1*mach^2*reynolds*prandtl)
    tx, ty, tz = temperature_gradient
    viscous = [
        0.0, txx, txy, txz, ux*txx+uy*txy+uz*txz+conductivity*tx,
        0.0, txy, tyy, tyz, ux*txy+uy*tyy+uz*tyz+conductivity*ty,
        0.0, txz, tyz, tzz, ux*txz+uy*tyz+uz*tzz+conductivity*tz,
    ]
    reshape(inviscid + viscous, 5, 3)
end

source(u, q, w, v, x, t, mu, eta) = zeros(5)

function normal_flux(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    flux(u, q, w, v, x, t, mu, eta) * n + tau[1] * (u-uhat)
end

fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = normal_flux(u, q, w, v, x, t, mu, eta, uhat, n, tau)
fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau) = normal_flux(u, q, w, v, x, t, mu, eta, uhat, n, tau)
ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau) = zeros(5)

function initu(x, mu, eta)
    gam, mach = mu[1], mu[4]
    x1, x2, x3 = x
    ux = sin(x1)*cos(x2)*cos(x3)
    uy = -cos(x1)*sin(x2)*cos(x3)
    uz = 0.0
    pressure = 1.0/(gam*mach^2) + (cos(2*x1)+cos(2*x2))*(cos(2*x3)+2.0)/16.0
    [1.0, ux, uy, uz, 0.5*(ux^2+uy^2)+pressure/(gam-1.0)]
end
