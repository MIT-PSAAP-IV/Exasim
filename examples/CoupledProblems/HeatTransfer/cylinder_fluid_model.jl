module CylinderFluidModel

lmax_value(value, alpha) = value * (atan(alpha * value) / pi + 0.5) - atan(alpha) / pi + 0.5
viscosity_value(mu_ref, tref, temperature) = mu_ref * (temperature / tref)^1.5 * (tref + 110.4) / (temperature + 110.4)
mass(u, q, w, v, x, t, mu, eta) = ones(4)

function flux(u, q, w, v, x, t, mu, eta)
    gam, reynolds, prandtl, mach_inf = mu[1:4]
    gam1, tref = gam - 1.0, mu[10]
    tinf = 1.0 / (gam * gam1 * mach_inf^2)
    alpha, rmin, pmin = 1.0e3, 1.0e-2, 1.0e-3
    r, ru, rv, rE = u
    rx, rux, rvx, rEx, ry, ruy, rvy, rEy = q
    r = rmin + lmax_value(r - rmin, alpha)
    dr = atan(alpha * (r-rmin))/pi + alpha*(r-rmin)/(pi*(alpha^2*(r-rmin)^2+1.0)) + 0.5
    rx, ry = rx * dr, ry * dr
    r1, ux, uy = 1.0/r, ru/r, rv/r
    kinetic = 0.5 * (ux^2 + uy^2)
    pressure = pmin + lmax_value(gam1 * (rE-r*kinetic) - pmin, alpha)
    dp = atan(alpha*(pressure-pmin))/pi + alpha*(pressure-pmin)/(pi*(alpha^2*(pressure-pmin)^2+1.0)) + 0.5
    enthalpy = (rE + pressure) * r1
    inviscid = [ru, ru*ux+pressure, rv*ux, ru*enthalpy,
                rv, ru*uy, rv*uy+pressure, rv*enthalpy]
    du_dx = (rux-rx*ux)*r1; dv_dx = (rvx-rx*uy)*r1
    dp_dx = gam1*(rEx-rx*kinetic-r*(ux*du_dx+uy*dv_dx))*dp
    dt_dx = (dp_dx*r-pressure*rx)*r1^2/gam1
    du_dy = (ruy-ry*ux)*r1; dv_dy = (rvy-ry*uy)*r1
    dp_dy = gam1*(rEy-ry*kinetic-r*(ux*du_dy+uy*dv_dy))*dp
    dt_dy = (dp_dy*r-pressure*ry)*r1^2/gam1
    temperature = pressure/(gam1*r)
    dynamic_viscosity = viscosity_value(1.0/reynolds, tref, tref*temperature/tinf)
    conductivity = dynamic_viscosity*gam/prandtl
    tau_xx = dynamic_viscosity*(2.0/3.0)*(2.0*du_dx-dv_dy)
    tau_xy = dynamic_viscosity*(du_dy+dv_dx)
    tau_yy = dynamic_viscosity*(2.0/3.0)*(2.0*dv_dy-du_dx)
    viscous = [0.0,tau_xx,tau_xy,ux*tau_xx+uy*tau_xy+conductivity*dt_dx,
               0.0,tau_xy,tau_yy,ux*tau_xy+uy*tau_yy+conductivity*dt_dy]
    # artificial viscosity c*tanh(k*d): wall distance d in v[1], c = mu[12], k = mu[13]
    av = mu[12]*tanh(mu[13]*v[1])
    artificial = av .* [rx,rux,rvx,rEx,ry,ruy,rvy,rEy]
    return reshape(inviscid + viscous + artificial, 4, 2)
end

source(u,q,w,v,x,t,mu,eta) = fill(0.0 * u[1], 4)
fbou(u,q,w,v,x,t,mu,eta,uhat,n,tau) = fill(0.0 * u[1], 4)
ubou(u,q,w,v,x,t,mu,eta,uhat,n,tau) = fill(0.0 * u[1], 4)

function fbouhdg(u,q,w,v,x,t,mu,eta,uhat,n,tau)
    t_wall = mu[11]/mu[10]*mu[9]
    inflow, outflow = mu[5:8]-uhat, u-uhat
    isothermal = 0 .* u
    isothermal[1] = u[1]-uhat[1]; isothermal[2] = -uhat[2]
    isothermal[3] = -uhat[3]; isothermal[4] = -uhat[4]+uhat[1]*t_wall
    f = flux(uhat,q,w,v,x,t,mu,eta)
    adiabatic = copy(isothermal)
    adiabatic[4] = f[4,1]*n[1]+f[4,2]*n[2]+tau[1]*(u[4]-uhat[4])
    temperature = copy(isothermal); temperature[4] = uhat[4]/uhat[1]
    return [inflow outflow isothermal adiabatic temperature]
end

function fint(u,q,w,v,x,t,mu,eta,uhat,n,tau)
    f = flux(uhat,q,w,v,x,t,mu,eta)
    return f[4,1]*n[1]+f[4,2]*n[2]+tau[1]*(u[4]-uhat[4])
end

initu(x,mu,eta) = mu[5:8]
initv(x,mu,eta) = [0.0, 0.0]

end
