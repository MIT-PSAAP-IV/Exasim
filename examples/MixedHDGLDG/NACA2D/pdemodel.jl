# Steady laminar NACA0012 flow, transcribed from the donor MATLAB model
# examples/NavierStokes/naca2d/pdemodel.m. physicsparam = [gamma, Re, Pr,
# Mach, rho_inf, (rho u)_inf, (rho v)_inf, (rho E)_inf], q = -grad(u).
# Boundary 1: no-slip adiabatic wall; boundary 2: characteristic far field.
mass(u,q,w,v,x,t,mu,eta) = [1.0, 1.0, 1.0, 1.0]

function flux(u,q,w,v,x,t,mu,eta)
    gam = mu[1]; Re = mu[2]; Pr = mu[3]; Minf = mu[4]
    gam1 = gam - 1.0; Re1 = 1/Re; M2 = Minf^2; c23 = 2.0/3.0
    fc = 1/(gam1*M2*Re*Pr)
    r = u[1]; ru = u[2]; rv = u[3]; rE = u[4]
    rx = q[1]; rux = q[2]; rvx = q[3]; rEx = q[4]
    ry = q[5]; ruy = q[6]; rvy = q[7]; rEy = q[8]
    r1 = 1/r; uv = ru*r1; vv = rv*r1; E = rE*r1
    ke = 0.5*(uv*uv + vv*vv)
    p = gam1*(rE - r*ke)
    h = E + p*r1
    fi = [ru, ru*uv+p, rv*uv, ru*h, rv, ru*vv, rv*vv+p, rv*h]
    ux = (rux - rx*uv)*r1; vx = (rvx - rx*vv)*r1
    kex = uv*ux + vv*vx
    px = gam1*(rEx - rx*ke - r*kex)
    Tx = gam*M2*(px*r - p*rx)*r1^2
    uy = (ruy - ry*uv)*r1; vy = (rvy - ry*vv)*r1
    key = uv*uy + vv*vy
    py = gam1*(rEy - ry*ke - r*key)
    Ty = gam*M2*(py*r - p*ry)*r1^2
    txx = Re1*c23*(2*ux - vy); txy = Re1*(uy + vx); tyy = Re1*c23*(2*vy - ux)
    fv = [0, txx, txy, uv*txx + vv*txy + fc*Tx, 0, txy, tyy, uv*txy + vv*tyy + fc*Ty]
    return reshape(fi + fv, 4, 2)
end

# Symbolic zeros: the HDG generator differentiates the source.
source(u,q,w,v,x,t,mu,eta) = 0*u

# Characteristic matrix An of the donor model at state and normal n.
function normalmatrix(state, n, mu, slope, analytic)
    gam = mu[1]; gam1 = gam - 1.0
    r = state[1]; ru = state[2]; rv = state[3]; rE = state[4]
    nx = n[1]; ny = n[2]
    r1 = 1/r; uv = ru*r1; vv = rv*r1; E = rE*r1
    p = gam1*(rE - r*0.5*(uv*uv + vv*vv))
    h = E + p*r1
    a = sqrt(gam*p*r1)
    un = (ru*nx + rv*ny)/r
    ut = (-ru*ny + rv*nx)/r
    K = [1 1 0 1; un-a un 0 un+a; ut ut 1 ut; h-un*a 0.5*(un^2 + ut^2) ut h+un*a]
    Kinv = (gam1/(2*a^2))*[h+(a/gam1)*(un-a) -(un+a/gam1) -ut 1;
                           -2*h+(4/gam1)*a^2 2*un 2*ut -2;
                           -2*(ut*a^2)/gam1 0 2*(a^2)/gam1 0;
                           h-a*(un+a)/gam1 -un+a/gam1 -ut 1]
    T = [1 0 0 0; 0 nx ny 0; 0 -ny nx 0; 0 0 0 1]
    Tinv = [1 0 0 0; 0 nx -ny 0; 0 ny nx 0; 0 0 0 1]
    Lambda = [tanh(slope*(un-a)) 0 0 0; 0 tanh(slope*un) 0 0;
              0 0 tanh(slope*un) 0; 0 0 0 tanh(slope*(un+a))]
    return analytic ? Tinv*(K*Lambda*Kinv)*T : (Tinv*K)*Lambda*(Kinv*T)
end

function ubou(u,q,w,v,x,t,mu,eta,uhat,n,tau)
    uinf = mu[5:8]
    An = normalmatrix(u, n, mu, 1e3, true)
    ui = 0.5*((u + uinf) + An*(u - uinf))
    uw = [u[1], 0.0, 0.0, u[4]]
    return hcat(uw, ui)
end

function fbou(u,q,w,v,x,t,mu,eta,uhat,n,tau)
    f = flux(u,q,w,v,x,t,mu,eta)
    fi = f[:,1]*n[1] + f[:,2]*n[2] + tau[1]*(u - uhat)
    fw = copy(fi)
    fw[1] = 0.0
    fw[4] = 0.0
    return hcat(fw, fi)
end

function fbouhdg(u,q,w,v,x,t,mu,eta,uhat,n,tau)
    uinf = mu[5:8]
    An = normalmatrix(uhat, n, mu, 1e2, false)
    f2 = 0.5*((u + uinf) + An*(u - uinf)) - uhat
    f = flux(uhat,q,w,v,x,t,mu,eta)
    f1 = [u[1] - uhat[1], 0.0 - uhat[2], 0.0 - uhat[3], f[4,1]*n[1] + f[4,2]*n[2] + tau[1]*(u[4] - uhat[4])]
    return hcat(f1, f2)
end

initu(x,mu,eta) = mu[5:8]
