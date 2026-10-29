"""Compressible planar Couette flow with viscous heating.

physicsparam = [gamma, Re, Pr, Mach, wall speed U, T(y=0), T(y=1)], with
constant viscosity 1/Re and q = -grad(u). The exact state has u = U*y,
v = 0, p = 1/(gamma*M^2), rho = gamma*M^2*p/T and
T = T0 + (T1-T0)*y + (gamma-1)*M^2*Pr*U^2*y*(1-y)/2. All four sides use
this exact state as Dirichlet data; the initial state is uniform.
"""
from numpy import array, reshape


def mass(u, q, w, v, x, t, mu, eta):
    return array([1.0, 1.0, 1.0, 1.0])


def flux(u, q, w, v, x, t, mu, eta):
    gam, Re, Pr, Minf = mu[0], mu[1], mu[2], mu[3]
    gam1 = gam - 1.0
    Re1 = 1/Re
    M2 = Minf*Minf
    c23 = 2.0/3.0
    fc = 1/(gam1*M2*Re*Pr)
    r, ru, rv, rE = u[0], u[1], u[2], u[3]
    rx, rux, rvx, rEx, ry, ruy, rvy, rEy = (q[k] for k in range(8))
    r1 = 1/r
    uv = ru*r1
    vv = rv*r1
    ke = 0.5*(uv*uv + vv*vv)
    p = gam1*(rE - r*ke)
    h = rE*r1 + p*r1
    ux = (rux - rx*uv)*r1
    vx = (rvx - rx*vv)*r1
    px = gam1*(rEx - rx*ke - r*(uv*ux + vv*vx))
    Tx = gam*M2*(px*r - p*rx)*r1*r1
    uy = (ruy - ry*uv)*r1
    vy = (rvy - ry*vv)*r1
    py = gam1*(rEy - ry*ke - r*(uv*uy + vv*vy))
    Ty = gam*M2*(py*r - p*ry)*r1*r1
    txx = Re1*c23*(2*ux - vy)
    txy = Re1*(uy + vx)
    tyy = Re1*c23*(2*vy - ux)
    f = array([ru, ru*uv + p + txx, rv*uv + txy, ru*h + uv*txx + vv*txy + fc*Tx,
               rv, ru*vv + txy, rv*vv + p + tyy, rv*h + uv*txy + vv*tyy + fc*Ty])
    return reshape(f, (4, 2), "F")


def exact(x, mu):
    gam, Pr, Minf, U, T0, T1 = mu[0], mu[2], mu[3], mu[4], mu[5], mu[6]
    y = x[1]
    M2 = Minf*Minf
    T = T0 + (T1 - T0)*y + 0.5*(gam - 1.0)*M2*Pr*U*U*y*(1.0 - y)
    p = 1/(gam*M2)
    r = gam*M2*p/T
    return array([r, r*U*y, 0.0*y, p/(gam - 1.0) + 0.5*r*U*U*y*y])


def source(u, q, w, v, x, t, mu, eta):
    return array([0.0, 0.0, 0.0, 0.0])


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return exact(x, mu)


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    ub = exact(x, mu)
    f = flux(ub, q, w, v, x, t, mu, eta)
    return f[:, 0]*n[0] + f[:, 1]*n[1] + tau[0]*(u - ub)


def fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return exact(x, mu) - uhat


def initu(x, mu, eta):
    gam, Minf, U = mu[0], mu[3], mu[4]
    return array([1.0, 0.5*U, 0.0, 1/(gam*(gam - 1.0)*Minf*Minf) + 0.125*U*U])
