"""Steady compressible Navier-Stokes manufactured solution on the unit square.

physicsparam = [gamma, Re, Pr, Mach], constant viscosity 1/Re, q = -grad(u).
The exact primitive state is rho = 1+0.1 sin(pi x)cos(pi y),
u = 0.5+0.1 cos(pi x)sin(pi y), v = 0.2+0.1 sin(pi x)sin(pi y) and
p = (1+0.1 cos(pi x)cos(pi y))/(gamma M^2). The steady source is the
divergence of the model flux at the exact state. All four sides use the
exact state as Dirichlet data; the initial state is uniform.
"""
from numpy import array, reshape
from sympy import cos, diff, pi, sin


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
    gam, Minf = mu[0], mu[3]
    r = 1.0 + 0.1*sin(pi*x[0])*cos(pi*x[1])
    uv = 0.5 + 0.1*cos(pi*x[0])*sin(pi*x[1])
    vv = 0.2 + 0.1*sin(pi*x[0])*sin(pi*x[1])
    p = (1.0 + 0.1*cos(pi*x[0])*cos(pi*x[1]))/(gam*Minf*Minf)
    return array([r, r*uv, r*vv, p/(gam - 1.0) + 0.5*r*(uv*uv + vv*vv)])


def source(u, q, w, v, x, t, mu, eta):
    # Steady manufactured source S = div F(u_e, -grad u_e), differentiated
    # symbolically with respect to the physical coordinates.
    ue = exact(x, mu)
    qe = array([-diff(ue[c], x[d]) for d in range(2) for c in range(4)])
    f = flux(ue, qe, w, v, x, t, mu, eta)
    return array([diff(f[c, 0], x[0]) + diff(f[c, 1], x[1]) for c in range(4)])


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return exact(x, mu)


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    ub = exact(x, mu)
    f = flux(ub, q, w, v, x, t, mu, eta)
    return f[:, 0]*n[0] + f[:, 1]*n[1] + tau[0]*(u - ub)


def fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return exact(x, mu) - uhat


def initu(x, mu, eta):
    gam, Minf = mu[0], mu[3]
    return array([1.0, 0.5, 0.2, 1/(gam*(gam - 1.0)*Minf*Minf) + 0.145])
