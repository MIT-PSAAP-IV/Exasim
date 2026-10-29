"""Steady laminar NACA0012 flow, transcribed from the donor MATLAB model.

physicsparam = [gamma, Re, Pr, Mach, rho_inf, (rho u)_inf, (rho v)_inf,
(rho E)_inf]; q = -grad(u). Boundary 1 is the no-slip adiabatic wall and
boundary 2 the characteristic far field, exactly as in
examples/NavierStokes/naca2d/pdemodel.m.
"""
from numpy import array, reshape
from sympy import Matrix, sqrt, tanh


def mass(u, q, w, v, x, t, mu, eta):
    return array([1.0, 1.0, 1.0, 1.0])


def flux(u, q, w, v, x, t, mu, eta):
    gam, Re, Pr, Minf = mu[0], mu[1], mu[2], mu[3]
    gam1 = gam - 1.0
    Re1 = 1/Re
    M2 = Minf**2
    c23 = 2.0/3.0
    fc = 1/(gam1*M2*Re*Pr)
    r, ru, rv, rE = u[0], u[1], u[2], u[3]
    rx, rux, rvx, rEx, ry, ruy, rvy, rEy = (q[k] for k in range(8))
    r1 = 1/r
    uv = ru*r1
    vv = rv*r1
    E = rE*r1
    ke = 0.5*(uv*uv + vv*vv)
    p = gam1*(rE - r*ke)
    h = E + p*r1
    fi = array([ru, ru*uv + p, rv*uv, ru*h, rv, ru*vv, rv*vv + p, rv*h])
    ux = (rux - rx*uv)*r1
    vx = (rvx - rx*vv)*r1
    kex = uv*ux + vv*vx
    px = gam1*(rEx - rx*ke - r*kex)
    Tx = gam*M2*(px*r - p*rx)*r1**2
    uy = (ruy - ry*uv)*r1
    vy = (rvy - ry*vv)*r1
    key = uv*uy + vv*vy
    py = gam1*(rEy - ry*ke - r*key)
    Ty = gam*M2*(py*r - p*ry)*r1**2
    txx = Re1*c23*(2*ux - vy)
    txy = Re1*(uy + vx)
    tyy = Re1*c23*(2*vy - ux)
    fv = array([0, txx, txy, uv*txx + vv*txy + fc*Tx, 0, txy, tyy, uv*txy + vv*tyy + fc*Ty])
    return reshape(fi + fv, (4, 2), "F")


def source(u, q, w, v, x, t, mu, eta):
    return array([0.0, 0.0, 0.0, 0.0])


def _normal_matrix(state, n, mu, slope, analytic):
    """Characteristic matrix An of the donor model at state and normal n."""
    gam = mu[0]
    gam1 = gam - 1.0
    r, ru, rv, rE = state[0], state[1], state[2], state[3]
    nx, ny = n[0], n[1]
    r1 = 1/r
    uv = ru*r1
    vv = rv*r1
    E = rE*r1
    p = gam1*(rE - r*0.5*(uv*uv + vv*vv))
    h = E + p*r1
    a = sqrt(gam*p*r1)
    un = (ru*nx + rv*ny)/r
    ut = (-ru*ny + rv*nx)/r
    K = Matrix([[1, 1, 0, 1], [un - a, un, 0, un + a], [ut, ut, 1, ut],
                [h - un*a, 0.5*(un**2 + ut**2), ut, h + un*a]])
    Kinv = (gam1/(2*a**2))*Matrix([[h + (a/gam1)*(un - a), -(un + a/gam1), -ut, 1],
                                   [-2*h + (4/gam1)*a**2, 2*un, 2*ut, -2],
                                   [-2*(ut*a**2)/gam1, 0, 2*(a**2)/gam1, 0],
                                   [h - a*(un + a)/gam1, -un + a/gam1, -ut, 1]])
    T = Matrix([[1, 0, 0, 0], [0, nx, ny, 0], [0, -ny, nx, 0], [0, 0, 0, 1]])
    Tinv = Matrix([[1, 0, 0, 0], [0, nx, -ny, 0], [0, ny, nx, 0], [0, 0, 0, 1]])
    Lambda = Matrix.diag(tanh(slope*(un - a)), tanh(slope*un), tanh(slope*un), tanh(slope*(un + a)))
    if analytic:
        return Tinv*(K*Lambda*Kinv)*T
    return (Tinv*K)*Lambda*(Kinv*T)


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    uinf = Matrix([mu[4], mu[5], mu[6], mu[7]])
    U = Matrix([u[0], u[1], u[2], u[3]])
    An = _normal_matrix(u, n, mu, 1e3, True)
    ui = 0.5*((U + uinf) + An*(U - uinf))
    uw = [u[0], 0.0, 0.0, u[3]]
    return reshape(array(uw + list(ui)), (4, 2), "F")


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    f = flux(u, q, w, v, x, t, mu, eta)
    fi = f[:, 0]*n[0] + f[:, 1]*n[1] + tau[0]*(u - uhat)
    fw = fi.copy()
    fw[0] = 0.0
    fw[3] = 0.0
    return reshape(array(list(fw) + list(fi)), (4, 2), "F")


def fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    uinf = Matrix([mu[4], mu[5], mu[6], mu[7]])
    U = Matrix([u[0], u[1], u[2], u[3]])
    Uh = Matrix([uhat[0], uhat[1], uhat[2], uhat[3]])
    An = _normal_matrix(uhat, n, mu, 1e2, False)
    f2 = 0.5*((U + uinf) + An*(U - uinf)) - Uh
    f = flux(uhat, q, w, v, x, t, mu, eta)
    f1 = [u[0] - uhat[0], 0.0 - uhat[1], 0.0 - uhat[2], f[3, 0]*n[0] + f[3, 1]*n[1] + tau[0]*(u[3] - uhat[3])]
    return reshape(array(f1 + list(f2)), (4, 2), "F")


def initu(x, mu, eta):
    return array([mu[4], mu[5], mu[6], mu[7]])
