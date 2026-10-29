"""Independent Python definitions used only for measurement.

These formulas are transcribed from the PDE definitions, not imported from
any pdemodel file, so they can check the generated models. Exasim stores
q = -grad(u) in direction-major order [q_x(0..ncu-1), q_y(...), ...]; the
model flux is F(u,q), the steady equation is div F = S, and an outward
numerical flux Fhat.n is positive when it leaves the element.
"""

import numpy as np
import sympy as sp

# Common physical parameters of each example: identical to every pdeapp.txt.
PARAMETERS = {
    "Poisson": [1.0, 0.0], "Poisson3D": [1.0, 0.0],
    # gamma, Re, Pr, Mach, wall speed, lower/upper wall temperature
    "Couette": [1.4, 10.0, 0.72, 0.8, 1.0, 1.0, 1.1],
    # gamma, Re, Pr, Mach
    "ManufacturedNavierStokes": [1.4, 10.0, 0.72, 0.5],
    # donor NACA: gamma, Re, Pr, Mach, freestream conservative state
    "NACA2D": [1.4, 100.0, 0.72, 0.2, 1.0, 1.0, 0.0, 45.1429],
}
TAU = {"Poisson": 1.0, "Poisson3D": 1.0, "Couette": 3.0, "ManufacturedNavierStokes": 3.0, "NACA2D": 3.0}


def _symbols(nd):
    return sp.symbols("x0:%d" % nd, real=True)


def exact_symbolic(case):
    """Conservative exact state as SymPy expressions of physical x."""
    mu = PARAMETERS[case]
    if case == "Poisson":
        x, y = _symbols(2)
        return (x, y), [sp.sin(sp.pi*x)*sp.sin(sp.pi*y)]
    if case == "Poisson3D":
        x, y, z = _symbols(3)
        return (x, y, z), [sp.sin(sp.pi*x)*sp.sin(sp.pi*y)*sp.sin(sp.pi*z)]
    x, y = _symbols(2)
    gam, Re, Pr, M = (sp.nsimplify(v) for v in mu[:4])
    if case == "Couette":
        U, T0, T1 = (sp.nsimplify(v) for v in mu[4:7])
        u = U*y
        # T'' = -(gam-1) M^2 Pr U^2 from the viscous heating balance.
        T = T0 + (T1 - T0)*y + (gam - 1)*M**2*Pr*U**2*y*(1 - y)/2
        p = 1/(gam*M**2)
        r = gam*M**2*p/T
        v = sp.Integer(0)
    else:
        r = 1 + sp.Rational(1, 10)*sp.sin(sp.pi*x)*sp.cos(sp.pi*y)
        u = sp.Rational(1, 2) + sp.Rational(1, 10)*sp.cos(sp.pi*x)*sp.sin(sp.pi*y)
        v = sp.Rational(1, 5) + sp.Rational(1, 10)*sp.sin(sp.pi*x)*sp.sin(sp.pi*y)
        p = (1 + sp.Rational(1, 10)*sp.cos(sp.pi*x)*sp.cos(sp.pi*y))/(gam*M**2)
    return (x, y), [r, r*u, r*v, p/(gam - 1) + r*(u**2 + v**2)/2]


def ns_flux(u, q, mu):
    """Exasim compressible NS flux F[n,4,2] from (u,q=-grad u), arrays [n,..]."""
    gam, Re, Pr, M = mu[:4]
    gam1, Re1, M2 = gam - 1, 1/Re, M*M
    fc = 1/(gam1*M2*Re*Pr)
    r, ru, rv, rE = (u[:, k] for k in range(4))
    rx, rux, rvx, rEx, ry, ruy, rvy, rEy = (q[:, k] for k in range(8))
    r1 = 1/r
    uv, vv = ru*r1, rv*r1
    ke = (uv*uv + vv*vv)/2
    p = gam1*(rE - r*ke)
    h = rE*r1 + p*r1
    ux, vx = (rux - rx*uv)*r1, (rvx - rx*vv)*r1
    uy, vy = (ruy - ry*uv)*r1, (rvy - ry*vv)*r1
    px = gam1*(rEx - rx*ke - r*(uv*ux + vv*vx))
    py = gam1*(rEy - ry*ke - r*(uv*uy + vv*vy))
    Tx = gam*M2*(px*r - p*rx)*r1*r1
    Ty = gam*M2*(py*r - p*ry)*r1*r1
    txx, txy, tyy = Re1*2/3*(2*ux - vy), Re1*(uy + vx), Re1*2/3*(2*vy - ux)
    F = np.empty((len(r), 4, 2))
    F[:, :, 0] = np.column_stack([ru, ru*uv + p + txx, rv*uv + txy, ru*h + uv*txx + vv*txy + fc*Tx])
    F[:, :, 1] = np.column_stack([rv, ru*vv + txy, rv*vv + p + tyy, rv*h + uv*txy + vv*tyy + fc*Ty])
    return F


def flux(case, u, q):
    """Model flux F[n,ncu,nd] for state u[n,ncu] and q[n,nd*ncu]."""
    mu = PARAMETERS[case]
    if case.startswith("Poisson"):
        ncu = u.shape[1]
        return mu[0]*q.reshape(len(q), q.shape[1]//ncu, ncu).transpose(0, 2, 1)
    return ns_flux(u, q, mu)


class Case:
    """Lambdified exact state/gradient/source and boundary data for one case."""

    def __init__(self, case):
        self.name = case
        x, state = exact_symbolic(case)
        self.nd, self.ncu = len(x), len(state)
        gradient = [-sp.diff(s, x[d]) for d in range(self.nd) for s in state]
        self._state = sp.lambdify([x], state, "numpy")
        self._gradient = sp.lambdify([x], gradient, "numpy")
        # Independent steady source: S = div F(u_exact, q_exact).
        if case.startswith("Poisson"):
            kappa = PARAMETERS[case][0]
            source = [sum(sp.diff(-kappa*sp.diff(state[0], xd), xd) for xd in x)]
        else:
            source = self._ns_source(x, state)
        self._source = sp.lambdify([x], source, "numpy")
        self.source_expressions = source

    def _ns_source(self, x, state):
        gam, Re, Pr, M = (sp.nsimplify(v) for v in PARAMETERS[self.name][:4])
        q = [-sp.diff(s, xd) for xd in x for s in state]
        r, ru, rv, rE = state
        rx, rux, rvx, rEx, ry, ruy, rvy, rEy = q
        uv, vv = ru/r, rv/r
        ke = (uv**2 + vv**2)/2
        p = (gam - 1)*(rE - r*ke)
        h = rE/r + p/r
        ux, vx, uy, vy = (rux - rx*uv)/r, (rvx - rx*vv)/r, (ruy - ry*uv)/r, (rvy - ry*vv)/r
        px = (gam - 1)*(rEx - rx*ke - r*(uv*ux + vv*vx))
        py = (gam - 1)*(rEy - ry*ke - r*(uv*uy + vv*vy))
        Tx, Ty = gam*M**2*(px*r - p*rx)/r**2, gam*M**2*(py*r - p*ry)/r**2
        txx, txy, tyy = 2*(2*ux - vy)/(3*Re), (uy + vx)/Re, 2*(2*vy - ux)/(3*Re)
        fc = 1/((gam - 1)*M**2*Re*Pr)
        fx = [ru, ru*uv + p + txx, rv*uv + txy, ru*h + uv*txx + vv*txy + fc*Tx]
        fy = [rv, ru*vv + txy, rv*vv + p + tyy, rv*h + uv*txy + vv*tyy + fc*Ty]
        return [sp.diff(fx[k], x[0]) + sp.diff(fy[k], x[1]) for k in range(4)]

    def _evaluate(self, function, points, count):
        values = function(tuple(points[:, d] for d in range(self.nd)))
        return np.column_stack([np.broadcast_to(np.asarray(v, dtype=float), (len(points),)) for v in values])

    def state(self, points):
        return self._evaluate(self._state, points, self.ncu)

    def exact(self, points):
        """[u, q] columns as expected by errors.measure."""
        return np.hstack([self.state(points), self._evaluate(self._gradient, points, self.nd*self.ncu)])

    def source(self, points):
        return self._evaluate(self._source, points, self.ncu)

    def ldg_boundary_flux(self, x, u, q, n):
        """Model Fbou: F(ub,q).n + tau(u-ub), with ub the exact/zero data."""
        if self.name.startswith("Poisson"):
            ub = np.full_like(u, PARAMETERS[self.name][1])
        else:
            ub = self.state(x)
        F = flux(self.name, ub, q)
        return np.einsum("ncd,nd->nc", F, n) + TAU[self.name]*(u - ub)

    def hdg_flux(self, uhat, u, q, n):
        """One-sided HDG/HL element flux F(uhat,q).n + tau(u-uhat)."""
        return np.einsum("ncd,nd->nc", flux(self.name, uhat, q), n) + TAU[self.name]*(u - uhat)
