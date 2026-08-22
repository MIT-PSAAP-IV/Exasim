from numpy import array, reshape
from sympy import atan, cos, pi, sin, sqrt


def mass(u, q, w, v, x, t, mu, eta):
    return array([1.0, 1.0, 1.0, 1.0, 1.0])


def flux(u, q, w, v, x, t, mu, eta):
    gam = mu[0]
    gam1 = gam - 1.0
    Re = mu[1]
    Pr = mu[2]
    Minf = mu[3]
    Re1 = 1.0 / Re
    M2 = Minf * Minf
    c23 = 2.0 / 3.0
    av = v[0]

    r = u[0]
    ru = u[1]
    rv = u[2]
    rw = u[3]
    rE = u[4]
    rx = q[0]
    rux = q[1]
    rvx = q[2]
    rwx = q[3]
    rEx = q[4]
    ry = q[5]
    ruy = q[6]
    rvy = q[7]
    rwy = q[8]
    rEy = q[9]
    rz = q[10]
    ruz = q[11]
    rvz = q[12]
    rwz = q[13]
    rEz = q[14]

    r1 = 1.0 / r
    uv = ru * r1
    vv = rv * r1
    wv = rw * r1
    E = rE * r1
    ke = 0.5 * (uv * uv + vv * vv + wv * wv)
    p = gam1 * (rE - r * ke)
    T = gam * M2 * p * r1
    mut = 1.4042 * T * sqrt(T) / (T + 0.40417)
    fc = mut / (gam1 * M2 * Re * Pr)
    h = E + p * r1

    fi = array([
        ru, ru * uv + p, rv * uv, rw * uv, ru * h,
        rv, ru * vv, rv * vv + p, rw * vv, rv * h,
        rw, ru * wv, rv * wv, rw * wv + p, rw * h,
    ])

    ux = (rux - rx * uv) * r1
    vx = (rvx - rx * vv) * r1
    wx = (rwx - rx * wv) * r1
    kex = uv * ux + vv * vx + wv * wx
    px = gam1 * (rEx - rx * ke - r * kex)
    Tx = gam * M2 * (px * r - p * rx) * r1 * r1

    uy = (ruy - ry * uv) * r1
    vy = (rvy - ry * vv) * r1
    wy = (rwy - ry * wv) * r1
    key = uv * uy + vv * vy + wv * wy
    py = gam1 * (rEy - ry * ke - r * key)
    Ty = gam * M2 * (py * r - p * ry) * r1 * r1

    uz = (ruz - rz * uv) * r1
    vz = (rvz - rz * vv) * r1
    wz = (rwz - rz * wv) * r1
    kez = uv * uz + vv * vz + wv * wz
    pz = gam1 * (rEz - rz * ke - r * kez)
    Tz = gam * M2 * (pz * r - p * rz) * r1 * r1

    txx = mut * Re1 * c23 * (2.0 * ux - vy - wz)
    txy = mut * Re1 * (uy + vx)
    txz = mut * Re1 * (uz + wx)
    tyy = mut * Re1 * c23 * (2.0 * vy - ux - wz)
    tyz = mut * Re1 * (vz + wy)
    tzz = mut * Re1 * c23 * (2.0 * wz - ux - vy)

    fv = array([
        0.0, txx, txy, txz, uv * txx + vv * txy + wv * txz + fc * Tx,
        0.0, txy, tyy, tyz, uv * txy + vv * tyy + wv * tyz + fc * Ty,
        0.0, txz, tyz, tzz, uv * txz + vv * tyz + wv * tzz + fc * Tz,
    ])
    fl = array([
        av * rx, av * rux, av * rvx, av * rwx, av * rEx,
        av * ry, av * ruy, av * rvy, av * rwy, av * rEy,
        av * rz, av * ruz, av * rvz, av * rwz, av * rEz,
    ])

    return reshape(fi + fv + fl, (5, 3), order="F")


def source(u, q, w, v, x, t, mu, eta):
    return array([0.0, 0.0, 0.0, 0.0, 0.0])


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    f = flux(u, q, w, v, x, t, mu, eta)
    return f[:, 0] * n[0] + f[:, 1] * n[1] + f[:, 2] * n[2] + tau[0] * (u - uhat)


def fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    f = flux(u, q, w, v, x, t, mu, eta)
    return f[:, 0] * n[0] + f[:, 1] * n[1] + f[:, 2] * n[2] + tau[0] * (u - uhat)


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return array([0.0, 0.0, 0.0, 0.0, 0.0])


def initu(x, mu, eta):
    L = 1.0
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    gam = mu[0]
    gam1 = gam - 1.0
    Minf = mu[3]
    M2 = Minf * Minf
    p0 = 1.0 / (gam * M2)

    uv = sin(x1 / L) * cos(x2 / L) * cos(x3 / L)
    vv = -cos(x1 / L) * sin(x2 / L) * cos(x3 / L)
    wv = 0.0
    p = p0 + 0.0625 * (cos(2.0 * x1 / L) + cos(2.0 * x2 / L)) * (2.0 + cos(2.0 * x3 / L))
    r = gam * M2 * p

    return array([
        r,
        r * uv,
        r * vv,
        r * wv,
        p / gam1 + 0.5 * r * (uv * uv + vv * vv + wv * wv),
    ])


def initv(x, mu, eta):
    return array([0.0])


def avfield(u, q, w, v, x, t, mu, eta):
    gam = mu[0]
    hm = mu[5]
    avcoeff = mu[6]
    porder = mu[7]
    return array([getavfield3d(u, q, hm, gam, avcoeff, porder)])


def qoivolume(u, q, w, v, x, t, mu, eta):
    gam = mu[0]
    Re = mu[1]
    Minf = mu[3]
    rhoRef = mu[4]
    M2 = Minf * Minf
    Omega = (2.0 * pi) ** 3

    r = u[0]
    ru = u[1]
    rv = u[2]
    rw = u[3]
    rE = u[4]
    rx = q[0]
    rux = q[1]
    rvx = q[2]
    rwx = q[3]
    ry = q[5]
    ruy = q[6]
    rvy = q[7]
    rwy = q[8]
    rz = q[10]
    ruz = q[11]
    rvz = q[12]
    rwz = q[13]

    r1 = 1.0 / r
    uv = ru * r1
    vv = rv * r1
    wv = rw * r1
    ke = 0.5 * (uv * uv + vv * vv + wv * wv)
    p = (gam - 1.0) * (rE - r * ke)
    T = gam * M2 * p * r1
    mut = 1.4042 * T * sqrt(T) / (T + 0.40417)

    ux = (rux - rx * uv) * r1
    vx = (rvx - rx * vv) * r1
    wx = (rwx - rx * wv) * r1
    uy = (ruy - ry * uv) * r1
    vy = (rvy - ry * vv) * r1
    wy = (rwy - ry * wv) * r1
    uz = (ruz - rz * uv) * r1
    vz = (rvz - rz * vv) * r1
    wz = (rwz - rz * wv) * r1

    curl2 = (wy - vz) ** 2 + (uz - wx) ** 2 + (vx - uy) ** 2
    divu = ux + vy + wz

    return array([
        0.5 * r * (uv * uv + vv * vv + wv * wv) / (rhoRef * Omega),
        mut * curl2 / (rhoRef * Re * Omega),
        (4.0 / 3.0) * mut * divu * divu / (rhoRef * Re * Omega),
    ])


def getavfield3d(u, q, hm, gam, avcoeff, porder):
    gam1 = gam - 1.0
    alpha = 1.0e3
    rmin = 1.0e-3
    Hmin = 1.0e-4
    divmax = 50.0
    sbmax = 4.0
    sb0 = 0.1

    r = u[0]
    ru = u[1]
    rv = u[2]
    rw = u[3]
    rE = u[4]
    rx = q[0]
    rux = q[1]
    rvx = q[2]
    rwx = q[3]
    ry = q[5]
    ruy = q[6]
    rvy = q[7]
    rwy = q[8]
    rz = q[10]
    ruz = q[11]
    rvz = q[12]
    rwz = q[13]

    r = rmin + lmax(r - rmin, alpha)
    r1 = 1.0 / r
    uv = ru * r1
    vv = rv * r1
    wv = rw * r1
    E = rE * r1
    kinetic = 0.5 * (uv * uv + vv * vv + wv * wv)
    H = gam * E - gam1 * kinetic
    H = Hmin + lmax(H - Hmin, alpha)
    cstar = sqrt((2.0 * gam1 * H) / (gam + 1.0))

    ux = (rux - rx * uv) * r1
    vx = (rvx - rx * vv) * r1
    wx = (rwx - rx * wv) * r1
    uy = (ruy - ry * uv) * r1
    vy = (rvy - ry * vv) * r1
    wy = (rwy - ry * wv) * r1
    uz = (ruz - rz * uv) * r1
    vz = (rvz - rz * vv) * r1
    wz = (rwz - rz * wv) * r1

    divu = ux + vy + wz
    vortx = wy - vz
    vorty = uz - wx
    vortz = vx - uy
    vort = sqrt(vortx * vortx + vorty * vorty + vortz * vortz)

    divu = limiting(divu, 0.0, divmax, alpha, 5.0)
    vort = limiting(vort, 0.0, divmax, alpha, 0.0)
    ducros = divu * divu / (divu * divu + vort * vort + 1.0e-16)

    sb = sqrt(hm / porder) * (divu / cstar) * ducros
    return avcoeff * limiting(sb, 0.0, sbmax, alpha, sb0)


def lmax(x, alpha):
    return x * (atan(alpha * x) / pi + 0.5) - atan(alpha) / pi + 0.5


def lmin(x, alpha):
    return x - lmax(x, alpha)


def limiting(x, xmin, xmax, alpha, x0):
    f = xmin + lmax(x - x0, alpha)
    return lmin(f - xmax, alpha) + xmax
