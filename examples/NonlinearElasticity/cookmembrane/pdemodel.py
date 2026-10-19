import numpy as np


def flux(u, q, w, v, x, t, mu, eta):
    f11, f21, f12, f22 = q
    determinant = f11 * f22 - f12 * f21
    shear, lame = mu[0], mu[1]
    correction = f12 * f21 - f11 * f22 + 1.0
    return np.array([
        [shear*f11 - lame*f22*correction - shear*f22/determinant,
         shear*f12 + lame*f21*correction + shear*f21/determinant],
        [shear*f21 + lame*f12*correction + shear*f12/determinant,
         shear*f22 - lame*f11*correction - shear*f11/determinant],
    ])


def source(u, q, w, v, x, t, mu, eta):
    return 0.0 * np.asarray(x)


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return 0.0 * np.asarray(x)


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    return 0.0 * np.asarray(x)


def initu(x, mu, eta):
    return np.asarray(x)


def fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    traction = flux(u, q, w, v, x, t, mu, eta) @ n + tau[0] * (u - uhat)
    fixed = tau[0] * (x - uhat)
    loaded = traction + np.asarray(mu[2:4])
    return np.column_stack((traction, fixed, loaded))
