"""Nonlinear generated-physics fixture with every AD input family active."""
import numpy as np


def flux(u, q, w, v, x, t, mu, eta):
    return np.array([[u[0]**2 + q[0]*v[0] + w[0], u[0]*u[1] + q[2]],
                     [u[1]**2 + q[1]*v[0], u[1]*w[0] + q[3]]])


def source(u, q, w, v, x, t, mu, eta):
    return np.array([u[0]*v[0] + w[0]**2, u[1]*w[0] + v[0]**2])


def mass(u, q, w, v, x, t, mu, eta):
    return np.array([1 + u[0]**2 + v[0]*w[0], 2 + u[1]*w[0]])


def avfield(u, q, w, v, x, t, mu, eta):
    return np.array([u[0]**2 + w[0]*v[0]])


def fbou(u, q, w, v, x, t, mu, eta, uh, n, tau):
    b = np.array([u[0]**2 + w[0]*v[0] - uh[0], u[1]*uh[1] + q[0]])
    return np.column_stack((b, 2*b))


def ubou(u, q, w, v, x, t, mu, eta, uh, n, tau):
    b = np.array([u[0]*w[0] + v[0] + uh[0], u[1]**2 + w[0]])
    return np.column_stack((b, -b))


def fbouhdg(u, q, w, v, x, t, mu, eta, uh, n, tau):
    return fbou(u, q, w, v, x, t, mu, eta, uh, n, tau)


def fhat(u1,q1,w1,v1,x,t,mu,eta,uh,n,tau,u2,q2,w2,v2):
    return np.array([u1[0]*u2[0] + w1[0]*v2[0] + uh[0],
                     u1[1]*w2[0] + u2[1]*v1[0] - uh[1]])


def stab(u1,q1,w1,v1,x,t,mu,eta,uh,n,tau,u2,q2,w2,v2):
    return np.array([tau[0]*(u1[0]+u2[0])**2 + w1[0]*w2[0],
                     v1[0]*v2[0] + uh[1]**2])


def initu(x, mu, eta):
    return np.array([0.1,0.2])


def initw(x, mu, eta):
    return np.array([1.0])
