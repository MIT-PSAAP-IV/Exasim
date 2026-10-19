import numpy as np
from sympy import cos, exp, pi, sin, sqrt

def flux(u,q,w,v,x,t,mu,eta):
    gradient=np.reshape(q,(2,2),order="F")
    return mu[0]*(gradient+gradient.T)+mu[1]*np.trace(gradient)*np.eye(2)

def source(u,q,w,v,x,t,mu,eta):
    reynolds=1.0/mu[0]; lam=0.5*(reynolds-sqrt(reynolds**2+16*pi**2)); x1,x2=x
    return np.array([lam*exp(lam*x1)*(cos(2*pi*x2)-exp(lam*x1)),
                     -(lam**2*exp(lam*x1)*sin(2*pi*x2))/(2*pi)])

def fbou(u,q,w,v,x,t,mu,eta,uhat,n,tau): return np.zeros(2)
def ubou(u,q,w,v,x,t,mu,eta,uhat,n,tau): return np.zeros(2)
def initu(x,mu,eta): return np.zeros(2)

def fbouhdg(u,q,w,v,x,t,mu,eta,uhat,n,tau):
    reynolds=1.0/mu[0]; lam=0.5*reynolds-sqrt(0.25*reynolds**2+4*pi**2); x1,x2=x
    exact=np.array([1-exp(lam*x1)*cos(2*pi*x2),lam*exp(lam*x1)*sin(2*pi*x2)/(2*pi)])
    return tau[0]*(exact-uhat)
