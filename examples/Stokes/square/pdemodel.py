import numpy as np
from sympy import cosh, sqrt

def flux(u,q,w,v,x,t,mu,eta):
    gradient=np.reshape(q,(2,2),order="F"); radius=sqrt(x[0]**2+x[1]**2)
    density=1+mu[3]/cosh(mu[4]*(radius**2-mu[5]**2)); source_trace=1-density/mu[2]
    return mu[0]*gradient+mu[1]*(np.trace(gradient)-source_trace)*np.eye(2)
def source(u,q,w,v,x,t,mu,eta): return np.zeros(2)
def fbou(u,q,w,v,x,t,mu,eta,uhat,n,tau): return np.zeros(2)
def ubou(u,q,w,v,x,t,mu,eta,uhat,n,tau): return np.zeros(2)
def initu(x,mu,eta): return np.zeros(2)
def fbouhdg(u,q,w,v,x,t,mu,eta,uhat,n,tau): return -tau[0]*uhat
