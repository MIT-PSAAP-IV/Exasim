import numpy as np
from sympy import exp, sin
def flux(u,q,w,v,x,t,mu,eta): return mu[0]*(2.0*u[0]/(1.0+u[0]*u[0]))*np.asarray(q)
def source(u,q,w,v,x,t,mu,eta): return np.array([x[0]*sin(x[1])])
def sourcew(u,q,w,v,x,t,mu,eta): return np.array([exp(w[0])-1.0-u[0]*u[0]])
def eos(u,q,w,v,x,t,mu,eta): return sourcew(u,q,w,v,x,t,mu,eta)
def fbou(u,q,w,v,x,t,mu,eta,uhat,n,tau):
 f=flux(u,q,w,v,x,t,mu,eta); return np.array([f@n+tau[0]*u[0]])
def ubou(u,q,w,v,x,t,mu,eta,uhat,n,tau): return np.array([0.0])
def initu(x,mu,eta): return np.array([1.0])
def initw(x,mu,eta): return np.array([0.693147180559945])
def fbouhdg(u,q,w,v,x,t,mu,eta,uhat,n,tau):
 f=flux(uhat,q,w,v,x,t,mu,eta); return np.array([tau[0]*(mu[1]-uhat[0]),f@n+tau[0]*(u[0]-uhat[0])])
