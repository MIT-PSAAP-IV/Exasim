import numpy as np
from sympy import exp
def flux(u,q,w,v,x,t,mu,eta): return x[0]*np.asarray(q)
def source(u,q,w,v,x,t,mu,eta):
 lref,eref,e_eps,n0,z0,sigma=mu[1],mu[3],mu[4],mu[6],mu[7],mu[8]
 nscaled=n0*lref**3; zscaled=z0/lref; sscaled=sigma/lref
 return np.array([x[0]*(e_eps/(eref*lref**2))*nscaled*exp(-((x[1]-zscaled)**2+x[0]**2)/sscaled**2)])
def fbou(u,q,w,v,x,t,mu,eta,uhat,n,tau):
 f=flux(u,q,w,v,x,t,mu,eta); natural=f@n+tau[0]*(u[0]-uhat[0]); return np.array([natural,0.0,natural,0.0])
def ubou(u,q,w,v,x,t,mu,eta,uhat,n,tau): return np.array([0.0,u[0],mu[5]/(mu[3]*mu[1]),u[0]])
def initu(x,mu,eta): return np.array([0.0])
def fbouhdg(u,q,w,v,x,t,mu,eta,uhat,n,tau):
 f=flux(u,q,w,v,x,t,mu,eta); natural=f@n+tau[0]*(u[0]-uhat[0]); return np.array([-tau[0]*uhat[0],natural,tau[0]*(mu[5]/(mu[3]*mu[1])-uhat[0]),natural])
