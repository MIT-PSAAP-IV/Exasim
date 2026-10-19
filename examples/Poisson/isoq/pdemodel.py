import numpy as np

def flux(u,q,w,v,x,t,mu,eta): return mu[0]*x[1]*np.asarray(q)
def source(u,q,w,v,x,t,mu,eta): return np.array([x[1]*mu[1]])
def fbou(u,q,w,v,x,t,mu,eta,uhat,n,tau): return np.array([0.0])
def ubou(u,q,w,v,x,t,mu,eta,uhat,n,tau): return np.array([0.0])
def initu(x,mu,eta): return np.array([0.0])
def fbouhdg(u,q,w,v,x,t,mu,eta,uhat,n,tau):
    physical=mu[0]*np.asarray(q)
    natural=physical@n+tau[0]*(u[0]-uhat[0])
    return np.array([natural,tau[0]*(0.0-uhat[0]),tau[0]*(1.0-uhat[0]),natural+mu[3]*(mu[2]-uhat[0])])
