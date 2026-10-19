import numpy as np
from sympy import Matrix

def flux(u,q,w,v,x,t,mu,eta):
    deformation = -Matrix(3,3,lambda i,j:q[i+3*j])
    jacobian = deformation.det()
    stress = mu[0]*deformation - mu[0]*deformation.inv().T + mu[1]*(jacobian-1)*jacobian*deformation.inv().T
    return -np.array(stress.tolist(), dtype=object)
def source(u,q,w,v,x,t,mu,eta): return 0.0*np.asarray(x)
def fbou(u,q,w,v,x,t,mu,eta,uhat,n,tau): return 0.0*np.asarray(x)
def ubou(u,q,w,v,x,t,mu,eta,uhat,n,tau): return 0.0*np.asarray(x)
def initu(x,mu,eta): return np.asarray(x)
def fbouhdg(u,q,w,v,x,t,mu,eta,uhat,n,tau):
    traction=flux(u,q,w,v,x,t,mu,eta)@n+tau[0]*(u-uhat)
    return np.column_stack((traction,x-uhat,traction+np.asarray(mu[2:5])))
