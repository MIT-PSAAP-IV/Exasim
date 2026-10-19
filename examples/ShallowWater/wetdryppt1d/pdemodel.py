import numpy as np
from sympy import atan, exp, log, pi, sin, sqrt

def bathymetry(x,mu):
    slope=(mu[3]-mu[2])/mu[1]
    return mu[2]+slope*x,slope
def mass(u,q,w,v,x,t,mu,eta): return np.ones(2)
def flux(u,q,w,v,x,t,mu,eta):
    depth=mu[6]+exp(u[0]); return np.array([exp(-u[0])*u[1],u[1]**2/depth+0.5*mu[0]*depth**2])
def source(u,q,w,v,x,t,mu,eta):
    depth=mu[6]+exp(u[0]); _,slope=bathymetry(x[0],mu); return np.array([-exp(-u[0])*u[1]*(-q[0]),-mu[0]*depth*slope])
def wavespeed(u,mu):
    depth=mu[6]+exp(u[0]); return 1+sqrt(mu[0]*depth)
def smoothmax(a,b,epsilon): return 0.5*(a+b+sqrt((a-b)**2+epsilon**2))
def rusanov(left,right,normal,mu):
    left_flux=flux(left,None,None,None,None,None,mu,None); right_flux=flux(right,None,None,None,None,None,mu,None)
    speed=smoothmax(wavespeed(left,mu),wavespeed(right,mu),1e-8)
    return 0.5*(left_flux+right_flux)*normal+0.5*speed*(left-right)
def states(u,x,t,mu):
    left=np.array([log(mu[6]),u[1]]); bed,_=bathymetry(x[0],mu); depth=mu[4]*sin(mu[5]*t)-bed
    right=np.array([log(depth-mu[6]),depth*u[1]/(mu[6]+exp(u[0]))]); return left,right
def fbou(u,q,w,v,x,t,mu,eta,uhat,n,tau):
    left,right=states(u,x,t,mu); return np.column_stack((rusanov(u,left,n[0],mu),rusanov(u,right,n[0],mu)))
def fbouhdg(u,q,w,v,x,t,mu,eta,uhat,n,tau):
    left,right=states(u,x,t,mu); physical=flux(uhat,q,w,v,x,t,mu,eta)*n[0]
    return np.column_stack((rusanov(uhat,left,n[0],mu)-physical,rusanov(uhat,right,n[0],mu)-physical))
def ubou(u,q,w,v,x,t,mu,eta,uhat,n,tau): return np.zeros((2,2))
def initu(x,mu,eta):
    bed,_=bathymetry(x[0],mu); alpha=100.0; height=-bed
    depth=height*(atan(alpha*height)/pi+0.5)-atan(alpha)/pi+0.5+mu[6]
    return np.array([log(depth),0.0])
