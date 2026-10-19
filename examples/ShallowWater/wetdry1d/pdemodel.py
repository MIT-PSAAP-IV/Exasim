import numpy as np
from sympy import atan, pi, simplify, sin, sqrt, tanh

def positivepart(value):
    alpha=1e6
    return value*(atan(alpha*value)/pi+0.5)-atan(alpha)/pi+0.5

def bathymetry(x,mu):
    slope=(mu[3]-mu[2])/mu[1]
    return mu[2]+slope*x,slope

def mass(u,q,w,v,x,t,mu,eta): return np.ones(2)

def flux(u,q,w,v,x,t,mu,eta):
    depth=positivepart(u[0]); safe=depth+mu[6]; momentum=u[1]; av=v[0]
    return np.array([momentum+30*av*q[0],momentum**2/safe+0.5*mu[0]*depth**2+30*av*q[1]])

def source(u,q,w,v,x,t,mu,eta):
    depth=positivepart(u[0]); _,slope=bathymetry(x[0],mu)
    wetness=0.5*(1-tanh((depth-1e-3)/1e-3))
    return np.array([0.0,-mu[0]*depth*slope-0.1*wetness*u[1]])

def avfield(u,q,w,v,x,t,mu,eta):
    alpha=1e3; depth,momentum=u; depth_x,momentum_x=q; positive=depth*(atan(alpha*depth)/pi+0.5)-atan(alpha)/pi+0.5
    derivative=atan(alpha*depth)/pi+0.5+depth*alpha/(pi*(1+alpha**2*depth**2)); denominator=positive+mu[6]
    velocity_x=(momentum_x*denominator-momentum*derivative*depth_x)/denominator**2
    sensor=velocity_x/0.001; sensor=limiting(sensor,0.0,0.9,alpha,0.0)/0.9
    return np.array([(sensor-0.1)*(atan(alpha*(sensor-0.1))/pi+0.5)-atan(alpha)/pi+0.5])*tanh(10*(mu[1]-x[0])/mu[1])

def limiting(value,lower,upper,alpha,offset):
    limited=lower+(value-offset)*(atan(alpha*(value-offset))/pi+0.5)-atan(alpha)/pi+0.5
    difference=limited-upper
    return limited-(difference*(atan(alpha*difference)/pi+0.5)-atan(alpha)/pi+0.5)

def fbou(u,q,w,v,x,t,mu,eta,uhat,n,tau): return flux(u,q,w,v,x,t,mu,eta)*n[0]+tau[0]*(u-uhat)
def ubou(u,q,w,v,x,t,mu,eta,uhat,n,tau): return np.zeros((2,2))

def initu(x,mu,eta):
    bed,_=bathymetry(x[0],mu)
    return np.array([positivepart(-bed),0.0])

def fbouhdg(u,q,w,v,x,t,mu,eta,uhat,n,tau):
    left=np.array([0.0,0.0]); bed,_=bathymetry(x[0],mu); depth=mu[4]*sin(mu[5]*t)-bed
    right=np.array([depth,depth*u[1]/u[0]])
    matrix=swematrix(uhat,n[0],mu)
    return np.column_stack((0.5*((u+left)+matrix@(u-left))-uhat,0.5*((u+right)+matrix@(u-right))-uhat))

def swematrix(state,normal,mu):
    depth=positivepart(state[0]); velocity=state[1]/depth; speed=sqrt(mu[0]*depth)
    eigen=np.array([[tanh(100*normal*(velocity-speed)),0.0],[0.0,tanh(100*normal*(velocity+speed))]])
    vectors=np.array([[1.0,1.0],[velocity-speed,velocity+speed]])
    inverse=np.array([[velocity+speed,-1.0],[speed-velocity,1.0]])/(2*speed)
    return np.vectorize(simplify)(vectors@eigen@inverse)
