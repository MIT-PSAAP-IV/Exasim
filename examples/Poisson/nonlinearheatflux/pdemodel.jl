flux(u,q,w,v,x,t,mu,eta)=mu[1].*(2u[1]/(1+u[1]^2)).*q
source(u,q,w,v,x,t,mu,eta)=[x[1]*sin(x[2])]
sourcew(u,q,w,v,x,t,mu,eta)=[exp(w[1])-1-u[1]^2]; eos(u,q,w,v,x,t,mu,eta)=sourcew(u,q,w,v,x,t,mu,eta)
function fbou(u,q,w,v,x,t,mu,eta,uhat,n,tau); f=flux(u,q,w,v,x,t,mu,eta); [sum(f.*n)+tau[1]*u[1]]; end
ubou(u,q,w,v,x,t,mu,eta,uhat,n,tau)=[0.0]; initu(x,mu,eta)=[1.0]; initw(x,mu,eta)=[0.693147180559945]
function fbouhdg(u,q,w,v,x,t,mu,eta,uhat,n,tau); f=flux(uhat,q,w,v,x,t,mu,eta); [tau[1]*(mu[2]-uhat[1]),sum(f.*n)+tau[1]*(u[1]-uhat[1])]; end
