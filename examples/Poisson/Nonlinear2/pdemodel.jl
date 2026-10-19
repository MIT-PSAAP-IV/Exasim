flux(u,q,w,v,x,t,mu,eta)=mu[1].*(2.0+u[1]^2).*q
source(u,q,w,v,x,t,mu,eta)=[x[1]*sin(x[2])]
function fbou(u,q,w,v,x,t,mu,eta,uhat,n,tau); f=flux(u,q,w,v,x,t,mu,eta); [sum(f.*n)+tau[1]*(u[1]-uhat[1]),0.0]; end
ubou(u,q,w,v,x,t,mu,eta,uhat,n,tau)=[mu[2],u[1]]
initu(x,mu,eta)=[1.0]
function fbouhdg(u,q,w,v,x,t,mu,eta,uhat,n,tau); f=flux(u,q,w,v,x,t,mu,eta); [tau[1]*(mu[2]-uhat[1]),sum(f.*n)+tau[1]*(u[1]-uhat[1])]; end
