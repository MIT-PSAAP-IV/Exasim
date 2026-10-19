flux(u,q,w,v,x,t,mu,eta)=mu[1].*x[2].*q
source(u,q,w,v,x,t,mu,eta)=[x[2]*mu[2]]
fbou(u,q,w,v,x,t,mu,eta,uhat,n,tau)=[0.0]
ubou(u,q,w,v,x,t,mu,eta,uhat,n,tau)=[0.0]
initu(x,mu,eta)=[0.0]
function fbouhdg(u,q,w,v,x,t,mu,eta,uhat,n,tau)
 physical=mu[1].*q; natural=sum(physical.*n)+tau[1]*(u[1]-uhat[1])
 [natural,tau[1]*(0.0-uhat[1]),tau[1]*(1.0-uhat[1]),natural+mu[4]*(mu[3]-uhat[1])]
end
