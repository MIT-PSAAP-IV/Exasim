flux(u,q,w,v,x,t,mu,eta)=x[1].*q
function source(u,q,w,v,x,t,mu,eta)
 lref,eref,e_eps,n0,z0,sigma=mu[2],mu[4],mu[5],mu[7],mu[8],mu[9]; nscaled=n0*lref^3; zscaled=z0/lref; sscaled=sigma/lref
 [x[1]*(e_eps/(eref*lref^2))*nscaled*exp(-((x[2]-zscaled)^2+x[1]^2)/sscaled^2)]
end
function fbou(u,q,w,v,x,t,mu,eta,uhat,n,tau); f=flux(u,q,w,v,x,t,mu,eta); natural=sum(f.*n)+tau[1]*(u[1]-uhat[1]); [natural,0.0,natural,0.0]; end
ubou(u,q,w,v,x,t,mu,eta,uhat,n,tau)=[0.0,u[1],mu[6]/(mu[4]*mu[2]),u[1]]; initu(x,mu,eta)=[0.0]
function fbouhdg(u,q,w,v,x,t,mu,eta,uhat,n,tau); f=flux(u,q,w,v,x,t,mu,eta); natural=sum(f.*n)+tau[1]*(u[1]-uhat[1]); [-tau[1]*uhat[1],natural,tau[1]*(mu[6]/(mu[4]*mu[2])-uhat[1]),natural]; end
