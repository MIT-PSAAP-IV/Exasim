# Unit-square Poisson: q=-grad(u), physicsparam=[kappa, ub].
flux(u,q,w,v,x,t,mu,eta) = mu[1]*q
source(u,q,w,v,x,t,mu,eta) = 2*pi^2*sin(pi*x[1])*sin(pi*x[2])
mass(u,q,w,v,x,t,mu,eta) = 1.0
ubou(u,q,w,v,x,t,mu,eta,uhat,n,tau) = mu[2]
fbouhdg(u,q,w,v,x,t,mu,eta,uhat,n,tau) = tau[1]*(mu[2]-uhat[1])
initu(x,mu,eta) = 0.0

function fbou(u,q,w,v,x,t,mu,eta,uhat,n,tau)
    f = flux(u,q,w,v,x,t,mu,eta)
    return f[1]*n[1] + f[2]*n[2] + tau[1]*(u[1]-mu[2])
end
