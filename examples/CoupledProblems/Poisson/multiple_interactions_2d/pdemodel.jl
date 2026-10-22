module pdemodel

# Poisson model shared by all four physical domains.
flux(u,q,w,v,x,t,mu,eta)=mu[1]*q

function source(u,q,w,v,x,t,mu,eta)
    2*pi*pi*sin(pi*x[1])*sin(pi*x[2])
end

function fbou(u,q,w,v,x,t,mu,eta,uhat,n,tau)
    f=flux(u,q,w,v,x,t,mu,eta)
    f[1]*n[1]+f[2]*n[2]+tau[1]*(u[1]-uhat[1])
end

ubou(u,q,w,v,x,t,mu,eta,uhat,n,tau)=sin(pi*x[1])*sin(pi*x[2])
initu(x,mu,eta)=0*x[1]*x[2]

function fbouhdg(u,q,w,v,x,t,mu,eta,uhat,n,tau)
    boundary_value=ubou(u,q,w,v,x,t,mu,eta,uhat,n,tau)
    normal_flux=fbou(u,q,w,v,x,t,mu,eta,uhat,n,tau)
    interface_equation=mu[2]*normal_flux+(1-mu[2])*uhat[1]
    [boundary_value-uhat[1] interface_equation interface_equation]
end

function fint(u,q,w,v,x,t,mu,eta,uhat,n,tau)
    # Columns 1 and 2 impose solution continuity and normal-flux balance.
    normal_flux=fbou(u,q,w,v,x,t,mu,eta,uhat,n,tau)
    [-uhat[1] normal_flux]
end

end
