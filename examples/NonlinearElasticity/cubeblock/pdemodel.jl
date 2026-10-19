using LinearAlgebra
function flux(u,q,w,v,x,t,mu,eta)
    deformation = -reshape(q,3,3); jacobian=det(deformation)
    stress=mu[1].*deformation-mu[1].*transpose(inv(deformation))+mu[2].*(jacobian-1).*jacobian.*transpose(inv(deformation))
    -stress
end
source(u,q,w,v,x,t,mu,eta)=0.0.*x; fbou(u,q,w,v,x,t,mu,eta,uhat,n,tau)=0.0.*x; ubou(u,q,w,v,x,t,mu,eta,uhat,n,tau)=0.0.*x; initu(x,mu,eta)=x
function fbouhdg(u,q,w,v,x,t,mu,eta,uhat,n,tau)
    traction=flux(u,q,w,v,x,t,mu,eta)*n+tau[1].*(u-uhat)
    hcat(traction,x-uhat,traction+mu[3:5])
end
