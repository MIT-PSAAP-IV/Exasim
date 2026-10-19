flux(u,q,w,v,x,t,mu,eta) = begin
    gradient=reshape(q,2,2); mu[1]*(gradient+transpose(gradient))+mu[2]*tr(gradient)*Matrix{Float64}(I,2,2)
end
function source(u,q,w,v,x,t,mu,eta)
    reynolds=1/mu[1]; lam=0.5*(reynolds-sqrt(reynolds^2+16*pi^2)); x1,x2=x
    [lam*exp(lam*x1)*(cos(2*pi*x2)-exp(lam*x1)),-lam^2*exp(lam*x1)*sin(2*pi*x2)/(2*pi)]
end
fbou(u,q,w,v,x,t,mu,eta,uhat,n,tau)=zeros(2)
ubou(u,q,w,v,x,t,mu,eta,uhat,n,tau)=zeros(2)
initu(x,mu,eta)=zeros(2)
function fbouhdg(u,q,w,v,x,t,mu,eta,uhat,n,tau)
    reynolds=1/mu[1]; lam=0.5*reynolds-sqrt(0.25*reynolds^2+4*pi^2); x1,x2=x
    exact=[1-exp(lam*x1)*cos(2*pi*x2),lam*exp(lam*x1)*sin(2*pi*x2)/(2*pi)]
    tau[1]*(exact-uhat)
end
