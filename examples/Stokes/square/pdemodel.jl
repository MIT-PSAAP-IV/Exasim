function flux(u,q,w,v,x,t,mu,eta)
    gradient=reshape(q,2,2); radius=sqrt(x[1]^2+x[2]^2)
    density=1+mu[4]*sech(mu[5]*(radius^2-mu[6]^2)); source_trace=1-density/mu[3]
    mu[1]*gradient+mu[2]*(tr(gradient)-source_trace)*Matrix{Float64}(I,2,2)
end
source(u,q,w,v,x,t,mu,eta)=zeros(2)
fbou(u,q,w,v,x,t,mu,eta,uhat,n,tau)=zeros(2)
ubou(u,q,w,v,x,t,mu,eta,uhat,n,tau)=zeros(2)
initu(x,mu,eta)=zeros(2)
fbouhdg(u,q,w,v,x,t,mu,eta,uhat,n,tau)=-tau[1]*uhat
