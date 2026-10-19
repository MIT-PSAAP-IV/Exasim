bathymetry(x,mu)=begin slope=(mu[4]-mu[3])/mu[2]; (mu[3]+slope*x,slope) end
mass(u,q,w,v,x,t,mu,eta)=ones(2)
flux(u,q,w,v,x,t,mu,eta)=begin depth=mu[7]+exp(u[1]); [exp(-u[1])*u[2],u[2]^2/depth+0.5*mu[1]*depth^2] end
source(u,q,w,v,x,t,mu,eta)=begin depth=mu[7]+exp(u[1]); _,slope=bathymetry(x[1],mu); [-exp(-u[1])*u[2]*(-q[1]),-mu[1]*depth*slope] end
wavespeed(u,mu)=1+sqrt(mu[1]*(mu[7]+exp(u[1])))
smoothmax(a,b,epsilon)=0.5*(a+b+sqrt((a-b)^2+epsilon^2))
function rusanov(left,right,normal,mu)
    speed=smoothmax(wavespeed(left,mu),wavespeed(right,mu),1e-8)
    0.5*(flux(left,nothing,nothing,nothing,nothing,nothing,mu,nothing)+flux(right,nothing,nothing,nothing,nothing,nothing,mu,nothing))*normal+0.5*speed*(left-right)
end
function states(u,x,t,mu)
    left=[log(mu[7]),u[2]]; bed,_=bathymetry(x[1],mu); depth=mu[5]*sin(mu[6]*t)-bed
    right=[log(depth-mu[7]),depth*u[2]/(mu[7]+exp(u[1]))]; left,right
end
function fbou(u,q,w,v,x,t,mu,eta,uhat,n,tau)
    left,right=states(u,x,t,mu); [rusanov(u,left,n[1],mu) rusanov(u,right,n[1],mu)]
end
function fbouhdg(u,q,w,v,x,t,mu,eta,uhat,n,tau)
    left,right=states(u,x,t,mu); physical=flux(uhat,q,w,v,x,t,mu,eta)*n[1]
    [rusanov(uhat,left,n[1],mu)-physical rusanov(uhat,right,n[1],mu)-physical]
end
ubou(u,q,w,v,x,t,mu,eta,uhat,n,tau)=zeros(2,2)
function initu(x,mu,eta)
    bed,_=bathymetry(x[1],mu); alpha=100.0; height=-bed
    depth=height*(atan(alpha*height)/pi+0.5)-atan(alpha)/pi+0.5+mu[7]; [log(depth),0.0]
end
