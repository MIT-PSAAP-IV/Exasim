positivepart(value)=begin alpha=1e6; value*(atan(alpha*value)/pi+0.5)-atan(alpha)/pi+0.5 end
bathymetry(x,mu)=begin slope=(mu[4]-mu[3])/mu[2]; (mu[3]+slope*x,slope) end
mass(u,q,w,v,x,t,mu,eta)=ones(2)
function flux(u,q,w,v,x,t,mu,eta)
    depth=positivepart(u[1]); safe=depth+mu[7]; momentum=u[2]; av=v[1]
    [momentum+30*av*q[1],momentum^2/safe+0.5*mu[1]*depth^2+30*av*q[2]]
end
function source(u,q,w,v,x,t,mu,eta)
    depth=positivepart(u[1]); _,slope=bathymetry(x[1],mu); wetness=0.5*(1-tanh((depth-1e-3)/1e-3))
    [0.0,-mu[1]*depth*slope-0.1*wetness*u[2]]
end
function limiting(value,lower,upper,alpha,offset)
    limited=lower+(value-offset)*(atan(alpha*(value-offset))/pi+0.5)-atan(alpha)/pi+0.5
    difference=limited-upper; limited-(difference*(atan(alpha*difference)/pi+0.5)-atan(alpha)/pi+0.5)
end
function avfield(u,q,w,v,x,t,mu,eta)
    alpha=1e3; depth,momentum=u; depth_x,momentum_x=q; positive=depth*(atan(alpha*depth)/pi+0.5)-atan(alpha)/pi+0.5
    derivative=atan(alpha*depth)/pi+0.5+depth*alpha/(pi*(1+alpha^2*depth^2)); denominator=positive+mu[7]
    velocity_x=(momentum_x*denominator-momentum*derivative*depth_x)/denominator^2
    sensor=limiting(velocity_x/0.001,0.0,0.9,alpha,0.0)/0.9
    [(sensor-0.1)*(atan(alpha*(sensor-0.1))/pi+0.5)-atan(alpha)/pi+0.5]*tanh(10*(mu[2]-x[1])/mu[2])
end
fbou(u,q,w,v,x,t,mu,eta,uhat,n,tau)=flux(u,q,w,v,x,t,mu,eta)*n[1]+tau[1]*(u-uhat)
ubou(u,q,w,v,x,t,mu,eta,uhat,n,tau)=zeros(2,2)
initu(x,mu,eta)=begin bed,_=bathymetry(x[1],mu); [positivepart(-bed),0.0] end
function swematrix(state,normal,mu)
    depth=positivepart(state[1]); velocity=state[2]/depth; speed=sqrt(mu[1]*depth)
    eigen=Diagonal([tanh(100*normal*(velocity-speed)),tanh(100*normal*(velocity+speed))])
    vectors=[1.0 1.0; velocity-speed velocity+speed]; inverse=[velocity+speed -1.0; speed-velocity 1.0]/(2*speed)
    vectors*eigen*inverse
end
function fbouhdg(u,q,w,v,x,t,mu,eta,uhat,n,tau)
    left=[0.0,0.0]; bed,_=bathymetry(x[1],mu); depth=mu[5]*sin(mu[6]*t)-bed; right=[depth,depth*u[2]/u[1]]
    matrix=swematrix(uhat,n[1],mu)
    [0.5*((u+left)+matrix*(u-left))-uhat 0.5*((u+right)+matrix*(u-right))-uhat]
end
