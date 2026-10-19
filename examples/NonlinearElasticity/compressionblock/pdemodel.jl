function flux(u,q,w,v,x,t,mu,eta)
 f11,f21,f12,f22=q; determinant=f11*f22-f12*f21; shear,lame=mu[1],mu[2]; c=f12*f21-f11*f22+1
 [shear*f11-lame*f22*c-shear*f22/determinant shear*f12+lame*f21*c+shear*f21/determinant; shear*f21+lame*f12*c+shear*f12/determinant shear*f22-lame*f11*c-shear*f11/determinant]
end
source(u,q,w,v,x,t,mu,eta)=0.0.*x; fbou(u,q,w,v,x,t,mu,eta,uhat,n,tau)=0.0.*x; ubou(u,q,w,v,x,t,mu,eta,uhat,n,tau)=0.0.*x; initu(x,mu,eta)=x
function fbouhdg(u,q,w,v,x,t,mu,eta,uhat,n,tau)
 traction=flux(u,q,w,v,x,t,mu,eta)*n+tau[1].*(u-uhat); fixed=tau[1].*(x-uhat); loaded=traction+mu[3:4]
 vertical=copy(traction); vertical[2]=tau[1]*(x[2]-uhat[2]); horizontal=copy(traction); horizontal[1]=tau[1]*(x[1]-uhat[1]); pressure=copy(horizontal); pressure[2]=traction[2]-mu[5]
 hcat(traction,fixed,loaded,vertical,horizontal,pressure)
end
