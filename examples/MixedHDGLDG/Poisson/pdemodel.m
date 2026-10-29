function pde = pdemodel
% Unit-square Poisson: q=-grad(u), physicsparam=[kappa, ub].
pde.flux = @flux;
pde.source = @source;
pde.mass = @mass;
pde.fbou = @fbou;
pde.fbouhdg = @fbouhdg;
pde.ubou = @ubou;
pde.initu = @initu;
end

function f = flux(u,q,w,v,x,t,mu,eta)
f = mu(1)*q;
end

function s = source(u,q,w,v,x,t,mu,eta)
s = 2*pi^2*sin(pi*x(1))*sin(pi*x(2));
end

function m = mass(u,q,w,v,x,t,mu,eta)
m = sym(1);
end

function ub = ubou(u,q,w,v,x,t,mu,eta,uhat,n,tau)
ub = mu(2);
end

function fb = fbou(u,q,w,v,x,t,mu,eta,uhat,n,tau)
f = flux(u,q,w,v,x,t,mu,eta);
fb = f(1)*n(1) + f(2)*n(2) + tau(1)*(u(1)-mu(2));
end

function fb = fbouhdg(u,q,w,v,x,t,mu,eta,uhat,n,tau)
fb = tau(1)*(mu(2)-uhat(1));
end

function u0 = initu(x,mu,eta)
u0 = sym(0);
end
