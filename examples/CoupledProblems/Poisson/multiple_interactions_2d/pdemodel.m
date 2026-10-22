function pde = pdemodel
%PDEMODEL Poisson equation used by all four physical domains.
pde.flux = @flux;
pde.source = @source;
pde.fbou = @fbou;
pde.fint = @fint;
pde.fbouhdg = @fbouhdg;
pde.ubou = @ubou;
pde.initu = @initu;
end

function f = flux(u,q,w,v,x,t,mu,eta)
% mu(1) is the diffusion coefficient; mu(2) selects the interface equation.
f = mu(1)*q;
end

function s = source(u,q,w,v,x,t,mu,eta)
% Manufactured source for u = sin(pi*x)*sin(pi*y).
x1 = x(1);
x2 = x(2);
s = 2*pi*pi*sin(pi*x1)*sin(pi*x2);
end

function fb = fbou(u,q,w,v,x,t,mu,eta,uhat,n,tau)
% Numerical normal flux on an exterior boundary.
f = flux(u,q,w,v,x,t,mu,eta);
fb = f(1)*n(1) + f(2)*n(2) + tau*(u(1)-uhat(1));
end

function ub = ubou(u,q,w,v,x,t,mu,eta,uhat,n,tau)
% Exact Dirichlet data.
ub = sin(pi*x(1))*sin(pi*x(2));
end

function u0 = initu(x,mu,eta)
u0 = 0*x(1)*x(2);
end

function fb = fbouhdg(u,q,w,v,x,t,mu,eta,uhat,n,tau)
% Boundary-condition columns used by boundarycondition in pdeapp1--pdeapp4.
% Odd domains use the numerical-flux equation (mu(2)=1); even domains use
% the trace equation (mu(2)=0). Both interface columns have the same role
% because an interior domain has one interaction on each side.
ub = ubou(u,q,w,v,x,t,mu,eta,uhat,n,tau);
f = flux(u,q,w,v,x,t,mu,eta);
fn = f(1)*n(1) + f(2)*n(2) + tau*(u(1)-uhat(1));
interfaceEquation = mu(2)*fn + (1-mu(2))*uhat(1);
fb = [ub-uhat(1), interfaceEquation, interfaceEquation];
end

function fi = fint(u,q,w,v,x,t,mu,eta,uhat,n,tau)
% Interface-condition columns selected in interaction.m:
%   1: solution continuity; 2: normal-flux balance.
f = flux(u,q,w,v,x,t,mu,eta);
fn = f(1)*n(1) + f(2)*n(2) + tau*(u(1)-uhat(1));
fi = [-uhat(1), fn];
end
