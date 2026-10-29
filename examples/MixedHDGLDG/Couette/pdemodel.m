function pde = pdemodel
% Compressible planar Couette flow with viscous heating; see README.md.
% physicsparam = [gamma, Re, Pr, Mach, U, T(y=0), T(y=1)], q = -grad(u).
pde.mass = @mass;
pde.flux = @flux;
pde.source = @source;
pde.fbou = @fbou;
pde.fbouhdg = @fbouhdg;
pde.ubou = @ubou;
pde.initu = @initu;
end

function m = mass(u,q,w,v,x,t,mu,eta)
m = sym([1.0; 1.0; 1.0; 1.0]);
end

function f = flux(u,q,w,v,x,t,mu,eta)
gam = mu(1); Re = mu(2); Pr = mu(3); Minf = mu(4);
gam1 = gam - 1.0; Re1 = 1/Re; M2 = Minf^2; c23 = 2.0/3.0;
fc = 1/(gam1*M2*Re*Pr);
r = u(1); ru = u(2); rv = u(3); rE = u(4);
rx = q(1); rux = q(2); rvx = q(3); rEx = q(4);
ry = q(5); ruy = q(6); rvy = q(7); rEy = q(8);
r1 = 1/r; uv = ru*r1; vv = rv*r1;
ke = 0.5*(uv*uv+vv*vv);
p = gam1*(rE-r*ke);
h = rE*r1 + p*r1;
ux = (rux - rx*uv)*r1; vx = (rvx - rx*vv)*r1;
px = gam1*(rEx - rx*ke - r*(uv*ux + vv*vx));
Tx = gam*M2*(px*r - p*rx)*r1^2;
uy = (ruy - ry*uv)*r1; vy = (rvy - ry*vv)*r1;
py = gam1*(rEy - ry*ke - r*(uv*uy + vv*vy));
Ty = gam*M2*(py*r - p*ry)*r1^2;
txx = Re1*c23*(2*ux - vy); txy = Re1*(uy + vx); tyy = Re1*c23*(2*vy - ux);
f = [ru, ru*uv+p+txx, rv*uv+txy, ru*h+uv*txx+vv*txy+fc*Tx, ...
     rv, ru*vv+txy, rv*vv+p+tyy, rv*h+uv*txy+vv*tyy+fc*Ty];
f = reshape(f,[4,2]);
end

function ue = exact(x,mu)
gam = mu(1); Pr = mu(3); Minf = mu(4); U = mu(5); T0 = mu(6); T1 = mu(7);
y = x(2); M2 = Minf^2;
T = T0 + (T1 - T0)*y + 0.5*(gam - 1.0)*M2*Pr*U^2*y*(1.0 - y);
p = 1/(gam*M2);
r = gam*M2*p/T;
ue = [r; r*U*y; 0*y; p/(gam - 1.0) + 0.5*r*U^2*y^2];
end

function s = source(u,q,w,v,x,t,mu,eta)
s = sym([0.0; 0.0; 0.0; 0.0]);
end

function ub = ubou(u,q,w,v,x,t,mu,eta,uhat,n,tau)
ub = exact(x,mu);
end

function fb = fbou(u,q,w,v,x,t,mu,eta,uhat,n,tau)
ub = exact(x,mu);
f = flux(ub,q,w,v,x,t,mu,eta);
fb = f(:,1)*n(1) + f(:,2)*n(2) + tau(1)*(u(:) - ub);
end

function fb = fbouhdg(u,q,w,v,x,t,mu,eta,uhat,n,tau)
fb = exact(x,mu) - uhat(:);
end

function u0 = initu(x,mu,eta)
gam = mu(1); Minf = mu(4); U = mu(5);
u0 = [sym(1.0); 0.5*U; sym(0.0); 1/(gam*(gam - 1.0)*Minf^2) + 0.125*U^2];
end
