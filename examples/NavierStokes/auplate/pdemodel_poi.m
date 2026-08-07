function pde = pdemodel_poi
%PDEMODEL_POI Poisson model for checking the auplate 2D mesh.
%
% Boundary-condition columns:
%   1: Dirichlet, u = 1
%   2: homogeneous Neumann
%   3: Dirichlet, u = 0

pde.flux = @flux;
pde.source = @source;
pde.fbou = @fbou;
pde.fbouhdg = @fbouhdg;
pde.ubou = @ubou;
pde.initu = @initu;
end

function f = flux(u, q, w, v, x, t, mu, eta)
f = mu(1)*q;
end

function s = source(u, q, w, v, x, t, mu, eta)
s = sym(0.0);
end

function fb = fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
f = flux(u, q, w, v, x, t, mu, eta);
neumann0 = f(1)*n(1) + f(2)*n(2) + tau*(u(1)-uhat(1));
fb = [tau*(u(1)-1.0), neumann0, tau*u(1)];
end

function ub = ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
ub = [sym(1.0), u(1), sym(0.0)];
end

function u0 = initu(x, mu, eta)
u0 = sym(0.0);
end

function fb = fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
f = flux(u, q, w, v, x, t, mu, eta);
neumann0 = f(1)*n(1) + f(2)*n(2) + tau*(u(1)-uhat(1));
dirichlet1 = tau*(1.0 - uhat(1));
dirichlet0 = tau*(0.0 - uhat(1));
fb = [dirichlet1, neumann0, dirichlet0];
end
