function pde = pdemodel1
% Model 1 of the 3D coupled Poisson examples: -div(grad u) = s with the exact
% solution u = exp(x) sin(pi y) sin(pi z), whose normal flux on the interface
% x = 0.5 is nonzero.
pde.flux = @flux;
pde.source = @source;
pde.fbou = @fbou;
pde.fint = @fint;
pde.fbouhdg = @fbouhdg;
pde.ubou = @ubou;
pde.initu = @initu;
end

function f = flux(u, q, w, v, x, t, mu, eta)
f = mu*q;
end

function s = source(u, q, w, v, x, t, mu, eta)
s = (2*pi*pi-1)*exp(x(1))*sin(pi*x(2))*sin(pi*x(3));
end

function fb = fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
f = flux(u, q, w, v, x, t, mu, eta);
fb = f(1)*n(1) + f(2)*n(2) + f(3)*n(3) + tau*(u(1)-uhat(1));
end

function ub = ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
ub = exp(x(1))*sin(pi*x(2))*sin(pi*x(3));
end

function u0 = initu(x, mu, eta)
u0 = 0*x(1)*x(2)*x(3);
end

function fb = fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
% Column 1: Dirichlet boundary (condition 1); column 2: the coupled interface.
ub = exp(x(1))*sin(pi*x(2))*sin(pi*x(3));
f = flux(u, q, w, v, x, t, mu, eta);
fi = f(1)*n(1) + f(2)*n(2) + f(3)*n(3) + tau*(u(1)-uhat(1));
fb = [ub-uhat(1), fi];
end

function fi = fint(u, q, w, v, x, t, mu, eta, uhat, n, tau)
% Model 1 sends its trace; model 2 enforces uhat2 = uhat1.
fi = -uhat(1);
end
