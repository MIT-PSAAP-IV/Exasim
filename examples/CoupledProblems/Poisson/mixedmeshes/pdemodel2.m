function pde = pdemodel2
pde.flux = @flux; pde.source = @source; pde.fbou = @fbou; pde.fint = @fint;
pde.fbouhdg = @fbouhdg; pde.ubou = @ubou; pde.initu = @initu;
end

function f = flux(u, q, w, v, x, t, mu, eta)
f = mu*q;
end

function s = source(u, q, w, v, x, t, mu, eta)
s = (2*pi*pi)*sin(pi*x(1))*sin(pi*x(2));
end

function fb = fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
f = flux(u, q, w, v, x, t, mu, eta);
fb = f(1)*n(1) + f(2)*n(2) + tau*u(1);
end

function ub = ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
ub = sym(0.0);
end

function u0 = initu(x, mu, eta)
u0 = 0*x(1)*x(2);
end

function fb = fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
fb = [-uhat(1), uhat(1)];
end

function fi = fint(u, q, w, v, x, t, mu, eta, uhat, n, tau)
f = flux(u, q, w, v, x, t, mu, eta);
fi = f(1)*n(1) + f(2)*n(2) + tau*(u(1)-uhat(1));
end
