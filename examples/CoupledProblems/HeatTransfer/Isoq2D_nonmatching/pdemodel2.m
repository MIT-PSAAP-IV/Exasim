function pde = pdemodel
pde.flux = @flux;
pde.source = @source;
pde.fbou = @fbou;
pde.fint = @fint;
pde.fbouhdg = @fbouhdg;
pde.ubou = @ubou;
pde.initu = @initu;
end

function f = flux(u, q, w, v, x, t, mu, eta)
f = mu(1)*q;
end

function s = source(u, q, w, v, x, t, mu, eta)
% Axisymmetric conduction: div(k grad(T)) + k*T_r/r = 0.
s = -mu(1)*q(2)/x(2);
end

function fb = fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
fb = sym(0.0);
end

function ub = ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
ub = sym(0.0);
end

function fb = fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
f = flux(u, q, w, v, x, t, mu, eta);
fh = f(1)*n(1) + f(2)*n(2) + tau*(u(1)-uhat(1));
fixedTemperature = tau*(mu(2)-uhat(1));

% Fixed temperature, lower symmetry boundary, coupled interface.
fb = [fixedTemperature fh fh];
end

function fi = fint(u, q, w, v, x, t, mu, eta, uhat, n, tau)
% Supply the solid temperature to the fluid interface condition.
fi = -uhat(1);
end

function u0 = initu(x, mu, eta)
u0 = mu(2);
end
