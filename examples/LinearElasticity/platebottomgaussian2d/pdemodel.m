function pde = pdemodel
pde.flux = @flux;
pde.source = @source;
pde.fbou = @fbou;
pde.fbouhdg = @fbouhdg;
pde.ubou = @ubou;
pde.initu = @initu;
end

function f = flux(u, q, w, v, x, t, mu, eta)
Q = reshape(q, [2 2]);
f = mu(1) * (Q + Q.') + mu(2) * (Q(1, 1) + Q(2, 2)) * eye(2, 2);
end

function s = source(u, q, w, v, x, t, mu, eta)
s = 0 * x;
end

function fb = fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
fb = 0 * x;
end

function ub = ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
ub = 0 * x;
end

function u0 = initu(x, mu, eta)
u0 = 0 * x;
end

function fb = fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
a = mu(3);
b = mu(4);
c = mu(5);

ub = 0 * uhat;
ub(1) = 0.0;
ub(2) = a * exp(-((x(1) - b)^2) / (2 * c^2));

fb_fixed = tau * (0 * uhat(:) - uhat(:));
fb_bottom = tau * (ub(:) - uhat(:));
fb = [fb_fixed fb_bottom];
end

