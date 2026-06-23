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
% mu = [mu_lam, lambda, bump_amp, bump_loc, bump_width]
%
% Boundarycondition = [1, 1, 2, 1]:
%   BC type 1 (col 1): fixed — zero displacement (groups 1,2,4)
%   BC type 2 (col 2): wall — prescribed normal displacement (group 3)

bump_amp = mu(3);
bump_loc = mu(4);
bump_width = mu(5);

% Type 1: fixed (zero displacement on all DOFs)
fb_fixed = tau * (0 * uhat(:) - uhat(:));

% Type 2: wall — Gaussian displacement along outward normal
u_bump = bump_amp * exp(-((x(1) - bump_loc)^2) / (2 * bump_width^2));
ub_wall = u_bump * n(:);
fb_wall = tau * (ub_wall - uhat(:));

fb = [fb_fixed fb_wall];
end
