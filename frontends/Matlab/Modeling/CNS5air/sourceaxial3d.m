function s = sourceaxial3d(u, q, w, v, x, t, mu, eta)

ns = 5;

% Start with any pre-existing source (chemistry, body forces, ...)
s = sourcend(u, q, w, v, x, t, mu, eta);

% Cylindrical radius
r = x(2);
rinv = 1.0 / r;

% Total radial flux and cylindrical stress components from flux routine
[f, p, ttt, trt] = fluxaxial3d(u, q, w, v, x, t, mu, eta);

% Generic cylindrical-divergence contribution: -F_r / r
s = s - f(:,2) * rinv;

% Conservative variables
rho_ur = u(ns+2);   % rho*u_r
rho_ut = u(ns+3);   % rho*u_theta
rho = sum(u(1:ns));
ut = rho_ut / rho;

% Radial momentum basis correction: +(p + rho*u_theta^2 - tau_thetatheta)/r
s(ns+2) = s(ns+2) + (p + rho_ut * ut - ttt) * rinv;

% Theta momentum basis correction: -(rho*u_r*u_theta - tau_rtheta)/r
s(ns+3) = s(ns+3) - (rho_ur * ut - trt) * rinv;

end