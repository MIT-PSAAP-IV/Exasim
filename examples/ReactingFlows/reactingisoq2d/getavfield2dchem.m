function avField = getavfield2dchem(y,u,q)
%GETAVFIELD2DCHEM Pointwise AV sensor for axisymmetric chemistry model
%
% State:
%   u = [rho_1,...,rho_ns, rho*u_z, rho*u_r, rhoE]
%
% Exasim convention:
%   q = -grad(u)

% number of species / channels
ns  = numel(u) - 3;
nch = ns + 3;

% regularization parameters
alpha = 1.0e3;
rmin  = 1.0e-3;
ymin  = 1.0e-8;

% conservative variables
rho_i = u(1:ns);
rho   = sum(rho_i);
rhou  = u(ns+1);
rhov  = u(ns+2);

% Exasim convention: q = -grad(u)
drho_dz_i = -q(1:ns);
drhou_dz  = -q(ns+1);
%drhov_dz  = -q(ns+2);

drho_dr_i = -q(nch+1:nch+ns);
%drhou_dr  = -q(nch+ns+1);
drhov_dr  = -q(nch+ns+2);

drho_dz = sum(drho_dz_i);
drho_dr = sum(drho_dr_i);

% regularize rho
rho = rmin + lmax(rho-rmin,alpha);
rhoinv = 1.0 / rho;

% primitive velocities
uz = rhou * rhoinv;
ur = rhov * rhoinv;

% velocity gradients
duz_dz = (drhou_dz - drho_dz * uz) * rhoinv;
%dur_dz = (drhov_dz - drho_dz * ur) * rhoinv;

%duz_dr = (drhou_dr - drho_dr * uz) * rhoinv;
dur_dr = (drhov_dr - drho_dr * ur) * rhoinv;

% regularize radial coordinate near axis
yreg = ymin + lmax(y-ymin,alpha);

% axisymmetric divergence
divu = duz_dz + dur_dr + ur / yreg;

% compression sensor
comp = -divu;

% limit compression
sigm = 100.0;
avField = limiting(comp,0,sigm,alpha,0);
