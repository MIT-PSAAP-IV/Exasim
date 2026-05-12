function [f, J_i] = fluxaxial1d(u, q, w, v, x, t, mu, eta)
%FLUXAXIAL1D Flux for 1D compressible Navier-Stokes in cylindrical geometry.
%   Coordinate: x(1)=z.
%   State: [rho_1,...,rho_5, rho*u_z, rhoE].
%   Exasim convention: q = -grad(u).
%
%   This routine is the 1D reduction of fluxaxial2d.m obtained by assuming
%       u_r = 0,  d/dr(.) = 0,
%   so the rmuined flux is the axial (z) flux only. The cylindrical
%   geometric effects must be supplied separately in the source term if a
%   quasi-1D / axisymmetric reduction is intended.
%
%   In this 1D reduction, the constitutive law reduces to
%       tau_zz = (4/3) * mu * du_z/dz / Re
%   which is the standard 1D compressible Newtonian stress.

ns = 5;
nch = ns + 2;

[~, Mw, ~] = thermodynamicsModels();

rho_scale   = mu(1);
u_scale     = mu(2);
rhoe_scale  = mu(3);
T_scale     = mu(4);
mu_scale    = mu(5);
kappa_scale = mu(6);
L_scale     = mu(8);
Ec          = mu(9);
Pr          = mu(10);
Re          = mu(11);

rho_i = zeros(ns,1);
rho = 0.0;
if class(u) == "sym"
  rho_i = sym(zeros(ns,1));
  rho = sym(0);  
end

for ispecies = 1:ns
    rho_i(ispecies) = u(ispecies);
    rho = rho + rho_i(ispecies);
end

rhou = u(ns+1);   % rho*u_z
rhoE = u(ns+2);

% Exasim convention: q = -grad(u)
drhou_dz  = -q(ns+1);
drhoe_dz  = -q(ns+2);
drho_dz_i = -q(1:ns);

av = v(1);
rhoinv = 1.0 / rho;
uz = rhou * rhoinv;
E  = rhoE * rhoinv;

rho_i_dim = rho_i * rho_scale;
T = w(1);
T_dim = T * T_scale;

p_dim = pressure(T_dim, rho_i_dim, Mw);
p = p_dim / rhoe_scale;
H = E + p * rhoinv;

fi = zeros(nch,1);
fv = zeros(nch,1);
if class(u) == "sym"
  fi = sym(zeros(nch,1));
  fv = sym(zeros(nch,1)); 
end

% Inviscid + artificial-viscosity fluxes
for ispecies = 1:ns
    fi(ispecies) = rho_i(ispecies) * uz - av * drho_dz_i(ispecies);
end
fi(ns+1) = rhou * uz + p - av * drhou_dz;
fi(ns+2) = rhou * H      - av * drhoe_dz;

[dT_drho_i_dim, dT_drhoe_dim, D_vec, h_vec, mu_d_dim, kappa_dim] = ...
    transportcoefficients(T_dim, rho_i_dim);

drhoz = sum(drho_dz_i);
duz_dz = (drhou_dz - drhoz * uz) * rhoinv;

uTu2 = 0.5 * uz * uz;
duTu2_dz = uz * duz_dz;

dT_drho_i = dT_drho_i_dim * rho_scale / T_scale;
dT_drhoe  = dT_drhoe_dim  * rhoe_scale / T_scale;

dre_drho  = -uTu2;
dre_duTu2 = -rho;
dre_drhoE =  1;

dre_dz = dre_drho * drhoz + dre_duTu2 * duTu2_dz + dre_drhoE * drhoe_dz;
dT_dz = sum(dT_drho_i .* drho_dz_i) + dT_drhoe * dre_dz;

h_scale = u_scale^2;
D_scale = u_scale * L_scale;

mu_d  = mu_d_dim / mu_scale;
kappa = kappa_dim / kappa_scale;
D_vec = D_vec / D_scale;
h_vec = h_vec / h_scale;

dY_dz_i = (drho_dz_i * rho - rho_i * drhoz) * rhoinv * rhoinv;
J_i_z = -rho * D_vec .* dY_dz_i + rho_i .* sum(D_vec .* dY_dz_i);

% 1D reduction of axisymmetric Newtonian stress: u_r = 0, d/dr(.) = 0
bmu = 0;
tzz = mu_d * (4.0/3.0) * duz_dz / Re + bmu * duz_dz / Re;

for i = 1:ns
    fv(i) = -J_i_z(i);
end
J_i = -fv(1:ns);

fv(ns+1) = tzz;
fv(ns+2) = uz * tzz ...
         - (sum(h_vec .* J_i_z) - kappa * dT_dz / (Re * Pr * Ec));

f = fi - fv;

end
