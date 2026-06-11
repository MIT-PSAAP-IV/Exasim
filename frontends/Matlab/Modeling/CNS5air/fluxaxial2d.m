function [f, p, ttt, J_i, rho_i_dim, T_dim, p_dim, stress_dim, heatflux_dim, chemflux_dim] = fluxaxial2d(u, q, w, v, x, t, mu, eta)
%FLUXAXIAL2D Flux for 2D compressible Navier-Stokes in cylindrical coords.
%   Coordinates: x(1)=z, x(2)=r,
%   State: [rho_1,...,rho_5, rho*u_z, rho*u_r, rhoE].
%   Exasim convention: q = -grad(u), where the third block stores
%   derivatives with respect to the coordinate theta (not arc length r*theta).
%
%   This routine returns the flux in the Exasim form
%       dU/dt + dF(:,1)/dz + dF(:,2)/dr  = S,
%
%   The remaining cylindrical geometric terms must be supplied in the source.

ns = 5;
nch = ns + 3;

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

% Coordinates: x(1)=z, x(2)=r
z = x(1); %#ok<NASGU>
r = x(2);

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
rhov = u(ns+2);   % rho*u_r
rhoE = u(ns+3);

% Exasim convention: q = -grad(u)
drho_dz_i = -q(1:ns);
drhou_dz  = -q(ns+1);
drhov_dz  = -q(ns+2);
drhoE_dz  = -q(ns+3);

drho_dr_i = -q((nch+1):(nch+ns));
drhou_dr  = -q(nch+ns+1);
drhov_dr  = -q(nch+ns+2);
drhoE_dr  = -q(nch+ns+3);

av = v(1);
rhoinv = 1.0 / rho;
uz = rhou * rhoinv;
ur = rhov * rhoinv;
E  = rhoE * rhoinv;

rho_i_dim = rho_i * rho_scale;
T = w(1);
T_dim = T * T_scale;

p_dim = pressure(T_dim, rho_i_dim, Mw);
p = p_dim / rhoe_scale;
H = E + p * rhoinv;

fi = zeros(nch,2);
fv = zeros(nch,2);
if class(u) == "sym"
  fi = sym(zeros(nch,2));
  fv = sym(zeros(nch,2)); 
end

% Inviscid + artificial-viscosity fluxes
for ispecies = 1:ns
    fi(ispecies,1) = rho_i(ispecies) * uz - av * drho_dz_i(ispecies);
    fi(ispecies,2) = rho_i(ispecies) * ur - av * drho_dr_i(ispecies);
end

fi(ns+1,1) = rhou * uz + p - av * drhou_dz;
fi(ns+2,1) = rhov * uz     - av * drhov_dz;
fi(ns+3,1) = rhou * H      - av * drhoE_dz;

fi(ns+1,2) = rhou * ur     - av * drhou_dr;
fi(ns+2,2) = rhov * ur + p - av * drhov_dr;
fi(ns+3,2) = rhov * H      - av * drhoE_dr;

[dT_drho_i_dim, dT_drhoe_dim, D_vec, h_vec, mu_d_dim, kappa_dim] = ...
    transportcoefficients(T_dim, rho_i_dim);

drho_dz = sum(drho_dz_i);
drho_dr = sum(drho_dr_i);

duz_dz = (drhou_dz - drho_dz * uz) * rhoinv;
dur_dz = (drhov_dz - drho_dz * ur) * rhoinv;

duz_dr = (drhou_dr - drho_dr * uz) * rhoinv;
dur_dr = (drhov_dr - drho_dr * ur) * rhoinv;

uTu2 = 0.5 * (uz * uz + ur * ur);
duTu2_dz = uz * duz_dz + ur * dur_dz;
duTu2_dr = uz * duz_dr + ur * dur_dr;

dT_drho_i = dT_drho_i_dim * rho_scale / T_scale;
dT_drhoe  = dT_drhoe_dim  * rhoe_scale / T_scale;

dre_drho  = -uTu2;
dre_duTu2 = -rho;
dre_drhoE =  1;

dre_dz = dre_drho * drho_dz + dre_duTu2 * duTu2_dz + dre_drhoE * drhoE_dz;
dre_dr = dre_drho * drho_dr + dre_duTu2 * duTu2_dr + dre_drhoE * drhoE_dr;

dT_dz = sum(dT_drho_i .* drho_dz_i) + dT_drhoe * dre_dz;
dT_dr = sum(dT_drho_i .* drho_dr_i) + dT_drhoe * dre_dr;

h_scale = u_scale^2;
D_scale = u_scale * L_scale;

mu_d  = mu_d_dim / mu_scale;
kappa = kappa_dim / kappa_scale;
D_vec = D_vec / D_scale;
h_vec = h_vec / h_scale;

dY_dz_i = (drho_dz_i * rho - rho_i * drho_dz) * rhoinv * rhoinv;
dY_dr_i = (drho_dr_i * rho - rho_i * drho_dr) * rhoinv * rhoinv;

J_i_z = -rho * D_vec .* dY_dz_i + rho_i .* sum(D_vec .* dY_dz_i);
J_i_r = -rho * D_vec .* dY_dr_i + rho_i .* sum(D_vec .* dY_dr_i);

% Correct axisymmetric Newtonian stresses
beta = 0;
rinv = 1.0 / r;
divu = duz_dz + dur_dr + ur * rinv;

tzz = mu_d * (2.0/3.0) * (2.0 * duz_dz - dur_dr - ur * rinv) / Re + beta * divu / Re;
tzr = mu_d * (duz_dr + dur_dz) / Re;
trr = mu_d * (2.0/3.0) * (2.0 * dur_dr - duz_dz - ur * rinv) / Re + beta * divu / Re;

% Hoop stress (not directly used in flux, but needed in sourceaxial2d)
ttt = mu_d * (2.0/3.0) * (2.0 * ur * rinv - duz_dz - dur_dr) / Re + beta * divu / Re; 

for i = 1:ns
    fv(i,1) = -J_i_z(i);
    fv(i,2) = -J_i_r(i);
end
J_i = -fv(1:ns,:);

fv(ns+1,1) = tzz;
fv(ns+2,1) = tzr;
fv(ns+3,1) = uz * tzz + ur * tzr ...
           - (sum(h_vec .* J_i_z) - kappa * dT_dz / (Re * Pr * Ec));

fv(ns+1,2) = tzr;
fv(ns+2,2) = trr;
fv(ns+3,2) = uz * tzr + ur * trr ...
           - (sum(h_vec .* J_i_r) - kappa * dT_dr / (Re * Pr * Ec));

f = fi - fv;

stress_dim = (mu_scale*u_scale/L_scale)*[tzz tzr; tzr trr];
heatflux_dim = -(kappa_scale*T_scale/L_scale)*[kappa*dT_dz  kappa*dT_dr];
chemflux_dim = (rho_scale*u_scale^3)*[sum(h_vec .* J_i_z) sum(h_vec .* J_i_r)];

end


