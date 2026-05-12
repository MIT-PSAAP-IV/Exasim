function [f, J_i] = fluxaxial1d_quasi1d(u, q, w, v, x, t, mu, eta)
%FLUXAXIAL1D_QUASI1D Quasi-1D nozzle-area flux for five-species reacting air.
%   Coordinate: x(1) = z.
%   Exasim convention: q = -d(u)/dz.
%
%   This routine implements the quasi-1D nozzle model in conservative,
%   area-weighted form:
%
%       d(AU)/dt + d(AF)/dz = S,
%
%   where the state stored in u is
%
%       u = [rho_1*A, ..., rho_5*A, rho*u_z*A, rhoE*A]^T.
%
%   The companion source term should provide
%
%       S_momentum = p * dA/dz
%
%   plus any Cartesian source terms multiplied by A (e.g. chemistry).
%
%   The helper function
%
%       [A, Aderiv] = nozzlearea(z)
%
%   is assumed available. If your function is named differently, change the
%   single line below accordingly.

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

z = x(1);
[A, ~] = nozzlearea(z);
Ainv = 1.0 / A;

%--------------------------------------------------------------------------
% Conservative variables stored in area-weighted form.
%--------------------------------------------------------------------------
rhoA_i = zeros(ns,1);
rho_i  = zeros(ns,1);
rho = 0.0;
if class(u) == "sym"
  rhoA_i = sym(zeros(ns,1));
  rho_i = sym(zeros(ns,1));
  rho = sym(0);  
end

for ispecies = 1:ns
    rhoA_i(ispecies) = u(ispecies);
    rho_i(ispecies)  = rhoA_i(ispecies) * Ainv;
    rho = rho + rho_i(ispecies);
end

rhouA = u(ns+1);
rhoEA = u(ns+2);

rhou = rhouA * Ainv;
rhoE = rhoEA * Ainv;

% Exasim convention: q = -d(u)/dz, where u = A*U.
drhouA_dz  = -q(ns+1);
drhoEA_dz  = -q(ns+2);
drhouA_i_dz = -q(1:ns);

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

%--------------------------------------------------------------------------
% Recover physical z-derivatives from derivatives of area-weighted state.
% If Uhat = A U, then U_z = (Uhat_z - A_z U)/A.
%--------------------------------------------------------------------------
[~, Aderiv] = nozzlearea(z);

drhou_dz  = (drhouA_dz  - Aderiv * rhou)  * Ainv;
drhoE_dz  = (drhoEA_dz  - Aderiv * rhoE)  * Ainv;
drhoz_i   = (drhouA_i_dz - Aderiv * rho_i) * Ainv;

drhoz = sum(drhoz_i);

%--------------------------------------------------------------------------
% Physical fluxes per unit area, then multiply by A to obtain the quasi-1D
% conserved flux AF.
%--------------------------------------------------------------------------
fi = zeros(nch,1);
fv = zeros(nch,1);
if class(u) == "sym"
  fi = sym(zeros(nch,1));
  fv = sym(zeros(nch,1)); 
end

for ispecies = 1:ns
    fi(ispecies) = rho_i(ispecies) * uz - av * drhoz_i(ispecies);
end
fi(ns+1) = rhou * uz + p - av * drhou_dz;
fi(ns+2) = rhou * H      - av * drhoE_dz;

[dT_drho_i_dim, dT_drhoe_dim, D_vec, h_vec, mu_d_dim, kappa_dim] = ...
    transportcoefficients(T_dim, rho_i_dim);

duz_dz = (drhou_dz - drhoz * uz) * rhoinv;

uTu2 = 0.5 * uz * uz;
duTu2_dz = uz * duz_dz;

dT_drho_i = dT_drho_i_dim * rho_scale / T_scale;
dT_drhoe  = dT_drhoe_dim  * rhoe_scale / T_scale;

dre_drho  = -uTu2;
dre_duTu2 = -rho;
dre_drhoE =  1;

dre_dz = dre_drho * drhoz + dre_duTu2 * duTu2_dz + dre_drhoE * drhoE_dz;
dT_dz = sum(dT_drho_i .* drhoz_i) + dT_drhoe * dre_dz;

h_scale = u_scale^2;
D_scale = u_scale * L_scale;

mu_d  = mu_d_dim / mu_scale;
kappa = kappa_dim / kappa_scale;
D_vec = D_vec / D_scale;
h_vec = h_vec / h_scale;

dY_dz_i = (drhoz_i * rho - rho_i * drhoz) * rhoinv * rhoinv;
J_i_z = -rho * D_vec .* dY_dz_i + rho_i .* sum(D_vec .* dY_dz_i);

beta = 0;
tzz = mu_d * (4.0/3.0) * duz_dz / Re + beta * duz_dz / Re;

for i = 1:ns
    fv(i) = -J_i_z(i);
end
J_i = -fv(1:ns);

fv(ns+1) = tzz;
fv(ns+2) = uz * tzz ...
         - (sum(h_vec .* J_i_z) - kappa * dT_dz / (Re * Pr * Ec));

% Return area-weighted total flux.
f = A * (fi - fv);

end
