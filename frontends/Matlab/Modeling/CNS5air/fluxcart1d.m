function [f, J_i] = fluxcart1d(u, q, w, v, x, t, mu, eta)

ns = 5;
nch = ns + 2;

% Molecular weights
[~, Mw, ~] = thermodynamicsModels();

% Reference scales
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

% Conservative variables
rho_i = zeros(ns,1);
rho = 0;
if class(u) == "sym"
  rho_i = sym(zeros(ns,1));
  rho = sym(0);  
end

for i = 1:ns
    rho_i(i) = u(i);
    rho = rho + rho_i(i);
end

rhou = u(ns+1);
rhoE = u(ns+2);

% Exasim convention: q = -grad(u)

drho_dx_i = -q(1:ns);
drhou_dx  = -q(ns+1);
drhoE_dx  = -q(ns+2);

av = v(1);

rhoinv = 1/rho;

uv = rhou * rhoinv;
E  = rhoE * rhoinv;

% Dimensional thermodynamic state
rho_i_dim = rho_i * rho_scale;

T = w(1);
T_dim = T * T_scale;

p_dim = pressure(T_dim, rho_i_dim, Mw);
p = p_dim / rhoe_scale;

H = E + p * rhoinv;

% Allocate flux
fi = zeros(nch,1);
fv = zeros(nch,1);
if class(u) == "sym"
  fi = sym(zeros(nch,1));
  fv = sym(zeros(nch,1)); 
end

% Inviscid flux
for i = 1:ns
    fi(i) = rho_i(i)*uv - av*drho_dx_i(i);
end

fi(ns+1) = rhou*uv + p - av*drhou_dx;
fi(ns+2) = rhou*H  - av*drhoE_dx;

% Transport properties
[dT_drho_i_dim, dT_drhoe_dim, D_vec, h_vec, mu_d_dim, kappa_dim] = ...
    transportcoefficients(T_dim, rho_i_dim);

drho_dx = sum(drho_dx_i);

du_dx = (drhou_dx - drho_dx*uv) * rhoinv;

% kinetic energy derivative
uTu2 = 0.5 * uv^2;
duTu2_dx = uv * du_dx;

% temperature derivatives
dT_drho_i = dT_drho_i_dim * rho_scale / T_scale;
dT_drhoe  = dT_drhoe_dim  * rhoe_scale / T_scale;

dre_drho  = -uTu2;
dre_duTu2 = -rho;
dre_drhoE =  1;

dre_dx = dre_drho*drho_dx + dre_duTu2*duTu2_dx + dre_drhoE*drhoE_dx;

dT_dx = sum(dT_drho_i .* drho_dx_i) + dT_drhoe * dre_dx;

% Nondimensional transport
h_scale = u_scale^2;
D_scale = u_scale * L_scale;

mu_d  = mu_d_dim / mu_scale;
kappa = kappa_dim / kappa_scale;
D_vec = D_vec / D_scale;
h_vec = h_vec / h_scale;

% Species diffusion
dY_dx_i = (drho_dx_i*rho - rho_i*drho_dx) * rhoinv^2;

J_i_x = -rho * D_vec .* dY_dx_i + rho_i .* sum(D_vec .* dY_dx_i);

% Newtonian stress tensor (1D)
beta = 0;

txx = mu_d * (4/3) * du_dx / Re + beta * du_dx / Re;

% Viscous flux
for i = 1:ns
    fv(i) = -J_i_x(i);
end
J_i = -fv(1:ns);

fv(ns+1) = txx;

fv(ns+2) = uv * txx ...
           - (sum(h_vec .* J_i_x) - kappa * dT_dx / (Re * Pr * Ec));

f = fi - fv;

end