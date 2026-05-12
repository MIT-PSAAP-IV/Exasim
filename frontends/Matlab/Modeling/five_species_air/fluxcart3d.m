function f = fluxcart3d(u, q, w, v, x, t, mu, eta)

ns = 5;
nch = ns + 4;

[~, Mw, ~] = thermodynamicsModels();

rho_scale   = eta(1);
u_scale     = eta(2);
rhoe_scale  = eta(3);
T_scale     = eta(4);
mu_scale    = eta(5);
kappa_scale = eta(6);
L_scale     = eta(8);
Ec          = eta(9);
Pr          = eta(10);
Re          = eta(11);

% Conservative variables
rho_i = zeros(ns,1);
rho = 0;

for i = 1:ns
    rho_i(i) = u(i);
    rho = rho + rho_i(i);
end

rhou = u(ns+1);
rhov = u(ns+2);
rhow = u(ns+3);
rhoE = u(ns+4);

% Exasim convention: q = -grad(u)

drho_dx_i = -q(1:ns);
drhou_dx  = -q(ns+1);
drhov_dx  = -q(ns+2);
drhow_dx  = -q(ns+3);
drhoE_dx  = -q(ns+4);

drho_dy_i = -q(nch+1:nch+ns);
drhou_dy  = -q(nch+ns+1);
drhov_dy  = -q(nch+ns+2);
drhow_dy  = -q(nch+ns+3);
drhoE_dy  = -q(nch+ns+4);

drho_dz_i = -q(2*nch+1:2*nch+ns);
drhou_dz  = -q(2*nch+ns+1);
drhov_dz  = -q(2*nch+ns+2);
drhow_dz  = -q(2*nch+ns+3);
drhoE_dz  = -q(2*nch+ns+4);

av = v(1);

rhoinv = 1/rho;

uv = rhou * rhoinv;
vv = rhov * rhoinv;
wv = rhow * rhoinv;

E = rhoE * rhoinv;

% Dimensional state
rho_i_dim = rho_i * rho_scale;

T = w(1);
T_dim = T * T_scale;

p_dim = pressure(T_dim, rho_i_dim, Mw);
p = p_dim / rhoe_scale;

H = E + p * rhoinv;

% Allocate fluxes
fi = zeros(nch,3);
fv = zeros(nch,3);

% Inviscid flux
for i = 1:ns
    fi(i,1) = rho_i(i)*uv - av*drho_dx_i(i);
    fi(i,2) = rho_i(i)*vv - av*drho_dy_i(i);
    fi(i,3) = rho_i(i)*wv - av*drho_dz_i(i);
end

fi(ns+1,1) = rhou*uv + p - av*drhou_dx;
fi(ns+2,1) = rhov*uv - av*drhov_dx;
fi(ns+3,1) = rhow*uv - av*drhow_dx;
fi(ns+4,1) = rhou*H  - av*drhoE_dx;

fi(ns+1,2) = rhou*vv - av*drhou_dy;
fi(ns+2,2) = rhov*vv + p - av*drhov_dy;
fi(ns+3,2) = rhow*vv - av*drhow_dy;
fi(ns+4,2) = rhov*H  - av*drhoE_dy;

fi(ns+1,3) = rhou*wv - av*drhou_dz;
fi(ns+2,3) = rhov*wv - av*drhov_dz;
fi(ns+3,3) = rhow*wv + p - av*drhow_dz;
fi(ns+4,3) = rhow*H  - av*drhoE_dz;

% Transport properties
[dT_drho_i_dim, dT_drhoe_dim, D_vec, h_vec, mu_d_dim, kappa_dim] = ...
    transportcoefficients(T_dim, rho_i_dim);

drho_dx = sum(drho_dx_i);
drho_dy = sum(drho_dy_i);
drho_dz = sum(drho_dz_i);

du_dx = (drhou_dx - drho_dx*uv)*rhoinv;
dv_dx = (drhov_dx - drho_dx*vv)*rhoinv;
dw_dx = (drhow_dx - drho_dx*wv)*rhoinv;

du_dy = (drhou_dy - drho_dy*uv)*rhoinv;
dv_dy = (drhov_dy - drho_dy*vv)*rhoinv;
dw_dy = (drhow_dy - drho_dy*wv)*rhoinv;

du_dz = (drhou_dz - drho_dz*uv)*rhoinv;
dv_dz = (drhov_dz - drho_dz*vv)*rhoinv;
dw_dz = (drhow_dz - drho_dz*wv)*rhoinv;

% Kinetic energy derivatives
uTu2 = 0.5*(uv*uv + vv*vv + wv*wv);

duTu2_dx = uv*du_dx + vv*dv_dx + wv*dw_dx;
duTu2_dy = uv*du_dy + vv*dv_dy + wv*dw_dy;
duTu2_dz = uv*du_dz + vv*dv_dz + wv*dw_dz;

% Temperature derivatives
dT_drho_i = dT_drho_i_dim * rho_scale / T_scale;
dT_drhoe  = dT_drhoe_dim  * rhoe_scale / T_scale;

dre_drho  = -Ec*uTu2;
dre_duTu2 = -Ec*rho;
dre_drhoE = Ec;

dre_dx = dre_drho*drho_dx + dre_duTu2*duTu2_dx + dre_drhoE*drhoE_dx;
dre_dy = dre_drho*drho_dy + dre_duTu2*duTu2_dy + dre_drhoE*drhoE_dy;
dre_dz = dre_drho*drho_dz + dre_duTu2*duTu2_dz + dre_drhoE*drhoE_dz;

dT_dx = sum(dT_drho_i .* drho_dx_i) + dT_drhoe * dre_dx;
dT_dy = sum(dT_drho_i .* drho_dy_i) + dT_drhoe * dre_dy;
dT_dz = sum(dT_drho_i .* drho_dz_i) + dT_drhoe * dre_dz;

% Nondimensional transport
h_scale = u_scale^2;
D_scale = u_scale * L_scale;

mu_d  = mu_d_dim / mu_scale;
kappa = kappa_dim / kappa_scale;
D_vec = D_vec / D_scale;
h_vec = h_vec / h_scale;

% Species diffusion
dY_dx_i = (drho_dx_i*rho - rho_i*drho_dx) * rhoinv^2;
dY_dy_i = (drho_dy_i*rho - rho_i*drho_dy) * rhoinv^2;
dY_dz_i = (drho_dz_i*rho - rho_i*drho_dz) * rhoinv^2;

J_i_x = -rho*D_vec.*dY_dx_i + rho_i.*sum(D_vec.*dY_dx_i);
J_i_y = -rho*D_vec.*dY_dy_i + rho_i.*sum(D_vec.*dY_dy_i);
J_i_z = -rho*D_vec.*dY_dz_i + rho_i.*sum(D_vec.*dY_dz_i);

% Stress tensor
beta = 0;
divu = du_dx + dv_dy + dw_dz;

txx = mu_d*(2/3)*(2*du_dx - dv_dy - dw_dz)/Re + beta*divu/Re;
tyy = mu_d*(2/3)*(2*dv_dy - du_dx - dw_dz)/Re + beta*divu/Re;
tzz = mu_d*(2/3)*(2*dw_dz - du_dx - dv_dy)/Re + beta*divu/Re;

txy = mu_d*(du_dy + dv_dx)/Re;
txz = mu_d*(du_dz + dw_dx)/Re;
tyz = mu_d*(dv_dz + dw_dy)/Re;

% Viscous flux
for i = 1:ns
    fv(i,1) = -J_i_x(i);
    fv(i,2) = -J_i_y(i);
    fv(i,3) = -J_i_z(i);
end

fv(ns+1,1) = txx;
fv(ns+2,1) = txy;
fv(ns+3,1) = txz;
fv(ns+4,1) = uv*txx + vv*txy + wv*txz ...
             - (sum(h_vec.*J_i_x) - kappa*dT_dx/(Re*Pr*Ec));

fv(ns+1,2) = txy;
fv(ns+2,2) = tyy;
fv(ns+3,2) = tyz;
fv(ns+4,2) = uv*txy + vv*tyy + wv*tyz ...
             - (sum(h_vec.*J_i_y) - kappa*dT_dy/(Re*Pr*Ec));

fv(ns+1,3) = txz;
fv(ns+2,3) = tyz;
fv(ns+3,3) = tzz;
fv(ns+4,3) = uv*txz + vv*tyz + wv*tzz ...
             - (sum(h_vec.*J_i_z) - kappa*dT_dz/(Re*Pr*Ec));

f = fi - fv;

end