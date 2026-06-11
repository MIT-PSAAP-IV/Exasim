function [f, p, ttt, trt] = fluxaxial3d(u, q, w, v, x, t, mu, eta)
%FLUXAXIAL3D Flux for 3D compressible Navier-Stokes in cylindrical coords.
%   Coordinates: x(1)=z, x(2)=r, x(3)=theta.
%   State: [rho_1,...,rho_5, rho*u_z, rho*u_r, rho*u_theta, rhoE].
%   Exasim convention: q = -grad(u), where the third block stores
%   derivatives with respect to the coordinate theta (not arc length r*theta).
%
%   This routine returns the flux in the Exasim form
%       dU/dt + dF(:,1)/dz + dF(:,2)/dr + dF(:,3)/dtheta = S,
%   where F(:,3) equals the physical theta-flux divided by r so that
%       dF(:,3)/dtheta = (1/r) d(physical theta-flux)/dtheta.
%
%   The remaining cylindrical geometric terms must be supplied in the source.

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

% Coordinates
z = x(1); %#ok<NASGU>
r = x(2);
theta = x(3); %#ok<NASGU>
rinv = 1.0 / r;
rinv2 = rinv * rinv;

% Conservative variables
rho_i = zeros(ns,1);
rho = 0.0;
for i = 1:ns
    rho_i(i) = u(i);
    rho = rho + rho_i(i);
end

rhouz = u(ns+1);
rhour = u(ns+2);
rhout = u(ns+3);
rhoE  = u(ns+4);

% Exasim convention: q = -grad(u)
% z-derivatives
idx = 0;
drhoz_i = -q(idx + (1:ns));
drhouz_z = -q(idx + ns + 1);
drhour_z = -q(idx + ns + 2);
drhout_z = -q(idx + ns + 3);
drhoe_z  = -q(idx + ns + 4);

% r-derivatives
idx = nch;
drho_r_i = -q(idx + (1:ns));
drhouz_r = -q(idx + ns + 1);
drhour_r = -q(idx + ns + 2);
drhout_r = -q(idx + ns + 3);
drhoe_r  = -q(idx + ns + 4);

% theta-coordinate derivatives
idx = 2*nch;
drho_t_i = -q(idx + (1:ns));
drhouz_t = -q(idx + ns + 1);
drhour_t = -q(idx + ns + 2);
drhout_t = -q(idx + ns + 3);
drhoe_t  = -q(idx + ns + 4);

av = v(1);

rhoinv = 1.0 / rho;
uz = rhouz * rhoinv;
ur = rhour * rhoinv;
ut = rhout * rhoinv;
E  = rhoE  * rhoinv;

% Dimensional thermodynamic state
rho_i_dim = rho_i * rho_scale;
T = w(1);
T_dim = T * T_scale;

p_dim = pressure(T_dim, rho_i_dim, Mw);
p = p_dim / rhoe_scale;
H = E + p * rhoinv;

% Allocate fluxes. Column 3 is the Exasim theta-flux = physical theta-flux / r.
fi = zeros(nch,3);
fv = zeros(nch,3);

% Inviscid + artificial-viscosity fluxes
for i = 1:ns
    fi(i,1) = rho_i(i) * uz - av * drhoz_i(i);
    fi(i,2) = rho_i(i) * ur - av * drho_r_i(i);
    fi(i,3) = rho_i(i) * ut * rinv - av * drho_t_i(i) * rinv2;
end

% z-flux
fi(ns+1,1) = rhouz * uz + p - av * drhouz_z;
fi(ns+2,1) = rhour * uz     - av * drhour_z;
fi(ns+3,1) = rhout * uz     - av * drhout_z;
fi(ns+4,1) = rhouz * H      - av * drhoe_z;

% r-flux
fi(ns+1,2) = rhouz * ur     - av * drhouz_r;
fi(ns+2,2) = rhour * ur + p - av * drhour_r;
fi(ns+3,2) = rhout * ur     - av * drhout_r;
fi(ns+4,2) = rhour * H      - av * drhoe_r;

% theta-flux divided by r
fi(ns+1,3) = (rhouz * ut) * rinv - av * drhouz_t * rinv2;
fi(ns+2,3) = (rhour * ut) * rinv - av * drhour_t * rinv2;
fi(ns+3,3) = (rhout * ut + p) * rinv - av * drhout_t * rinv2;
fi(ns+4,3) = (rhout * H) * rinv - av * drhoe_t * rinv2;

% Transport properties and temperature sensitivities
[dT_drho_i_dim, dT_drhoe_dim, D_vec, h_vec, mu_d_dim, kappa_dim] = ...
    transportcoefficients(T_dim, rho_i_dim);

drhoz = sum(drhoz_i);
drhor = sum(drho_r_i);
drhot = sum(drho_t_i);

% Velocity derivatives wrt coordinates z, r, theta
% Third-direction coordinate derivative must later be converted to physical
% theta-derivative by division by r.
duz_dz = (drhouz_z - drhoz * uz) * rhoinv;
dur_dz = (drhour_z - drhoz * ur) * rhoinv;
dut_dz = (drhout_z - drhoz * ut) * rhoinv;

duz_dr = (drhouz_r - drhor * uz) * rhoinv;
dur_dr = (drhour_r - drhor * ur) * rhoinv;
dut_dr = (drhout_r - drhor * ut) * rhoinv;

duz_dtcoord = (drhouz_t - drhot * uz) * rhoinv;
dur_dtcoord = (drhour_t - drhot * ur) * rhoinv;
dut_dtcoord = (drhout_t - drhot * ut) * rhoinv;

% Physical theta-derivatives = coordinate derivative / r
duz_dt = duz_dtcoord * rinv;
dur_dt = dur_dtcoord * rinv;
dut_dt = dut_dtcoord * rinv;

% Kinetic energy derivatives
uTu2 = 0.5 * (uz * uz + ur * ur + ut * ut);
duTu2_dz = uz * duz_dz + ur * dur_dz + ut * dut_dz;
duTu2_dr = uz * duz_dr + ur * dur_dr + ut * dut_dr;
duTu2_dtcoord = uz * duz_dtcoord + ur * dur_dtcoord + ut * dut_dtcoord;

% dT/drho_i and dT/drhoe in dimensionless variables
dT_drho_i = dT_drho_i_dim * rho_scale / T_scale;
dT_drhoe  = dT_drhoe_dim  * rhoe_scale / T_scale;

dre_drho  = -Ec * uTu2;
dre_duTu2 = -Ec * rho;
dre_drhoE =  Ec;

dre_dz = dre_drho * drhoz + dre_duTu2 * duTu2_dz + dre_drhoE * drhoe_z;
dre_dr = dre_drho * drhor + dre_duTu2 * duTu2_dr + dre_drhoE * drhoe_r;
dre_dtcoord = dre_drho * drhot + dre_duTu2 * duTu2_dtcoord + dre_drhoE * drhoe_t;

dT_dz = sum(dT_drho_i .* drhoz_i) + dT_drhoe * dre_dz;
dT_dr = sum(dT_drho_i .* drho_r_i) + dT_drhoe * dre_dr;
dT_dtcoord = sum(dT_drho_i .* drho_t_i) + dT_drhoe * dre_dtcoord;
dT_dt = dT_dtcoord * rinv;

% Nondimensionalize transport properties
h_scale = u_scale^2;
D_scale = u_scale * L_scale;

mu_d  = mu_d_dim / mu_scale;
kappa = kappa_dim / kappa_scale;
D_vec = D_vec / D_scale;
h_vec = h_vec / h_scale;

% Species mass fractions and mixture-averaged diffusion fluxes
dY_dz_i = (drhoz_i * rho - rho_i * drhoz) * rhoinv * rhoinv;
dY_dr_i = (drho_r_i * rho - rho_i * drhor) * rhoinv * rhoinv;
dY_dtcoord_i = (drho_t_i * rho - rho_i * drhot) * rhoinv * rhoinv;
dY_dt_i = dY_dtcoord_i * rinv;

J_i_z = -rho * D_vec .* dY_dz_i + rho_i .* sum(D_vec .* dY_dz_i);
J_i_r = -rho * D_vec .* dY_dr_i + rho_i .* sum(D_vec .* dY_dr_i);
J_i_t = -rho * D_vec .* dY_dt_i + rho_i .* sum(D_vec .* dY_dt_i);

% Cylindrical Newtonian stress tensor (physical components)
% div(u) = uz_z + ur_r + ur/r + (1/r) ut_theta
beta = 0;
divu = duz_dz + dur_dr + ur * rinv + dut_dt;

tzz = mu_d * (2.0/3.0) * (2.0 * duz_dz - dur_dr - ur * rinv - dut_dt) / Re + beta * divu / Re;
trr = mu_d * (2.0/3.0) * (2.0 * dur_dr - duz_dz - ur * rinv - dut_dt) / Re + beta * divu / Re;
ttt = mu_d * (2.0/3.0) * (2.0 * (dut_dt + ur * rinv) - duz_dz - dur_dr) / Re + beta * divu / Re;

tzr = mu_d * (duz_dr + dur_dz) / Re;
tzt = mu_d * (duz_dt + dut_dz) / Re;
trt = mu_d * (dur_dt + dut_dr - ut * rinv) / Re;

% Viscous fluxes. Third column is physical theta-flux / r.
for i = 1:ns
    fv(i,1) = -J_i_z(i);
    fv(i,2) = -J_i_r(i);
    fv(i,3) = -J_i_t(i) * rinv;
end

% z-momentum
fv(ns+1,1) = tzz;
fv(ns+1,2) = tzr;
fv(ns+1,3) = tzt * rinv;

% r-momentum
fv(ns+2,1) = tzr;
fv(ns+2,2) = trr;
fv(ns+2,3) = trt * rinv;

% theta-momentum
fv(ns+3,1) = tzt;
fv(ns+3,2) = trt;
fv(ns+3,3) = ttt * rinv;

% energy
fv(ns+4,1) = uz * tzz + ur * tzr + ut * tzt ...
           - (sum(h_vec .* J_i_z) - kappa * dT_dz / (Re * Pr * Ec));

fv(ns+4,2) = uz * tzr + ur * trr + ut * trt ...
           - (sum(h_vec .* J_i_r) - kappa * dT_dr / (Re * Pr * Ec));

qheat_t = sum(h_vec .* J_i_t) - kappa * dT_dt / (Re * Pr * Ec);
fv(ns+4,3) = (uz * tzt + ur * trt + ut * ttt - qheat_t) * rinv;

f = fi - fv;

end
