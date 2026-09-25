function pde = pdemodel
%PDEMODEL_AXIAL Calorically perfect ideal-gas axial Navier--Stokes.
%
% Conservative variables:
%   nd=1: [rho, rho*u_z, rho*E]
%   nd=2: [rho, rho*u_z, rho*u_r, rho*E], x(1)=z, x(2)=r
%   nd=3: [rho, rho*u_z, rho*u_r, rho*u_theta, rho*E],
%         x(1)=z, x(2)=r, x(3)=theta
%
% Parameter convention:
%   mu(1)  rho_ref       density scale
%   mu(2)  u_ref         velocity scale
%   mu(3)  rhoe_ref      pressure / energy-density scale
%   mu(4)  T_ref         temperature scale
%   mu(5)  mu_ref        viscosity scale
%   mu(6)  kappa_ref     thermal-conductivity scale
%   mu(7)  R_gas         dimensional gas constant
%   mu(8)  L_ref         length scale
%   mu(9)  Ec            Eckert number
%   mu(10) Pr            Prandtl number
%   mu(11) Re            Reynolds number
%   mu(12) T_wall        isothermal wall temperature [K]
%   mu(13) gamma         ratio of specific heats
%   mu(14) T_suth_ref    Sutherland reference temperature [K]
%   mu(15) S_suth        Sutherland constant [K]
%   mu(16) mu_suth_ref   Sutherland reference viscosity [Pa s]
%   mu(17) p_out         optional nondimensional outlet static pressure
%
% Boundary columns:
%   1 supersonic inflow, eta(1:ncu)
%   2 supersonic outflow / extrapolation
%   3 isothermal no-slip wall
%   4 adiabatic no-slip wall
%   5 symmetry / inviscid slip, zero normal heat flux
%   6 zero normal gradient for all conservative variables
%   7 characteristic inflow, eta(1:ncu)
%   8 characteristic pressure outflow, mu(17) when present

pde.mass = @mass;
pde.flux = @flux;
pde.source = @source;
pde.fbou = @fbou;
pde.fbouhdg = @fbouhdg;
pde.ubou = @ubou;
pde.initu = @initu;
end

function m = mass(u, q, w, v, x, t, mu, eta) %#ok<INUSD>
ncu = numel(x) + 2;
m = ones(ncu, 1);
end

function f = flux(u, q, w, v, x, t, mu, eta) %#ok<INUSD>
f = cnsig_flux_axial(u, q, v, x, mu);
end

function s = source(u, q, w, v, x, t, mu, eta) %#ok<INUSD>
s = cnsig_source_axial(u, q, v, x, mu);
end

function fb = fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau) %#ok<INUSD>
[ub, qn] = cnsig_boundary_states(u, q, x, mu, eta, n);
fb = cnsig_fbou_from_states(ub, q, v, x, mu, n, tau, uhat, qn, true);
end

function fb = fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau) %#ok<INUSD>
[ub, qn] = cnsig_boundary_states(u, q, x, mu, eta, n);
fb = cnsig_fbouhdg_from_states(ub, q, v, x, mu, n, tau, uhat, qn, true);
end

function ub = ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau) %#ok<INUSD>
[ub, ~] = cnsig_boundary_states(u, q, x, mu, eta, n);
end

function u0 = initu(x, mu, eta) %#ok<INUSD>
nd = numel(x);
ncu = nd + 2;
if numel(eta) >= ncu
    u0 = eta(1:ncu);
else
    gamma = mu(13);
    rho = 1.0;
    vel = zeros(nd, 1);
    p = 1.0;
    rhoE = p/(gamma - 1.0);
    u0 = [rho; rho*vel; rhoE];
end
end

function f = cnsig_flux_cart(u, q, v, x, mu)
nd = numel(x);
ncu = nd + 2;
[rho, vel, rhoE, p, H, T] = cnsig_state(u, mu);
gU = cnsig_grad(q, ncu, nd);
[gvel, gT] = cnsig_grad_primitives(u, gU, mu);
[tauv, kappa_nd] = cnsig_stress_heat(vel, gvel, T, mu);

f = zeros(ncu, nd);
if ~isa(u,'sym')
    f = zeros(ncu, nd);
end

av = v(1);
for d = 1:nd
    f(1,d) = rho*vel(d) - av*gU(1,d);
    for i = 1:nd
        f(1+i,d) = rho*vel(i)*vel(d) - av*gU(1+i,d);
    end
    f(1+d,d) = f(1+d,d) + p;
    f(ncu,d) = rho*vel(d)*H - av*gU(ncu,d);
end

for d = 1:nd
    f(2:(nd+1),d) = f(2:(nd+1),d) - tauv(:,d);
    heat_contrib = 0*vel(1);
    for i = 1:nd
        heat_contrib = heat_contrib + vel(i)*tauv(i,d);
    end
    f(ncu,d) = f(ncu,d) - heat_contrib - kappa_nd*gT(d);
end
end

function [rho, vel, rhoE, p, H, T] = cnsig_state(u, mu)
nd = numel(u) - 2;
rho = u(1);
mom = u(2:(nd+1));
rhoE = u(nd+2);
vel = mom/rho;
ke = 0.0;
if ~isa(u,'sym')
    ke = 0.0;
end
for d = 1:nd
    ke = ke + 0.5*vel(d)*vel(d);
end
gamma = mu(13);
R_nd = mu(7)*mu(4)/(mu(2)*mu(2));
p = (gamma - 1.0)*(rhoE - rho*ke);
T = p/(rho*R_nd);
H = (rhoE + p)/rho;
end

function gU = cnsig_grad(q, ncu, nd)
if ismatrix(q) && ~isvector(q)
    gU = -q;
else
    gU = -reshape(q(:), [ncu, nd]);
end
end

function [gvel, gT] = cnsig_grad_primitives(u, gU, mu)
nd = numel(u) - 2;
rho = u(1);
[~, vel, rhoE, p, ~, ~] = cnsig_state(u, mu);
gamma = mu(13);
R_nd = mu(7)*mu(4)/(mu(2)*mu(2));

gvel = zeros(nd, nd);
gT = zeros(nd, 1);
if ~isa(u,'sym')
    gvel = zeros(nd, nd);
    gT = zeros(nd, 1);
end

for d = 1:nd
    grho = gU(1,d);
    for i = 1:nd
        gvel(i,d) = (gU(1+i,d) - grho*vel(i))/rho;
    end
    gke = 0*gvel(1,d);
    for i = 1:nd
        gke = gke + vel(i)*gvel(i,d);
    end
    gp = (gamma - 1.0)*(gU(nd+2,d) - grho*0.5*sum(vel.*vel) - rho*gke);
    gT(d) = (gp/rho - p*grho/(rho*rho))/R_nd;
end
end

function [tauv, kappa_nd] = cnsig_stress_heat(vel, gvel, T, mu)
nd = numel(vel);
mu_nd = cnsig_sutherland_mu(T, mu);
Pr = mu(10);
Re = mu(11);
gamma = mu(13);
R_nd = mu(7)*mu(4)/(mu(2)*mu(2));
cp_nd = gamma*R_nd/(gamma - 1.0);
kappa_nd = mu_nd*cp_nd/(Re*Pr);

divu = trace(gvel);
tauv = zeros(nd, nd);
if ~isa(vel,'sym')
    tauv = zeros(nd, nd);
end
for i = 1:nd
    for j = 1:nd
        delta = 0.0;
        if i == j
            delta = 1.0;
        end
        tauv(i,j) = (mu_nd/Re)*(gvel(i,j) + gvel(j,i) - (2.0/3.0)*divu*delta);
    end
end
end

function mu_nd = cnsig_sutherland_mu(T, mu)
T_dim = T*mu(4);
mu_dim = mu(16)*(T_dim/mu(14))^1.5*(mu(14) + mu(15))/(T_dim + mu(15));
mu_nd = mu_dim/mu(5);
end

function [ub, qn] = cnsig_boundary_states(u, q, x, mu, eta, n)
nd = numel(x);
ncu = nd + 2;
qn = cnsig_qmat(q, ncu, nd)*n(:);

u_in = eta(1:ncu);
u_out = cnsig_outlet_state(u, n, mu);
u_iso = cnsig_wall_state_temperature(u, mu(12), mu);
u_ad = u;
u_ad(2:(nd+1)) = 0;
u_sym = cnsig_remove_normal_momentum(u, n);
u_grad = u;
ub = [u_in(:), u_out(:), u_iso(:), u_ad(:), u_sym(:), u_grad(:), u_in(:), u_out(:)];
end

function uw = cnsig_wall_state_temperature(u, T_wall_dim, mu)
nd = numel(u) - 2;
rho = u(1);
R_nd = mu(7)*mu(4)/(mu(2)*mu(2));
gamma = mu(13);
T_wall = T_wall_dim/mu(4);
rhoE = rho*R_nd*T_wall/(gamma - 1.0);
uw = u;
uw(2:(nd+1)) = 0;
uw(nd+2) = rhoE;
end

function us = cnsig_remove_normal_momentum(u, n)
nd = numel(u) - 2;
mom = u(2:(nd+1));
mn = mom(:).'*n(:);
us = u;
us(2:(nd+1)) = mom - mn*n(:);
end

function qmat = cnsig_qmat(q, ncu, nd)
if ismatrix(q) && ~isvector(q)
    qmat = q;
else
    qmat = reshape(q(:), [ncu, nd]);
end
end

function fb = cnsig_fbou_from_states(ub, q, v, x, mu, n, tau, uhat, qn, axial)
ncu = size(ub, 1);
nbc = size(ub, 2);
fb = zeros(ncu, nbc);
if ~isa(ub,'sym')
    fb = zeros(ncu, nbc);
end
uint = ub(:,6);
fb(:,1) = cnsig_normal_flux(ub(:,1), q, v, x, mu, n, axial) + tau.*(ub(:,1) - uhat);
fb(:,2) = cnsig_normal_flux(uint, q, v, x, mu, n, axial) + tau.*(uint - uhat);
fb(:,3) = cnsig_normal_flux(ub(:,3), q, v, x, mu, n, axial) + tau.*(ub(:,3) - uhat);

fb(:,4) = cnsig_normal_flux(ub(:,4), q, v, x, mu, n, axial) + tau.*(ub(:,4) - uhat);
fb(1,4) = 0;
fb(ncu,4) = 0;

fb(:,5) = cnsig_symmetry_residual(ub(:,6), q, x, mu, n, tau, uhat, axial);

fb(:,6) = qn + tau.*(ub(:,6) - uhat);
ui = cnsig_characteristic_state_dg(uint, ub(:,7), n, mu);
uo = cnsig_characteristic_state_dg(uint, ub(:,8), n, mu);
fb(:,7) = cnsig_normal_flux(uint, q, v, x, mu, n, axial) + tau.*(uint - ui);
fb(:,8) = cnsig_normal_flux(uint, q, v, x, mu, n, axial) + tau.*(uint - uo);
end

function fb = cnsig_fbouhdg_from_states(ub, q, v, x, mu, n, tau, uhat, qn, axial)
ncu = size(ub, 1);
nd = numel(n);
fb = tau.*(ub - uhat);

fb(:,1) = ub(:,1) - uhat;
fb(:,2) = ub(:,6) - uhat;
fb(:,7) = cnsig_characteristic_state(ub(:,6), ub(:,7), uhat, n, mu) - uhat;
fb(:,8) = cnsig_characteristic_state(ub(:,6), ub(:,8), uhat, n, mu) - uhat;

fad = 0*uhat;
fad(1) = ub(1,4) - uhat(1);
fad(2:(nd+1)) = -uhat(2:(nd+1));
fn_hat = cnsig_normal_flux(uhat, q, v, x, mu, n, axial);
fad(ncu) = fn_hat(ncu) + cnsig_tau_entry(tau,ncu).*(ub(ncu,4) - uhat(ncu));
fb(:,4) = fad;

fb(:,5) = cnsig_symmetry_residual(ub(:,6), q, x, mu, n, tau, uhat, axial);
fb(:,6) = qn + tau.*(ub(:,6) - uhat);
end

function tauc = cnsig_tau_entry(tau, idx)
if numel(tau) == 1
    tauc = tau;
else
    tauc = tau(idx);
end
end

function ui = cnsig_characteristic_state(u, uinf, uhat, n, mu)
if numel(u) == 4
    An = cnsig_sign_matrix(uhat, n, mu);
    ui = 0.5*((u(:) + uinf(:)) + An*(u(:) - uinf(:)));
else
    ui = uinf(:);
end
end

function ui = cnsig_characteristic_state_dg(u, uinf, n, mu)
if numel(u) == 4
    An = cnsig_sign_matrix(u, n, mu);
    ui = 0.5*((u(:) + uinf(:)) + An*(u(:) - uinf(:)));
else
    ui = uinf(:);
end
end

function An = cnsig_sign_matrix(state, n, mu)
gam = mu(13);
gm1 = gam - 1.0;
nd = numel(state) - 2;
ncu = nd + 2;
rho = state(1);
mom = state(2:(nd+1));
rE = state(ncu);
nh = n(:)/sqrt(n(:).'*n(:));
Q = cnsig_normal_basis(nh);
mloc = Q*mom(:);
vloc = mloc/rho;
un = vloc(1);
ke = 0.0;
for i = 1:nd
    ke = ke + 0.5*vloc(i)*vloc(i);
end
p = gm1*(rE - rho*ke);
h = (rE + p)/rho;
a = sqrt(gam*p/rho);

K = zeros(ncu,ncu);
K(1,1) = 1.0;
K(2,1) = un - a;
for i = 2:nd
    K(1+i,1) = vloc(i);
end
K(ncu,1) = h - un*a;
K(1,2) = 1.0;
K(2,2) = un;
for i = 2:nd
    K(1+i,2) = vloc(i);
end
K(ncu,2) = ke;
for j = 1:(nd-1)
    col = 2 + j;
    K(1,col) = 0.0;
    K(2,col) = 0.0;
    for i = 2:nd
        K(1+i,col) = 0.0;
    end
    K(2+j,col) = 1.0;
    K(ncu,col) = vloc(1+j);
end
K(1,ncu) = 1.0;
K(2,ncu) = un + a;
for i = 2:nd
    K(1+i,ncu) = vloc(i);
end
K(ncu,ncu) = h + un*a;

T = zeros(ncu,ncu);
T(1,1) = 1.0;
T(2:(nd+1),2:(nd+1)) = Q;
T(ncu,ncu) = 1.0;
Ti = zeros(ncu,ncu);
Ti(1,1) = 1.0;
Ti(2:(nd+1),2:(nd+1)) = Q.';
Ti(ncu,ncu) = 1.0;

lambda = zeros(ncu,1);
lambda(1) = tanh(100*(un-a));
lambda(2) = tanh(100*un);
for j = 1:(nd-1)
    lambda(2+j) = tanh(100*un);
end
lambda(ncu) = tanh(100*(un+a));
L = diag(lambda);
An = Ti*K*L*inv(K)*T;
end

function Q = cnsig_normal_basis(nh)
nd = numel(nh);
if nd == 1
    Q = 1.0;
elseif nd == 2
    Q = [nh(1), nh(2); -nh(2), nh(1)];
else
    if abs(nh(1)) + abs(nh(2)) > 1.0e-12
        s = sqrt(nh(1)*nh(1) + nh(2)*nh(2));
        t1 = [-nh(2)/s; nh(1)/s; 0.0];
    else
        t1 = [1.0; 0.0; 0.0];
    end
    t2 = [nh(2)*t1(3)-nh(3)*t1(2); nh(3)*t1(1)-nh(1)*t1(3); nh(1)*t1(2)-nh(2)*t1(1)];
    Q = [nh(:).'; t1(:).'; t2(:).'];
end
end

function us = cnsig_outlet_state(u, n, mu)
gam = mu(13);
gm1 = gam - 1.0;
nd = numel(u) - 2;
ncu = nd + 2;
rho = u(1);
mom = u(2:(nd+1));
rE = u(ncu);
nh = n(:)/sqrt(n(:).'*n(:));
Q = cnsig_normal_basis(nh);
mloc = Q*mom(:);
vloc = mloc/rho;
un = vloc(1);
ke = 0.0;
for i = 1:nd
    ke = ke + 0.5*vloc(i)*vloc(i);
end
p = gm1*(rE - rho*ke);
pb = cnsig_outlet_pressure(p, mu);
a = sqrt(gam*p/rho);
entropyConstant = p/rho^gam;
outgoingInvariant = un + 2*a/gm1;
rb = (pb/entropyConstant)^(1/gam);
ab = sqrt(gam*pb/rb);
vlocb = vloc;
vlocb(1) = outgoingInvariant - 2*ab/gm1;
velb = Q.'*vlocb;
momb = rb*velb;
keb = 0.0;
for i = 1:nd
    keb = keb + 0.5*velb(i)*velb(i);
end
us = [rb; momb; pb/gm1 + rb*keb];
end

function pb = cnsig_outlet_pressure(p, mu)
if numel(mu) >= 17
    pb = mu(17);
else
    pb = p;
end
end

function fn = cnsig_normal_flux(u, q, v, x, mu, n, axial)
if axial
    F = cnsig_flux_axial(u, q, v, x, mu);
else
    F = cnsig_flux_cart(u, q, v, x, mu);
end
fn = F*n(:);
end

function fsym = cnsig_symmetry_residual(u, q, x, mu, n, tau, uhat, axial)
nd = numel(x);
ncu = nd + 2;
n = n(:);
nh = n/sqrt(n.'*n);
qmat = cnsig_qmat(q, ncu, nd);

mom_hat = uhat(2:(nd+1));
mn = mom_hat(:).'*nh;
dmom_dn_q = qmat(2:(nd+1),:)*nh;
Pt = eye(nd) - nh*nh.';
dtmom_dn_q = Pt*dmom_dn_q;

fsym = zeros(ncu, 1);
fsym(1) = qmat(1,:)*nh + cnsig_tau_entry(tau,1).*(u(1)-uhat(1));
fsym(2:(nd+1)) = mn*nh + dtmom_dn_q;
fsym(ncu) = qmat(ncu,:)*nh + cnsig_tau_entry(tau,ncu).*(u(ncu)-uhat(ncu));
end

function f = cnsig_flux_axial(u, q, v, x, mu)
nd = numel(x);
if nd == 1
    f = cnsig_flux_cart(u, q, v, x, mu);
    return;
end

ncu = nd + 2;
r = x(2);
rinv = 1.0/r;
[rho, vel, ~, p, H, T] = cnsig_state(u, mu);
gU = cnsig_grad(q, ncu, nd);
[gvel, gT] = cnsig_grad_primitives_axial(u, gU, x, mu);
[tauv, kappa_nd] = cnsig_stress_heat_axial(vel, gvel, T, x, mu);

f = zeros(ncu, nd);
av = v(1);
for d = 1:nd
    metric = 1.0;
    avmetric = 1.0;
    if d == 3
        metric = rinv;
        avmetric = rinv*rinv;
    end
    f(1,d) = rho*vel(d)*metric - av*gU(1,d)*avmetric;
    for i = 1:nd
        f(1+i,d) = rho*vel(i)*vel(d)*metric - av*gU(1+i,d)*avmetric;
    end
    f(1+d,d) = f(1+d,d) + p*metric;
    f(ncu,d) = rho*vel(d)*H*metric - av*gU(ncu,d)*avmetric;
end

for d = 1:nd
    metric = 1.0;
    if d == 3
        metric = rinv;
    end
    f(2:(nd+1),d) = f(2:(nd+1),d) - tauv(1:nd,d)*metric;
    heat_contrib = 0*vel(1);
    for i = 1:nd
        heat_contrib = heat_contrib + vel(i)*tauv(i,d);
    end
    f(ncu,d) = f(ncu,d) - (heat_contrib + kappa_nd*gT(d))*metric;
end
end

function s = cnsig_source_axial(u, q, v, x, mu)
nd = numel(x);
ncu = nd + 2;
s = zeros(ncu, 1);
if nd == 1
    return;
end

r = x(2);
rinv = 1.0/r;
[f, p, ttt, trt] = cnsig_flux_axial_with_stress(u, q, v, x, mu);
s = s - f(:,2)*rinv;
if nd == 2
    s(3) = s(3) + (p - ttt)*rinv;
else
    rho = u(1);
    ur = u(3)/rho;
    ut = u(4)/rho;
    s(3) = s(3) + (p + rho*ut*ut - ttt)*rinv;
    s(4) = s(4) - (rho*ur*ut - trt)*rinv;
end
end

function [f, p, ttt, trt] = cnsig_flux_axial_with_stress(u, q, v, x, mu)
f = cnsig_flux_axial(u, q, v, x, mu);
[~, vel, ~, p, ~, T] = cnsig_state(u, mu);
nd = numel(x);
ncu = nd + 2;
gU = cnsig_grad(q, ncu, nd);
[gvel, ~] = cnsig_grad_primitives_axial(u, gU, x, mu);
[tauv, ~] = cnsig_stress_heat_axial(vel, gvel, T, x, mu);
ttt = tauv(3,3);
if nd == 2
    trt = 0*ttt;
else
    trt = tauv(2,3);
end
end

function [gvel, gT] = cnsig_grad_primitives_axial(u, gU, x, mu)
nd = numel(x);
rho = u(1);
[~, vel, ~, p, ~, ~] = cnsig_state(u, mu);
gamma = mu(13);
R_nd = mu(7)*mu(4)/(mu(2)*mu(2));

nphys = nd;
if nd == 2
    nphys = 3;
end
gvel = zeros(nphys, nphys);
gT = zeros(nd, 1);

for d = 1:nd
    grho = gU(1,d);
    for i = 1:nd
        gvel(i,d) = (gU(1+i,d) - grho*vel(i))/rho;
    end
    gke = 0*gvel(1,d);
    for i = 1:nd
        gke = gke + vel(i)*gvel(i,d);
    end
    gp = (gamma - 1.0)*(gU(nd+2,d) - grho*0.5*sum(vel.*vel) - rho*gke);
    gT(d) = (gp/rho - p*grho/(rho*rho))/R_nd;
end

r = x(2);
rinv = 1.0/r;
ur = vel(2);
if nd == 2
    gvel(3,3) = ur*rinv;
else
    ut = vel(3);
    gvel(1,3) = gvel(1,3)*rinv;
    gvel(2,3) = gvel(2,3)*rinv - ut*rinv;
    gvel(3,3) = gvel(3,3)*rinv + ur*rinv;
    gT(3) = gT(3)*rinv;
end
end

function [tauv, kappa_nd] = cnsig_stress_heat_axial(vel, gvel, T, x, mu) %#ok<INUSD>
nphys = size(gvel, 1);
mu_nd = cnsig_sutherland_mu(T, mu);
Pr = mu(10);
Re = mu(11);
gamma = mu(13);
R_nd = mu(7)*mu(4)/(mu(2)*mu(2));
cp_nd = gamma*R_nd/(gamma - 1.0);
kappa_nd = mu_nd*cp_nd/(Re*Pr);

divu = trace(gvel);
tauv = zeros(nphys, nphys);
for i = 1:nphys
    for j = 1:nphys
        delta = 0.0;
        if i == j
            delta = 1.0;
        end
        tauv(i,j) = (mu_nd/Re)*(gvel(i,j) + gvel(j,i) - (2.0/3.0)*divu*delta);
    end
end
end
