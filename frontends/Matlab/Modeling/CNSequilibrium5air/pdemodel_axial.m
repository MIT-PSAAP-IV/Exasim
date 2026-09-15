function pde = pdemodel_axial
%PDEMODEL_AXIAL Axisymmetric equilibrium five-species-air CNS model.
%
% The model supports the same conservative state order as pdemodel.m.  In
% axial mode, nd=1 falls back to the Cartesian 1-D equations, nd=2 denotes
% axisymmetric (z,r) flow with no swirl, and nd=3 denotes axisymmetric
% (z,r,theta) flow with swirl.  The conservative variables are
%
%   nd=1: [rho, rho*u, rho*E]
%   nd=2: [rho, rho*u, rho*v, rho*E]
%   nd=3: [rho, rho*u, rho*v, rho*w, rho*E]
%
% Species mass fractions (N2, O2, NO, N, O) are not transported.  The
% thermochemical state is assumed to be in local chemical equilibrium and is
% supplied by an Exasim material database with state coordinates
%
%   xi = log(rho_dimensional),  e_dimensional.
%
% The database property order is
%
%   w(1)  p              [Pa]
%   w(2)  T              [K]
%   w(3)  mu             [Pa s]
%   w(4)  kappa_equi     [W/(m K)] frozen/equilibrium-composition part
%   w(5)  kappa_chem     [W/(m K)] reactive/chemical part
%   w(6)  a_equi         [m/s]
%   w(7)  p_xi           [(Pa) per log(rho)]
%   w(8)  p_e            [Pa/(J/kg)]
%   w(9)  T_xi           [K per log(rho)]
%   w(10) T_e            [K/(J/kg)]
%   w(11:15) Y_N2, Y_O2, Y_NO, Y_N, Y_O.
%
% Here p_xi and T_xi are derivatives with respect to xi=log(rho), not rho.
% Whenever a rho derivative is needed, the model uses
%
%   p_rho = p_xi/rho_dimensional,  T_rho = T_xi/rho_dimensional.
%
% Spatial gradients use the native database coordinates:
%
%   grad(xi) = grad(rho_nd)/rho_nd,
%   grad(T)  = T_xi*grad(xi) + T_e*grad(e_dimensional).
%
% Physics parameter convention:
%   mu(1)  rho_ref       [kg/m^3]
%   mu(2)  u_ref         [m/s]
%   mu(3)  p_ref         [Pa]
%   mu(4)  e_ref         [J/kg]
%   mu(5)  L_ref         [m]
%   mu(6)  transportFactor multiplying database mu/kappa
%   mu(7)  c in kappa = kappa_equi + c*kappa_chem
%   mu(8)  T_wall        [K] for isothermal wall
%   mu(9)  p_out         [Pa] for characteristic pressure outflow
%   mu(10) outlet pressure relaxation factor theta
%
% Boundary columns:
%   1 supersonic inflow, eta(1:ncu)
%   2 supersonic outflow / extrapolation
%   3 isothermal no-slip wall, linearized with T_e about interior state
%   4 adiabatic no-slip wall
%   5 symmetry / slip in conservative variables
%   6 zero normal gradient for all conservative variables
%   7 characteristic inflow, eta(1:ncu)
%   8 characteristic pressure outflow, mu(9) when present

pde.mass = @mass;
pde.flux = @flux;
pde.source = @source;
pde.fbou = @fbou;
pde.fbouhdg = @fbouhdg;
pde.ubou = @ubou;
pde.initu = @initu;
pde.initw = @initw;
pde.materialstate = @materialstate;
end

function m = mass(u, q, w, v, x, t, mu, eta) %#ok<INUSD>
ncu = numel(x) + 2;
m = ones(ncu, 1);
end

function state = materialstate(u, q, w, v, x, t, mu, eta) %#ok<INUSD>
rho = u(1);
nd = numel(u) - 2;
vel2 = 0.0;
if ~isa(u,'sym')
    vel2 = 0.0;
end
for d = 1:nd
    vd = u(1+d)/rho;
    vel2 = vel2 + vd*vd;
end
e = u(nd+2)/rho - 0.5*vel2;
rho_dim = mu(1)*rho;
e_dim = mu(4)*e;
state = [log(rho_dim); e_dim];
end

function f = flux(u, q, w, v, x, t, mu, eta) %#ok<INUSD>
f = cnseq_flux_axial(u, q, w, v, x, mu);
end

function s = source(u, q, w, v, x, t, mu, eta) %#ok<INUSD>
s = cnseq_source_axial(u, q, w, v, x, mu);
end

function fb = fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau) %#ok<INUSD>
[ub, qn] = cnseq_boundary_states(u, q, w, x, mu, eta, n);
fb = cnseq_fbou_from_states(ub, q, w, v, x, mu, n, tau, uhat, qn);
end

function fb = fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau) %#ok<INUSD>
[ub, qn] = cnseq_boundary_states(u, q, w, x, mu, eta, n);
fb = cnseq_fbouhdg_from_states(ub, q, w, v, x, mu, n, tau, uhat, qn);
end

function ub = ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau) %#ok<INUSD>
[ub, ~] = cnseq_boundary_states(u, q, w, x, mu, eta, n);
end

function u0 = initu(x, mu, eta) %#ok<INUSD>
nd = numel(x);
ncu = nd + 2;
if numel(eta) >= ncu
    u0 = eta(1:ncu);
else
    rho = 1.0;
    vel = zeros(nd, 1);
    e = 1.0;
    u0 = [rho; rho*vel; rho*(e + 0.5*(vel.'*vel))];
end
end

function w0 = initw(x, mu, eta) %#ok<INUSD>
% Size declaration for code generation.  Runtime values are supplied by the
% material database interpolation.
w0 = zeros(15, 1);
w0(1) = 1.0;      % p
w0(2) = 300.0;    % T
w0(3) = 1.0e-5;   % mu
w0(4) = 1.0e-2;   % kappa_equi
w0(5) = 0.0;      % kappa_chem
w0(6) = 300.0;    % a_equi
w0(7) = 1.0;      % p_xi
w0(8) = 1.0e-3;   % p_e
w0(9) = 0.0;      % T_xi
w0(10) = 1.0e-3;  % T_e
w0(11) = 0.767;
w0(12) = 0.233;
end

function f = cnseq_flux_cart(u, q, w, v, x, mu)
nd = numel(x);
ncu = nd + 2;
[rho, vel, rhoE, p, H] = cnseq_state(u, w, mu);
gU = cnseq_grad(q, ncu, nd);
[gvel, gT_dim] = cnseq_grad_primitives(u, gU, w, mu);
[tauv, kappa_scale] = cnseq_stress_heat(vel, gvel, w, mu);

f = cnseq_zeros(ncu, nd, u);

av = 0.0;
if numel(v) >= 1
    av = v(1);
end

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
    f(ncu,d) = f(ncu,d) - heat_contrib - kappa_scale*gT_dim(d);
end
end

function [rho, vel, rhoE, p, H] = cnseq_state(u, w, mu)
nd = numel(u) - 2;
rho = u(1);
mom = u(2:(nd+1));
rhoE = u(nd+2);
vel = mom/rho;
p = w(1)/mu(3);
H = (rhoE + p)/rho;
end

function gU = cnseq_grad(q, ncu, nd)
if ismatrix(q) && ~isvector(q)
    gU = -q;
else
    gU = -reshape(q(:), [ncu, nd]);
end
end

function [gvel, gT_dim] = cnseq_grad_primitives(u, gU, w, mu)
nd = numel(u) - 2;
rho = u(1);
rhoE = u(nd+2);
vel = u(2:(nd+1))/rho;
E = rhoE/rho;

gvel = cnseq_zeros(nd, nd, u);
gT_dim = cnseq_zeros(nd, 1, u);

T_xi = w(9);
T_e = w(10);
eRef = mu(4);
for d = 1:nd
    grho = gU(1,d);
    for i = 1:nd
        gvel(i,d) = (gU(1+i,d) - grho*vel(i))/rho;
    end
    gE = (gU(nd+2,d) - grho*E)/rho;
    ge = gE;
    for i = 1:nd
        ge = ge - vel(i)*gvel(i,d);
    end
    gxi = grho/rho;
    gT_dim(d) = T_xi*gxi + T_e*eRef*ge;
end
end

function [tauv, kappa_scale] = cnseq_stress_heat(vel, gvel, w, mu)
nd = numel(vel);
rhoRef = mu(1);
uRef = mu(2);
LRef = mu(5);
transportFactor = mu(6);
cChem = mu(7);
muPhys = transportFactor*w(3);
kappaPhys = transportFactor*(w(4) + cChem*w(5));
muScale = muPhys/(rhoRef*uRef*LRef);
kappa_scale = kappaPhys/(rhoRef*uRef^3*LRef);

divu = trace(gvel);
tauv = cnseq_zeros(nd, nd, vel);
for i = 1:nd
    for j = 1:nd
        delta = 0.0;
        if i == j
            delta = 1.0;
        end
        tauv(i,j) = muScale*(gvel(i,j) + gvel(j,i) - (2.0/3.0)*divu*delta);
    end
end
end

function [ub, qn] = cnseq_boundary_states(u, q, w, x, mu, eta, n)
nd = numel(x);
ncu = nd + 2;
qn = cnseq_qmat(q, ncu, nd)*n(:);

u_in = eta(1:ncu);
u_out = cnseq_outlet_state(u, w, n, mu);
u_iso = cnseq_wall_state_temperature(u, w, mu);
u_ad = u;
u_ad(2:(nd+1)) = 0;
u_sym = cnseq_remove_normal_momentum(u, n);
u_grad = u;
ub = [u_in(:), u_out(:), u_iso(:), u_ad(:), u_sym(:), u_grad(:), u_in(:), u_out(:)];
end

function uw = cnseq_wall_state_temperature(u, w, mu)
nd = numel(u) - 2;
rho = u(1);
vel = u(2:(nd+1))/rho;
rhoE = u(nd+2);
e = rhoE/rho - 0.5*(vel.'*vel);
Twall = mu(8);
eWall = e + (Twall - w(2))/(w(10)*mu(4));
uw = u;
uw(2:(nd+1)) = 0;
uw(nd+2) = rho*(eWall);
end

function us = cnseq_remove_normal_momentum(u, n)
nd = numel(u) - 2;
mom = u(2:(nd+1));
nh = n(:)/sqrt(n(:).'*n(:));
mn = mom(:).'*nh;
us = u;
us(2:(nd+1)) = mom - mn*nh;
end

function qmat = cnseq_qmat(q, ncu, nd)
if ismatrix(q) && ~isvector(q)
    qmat = q;
else
    qmat = reshape(q(:), [ncu, nd]);
end
end

function fb = cnseq_fbou_from_states(ub, q, w, v, x, mu, n, tau, uhat, qn)
ncu = size(ub, 1);
nbc = size(ub, 2);
fb = cnseq_zeros(ncu, nbc, ub);
uint = ub(:,6);
fb(:,1) = cnseq_normal_flux(ub(:,1), q, w, v, x, mu, n) + tau.*(ub(:,1) - uhat);
fb(:,2) = cnseq_normal_flux(uint, q, w, v, x, mu, n) + tau.*(uint - uhat);
fb(:,3) = cnseq_normal_flux(ub(:,3), q, w, v, x, mu, n) + tau.*(ub(:,3) - uhat);

fb(:,4) = cnseq_normal_flux(ub(:,4), q, w, v, x, mu, n) + tau.*(ub(:,4) - uhat);
fb(1,4) = 0;
fb(ncu,4) = 0;

fb(:,5) = cnseq_symmetry_residual(ub(:,6), q, x, n, tau, uhat);
fb(:,6) = qn + tau.*(ub(:,6) - uhat);

ui = cnseq_characteristic_state_dg(uint, ub(:,7), n, w, mu);
uo = cnseq_characteristic_state_dg(uint, ub(:,8), n, w, mu);
fb(:,7) = cnseq_normal_flux(uint, q, w, v, x, mu, n) + tau.*(uint - ui);
fb(:,8) = cnseq_normal_flux(uint, q, w, v, x, mu, n) + tau.*(uint - uo);
end

function fb = cnseq_fbouhdg_from_states(ub, q, w, v, x, mu, n, tau, uhat, qn)
ncu = size(ub, 1);
nd = numel(n);
fb = tau.*(ub - uhat);

fb(:,1) = ub(:,1) - uhat;
fb(:,2) = ub(:,6) - uhat;
fb(:,7) = cnseq_characteristic_state(ub(:,6), ub(:,7), uhat, n, w, mu) - uhat;
fb(:,8) = cnseq_characteristic_state(ub(:,6), ub(:,8), uhat, n, w, mu) - uhat;

fad = 0*uhat;
fad(1) = ub(1,4) - uhat(1);
fad(2:(nd+1)) = -uhat(2:(nd+1));
fn_hat = cnseq_normal_flux(uhat, q, w, v, x, mu, n);
fad(ncu) = fn_hat(ncu) + cnseq_tau_entry(tau,ncu).*(ub(ncu,4) - uhat(ncu));
fb(:,4) = fad;

fb(:,5) = cnseq_symmetry_residual(ub(:,6), q, x, n, tau, uhat);
fb(:,6) = qn + tau.*(ub(:,6) - uhat);
end

function tauc = cnseq_tau_entry(tau, idx)
if numel(tau) == 1
    tauc = tau;
else
    tauc = tau(idx);
end
end

function ui = cnseq_characteristic_state(u, uinf, uhat, n, w, mu)
An = cnseq_sign_matrix(uhat, n, w, mu);
ui = 0.5*((u(:) + uinf(:)) + An*(u(:) - uinf(:)));
end

function ui = cnseq_characteristic_state_dg(u, uinf, n, w, mu)
An = cnseq_sign_matrix(u, n, w, mu);
ui = 0.5*((u(:) + uinf(:)) + An*(u(:) - uinf(:)));
end

function An = cnseq_sign_matrix(state, n, w, mu)
nd = numel(state) - 2;
ncu = nd + 2;
rho = state(1);
mom = state(2:(nd+1));
rhoE = state(ncu);
nh = n(:)/sqrt(n(:).'*n(:));
Q = cnseq_normal_basis(nh);
mloc = Q*mom(:);
vloc = mloc/rho;
un = vloc(1);
E = rhoE/rho;
ke = 0.0;
for i = 1:nd
    ke = ke + 0.5*vloc(i)*vloc(i);
end

a = w(6)/mu(2);
p_xi = w(7);
p_e = w(8);
pRef = mu(3);
eRef = mu(4);
p_rho_nd = p_xi/(rho*pRef);
p_e_nd = eRef*p_e/pRef;
B = E + rho*(a*a - p_rho_nd)/p_e_nd;
C = E - rho*p_rho_nd/p_e_nd;

K = cnseq_zeros(ncu, ncu, state);
K(1,1) = 1.0;
K(2,1) = un - a;
for i = 2:nd
    K(1+i,1) = vloc(i);
end
K(ncu,1) = B - un*a;

K(1,2) = 1.0;
K(2,2) = un;
for i = 2:nd
    K(1+i,2) = vloc(i);
end
K(ncu,2) = C;

for j = 1:(nd-1)
    col = 2 + j;
    K(2+j,col) = 1.0;
    K(ncu,col) = vloc(1+j);
end

K(1,ncu) = 1.0;
K(2,ncu) = un + a;
for i = 2:nd
    K(1+i,ncu) = vloc(i);
end
K(ncu,ncu) = B + un*a;

T = cnseq_zeros(ncu, ncu, state);
T(1,1) = 1.0;
T(2:(nd+1),2:(nd+1)) = Q;
T(ncu,ncu) = 1.0;
Ti = cnseq_zeros(ncu, ncu, state);
Ti(1,1) = 1.0;
Ti(2:(nd+1),2:(nd+1)) = Q.';
Ti(ncu,ncu) = 1.0;

lambda = cnseq_zeros(ncu, 1, state);
lambda(1) = tanh(100*(un-a));
lambda(2) = tanh(100*un);
for j = 1:(nd-1)
    lambda(2+j) = tanh(100*un);
end
lambda(ncu) = tanh(100*(un+a));
An = Ti*K*diag(lambda)*inv(K)*T;
end

function Q = cnseq_normal_basis(nh)
nd = numel(nh);
if nd == 1
    Q = 1.0;
elseif nd == 2
    Q = [nh(1), nh(2); -nh(2), nh(1)];
else
    % Algebraic orthonormal basis for symbolic-code-generation
    % compatibility.  This form is nonsingular except at nh=[0;0;-1],
    % where a different tangent chart is required.
    denom = 1.0 + nh(3);
    t1 = [1.0 - nh(1)*nh(1)/denom; -nh(1)*nh(2)/denom; -nh(1)];
    t2 = [-nh(1)*nh(2)/denom; 1.0 - nh(2)*nh(2)/denom; -nh(2)];
    Q = [nh(:).'; t1(:).'; t2(:).'];
end
end

function uo = cnseq_outlet_state(u, w, n, mu)
nd = numel(u) - 2;
ncu = nd + 2;
rho = u(1);
mom = u(2:(nd+1));
rhoE = u(ncu);
nh = n(:)/sqrt(n(:).'*n(:));
Q = cnseq_normal_basis(nh);
mloc = Q*mom(:);
vloc = mloc/rho;
un = vloc(1);
E = rhoE/rho;

a = w(6)/mu(2);
p_xi = w(7);
p_e = w(8);
pRef = mu(3);
eRef = mu(4);
p_rho_nd = p_xi/(rho*pRef);
p_e_nd = eRef*p_e/pRef;
B = E + rho*(a*a - p_rho_nd)/p_e_nd;

pb = cnseq_outlet_pressure(w(1), mu);
theta = cnseq_outlet_relaxation(mu);
dp = theta*(pb - w(1))/pRef;
dr = dp/(a*a);

rminus = cnseq_zeros(ncu, 1, u);
rminus(1) = 1.0;
rminus(2) = un - a;
for i = 2:nd
    rminus(1+i) = vloc(i);
end
rminus(ncu) = B - un*a;

Tinv = cnseq_zeros(ncu, ncu, u);
Tinv(1,1) = 1.0;
Tinv(2:(nd+1),2:(nd+1)) = Q.';
Tinv(ncu,ncu) = 1.0;
uo = u(:) + dr*(Tinv*rminus);
end

function pb = cnseq_outlet_pressure(pInterior, mu)
if numel(mu) >= 9
    pb = mu(9);
else
    pb = pInterior;
end
end

function theta = cnseq_outlet_relaxation(mu)
if numel(mu) >= 10
    theta = mu(10);
else
    theta = 1.0;
end
end

function fn = cnseq_normal_flux(u, q, w, v, x, mu, n)
F = cnseq_flux_axial(u, q, w, v, x, mu);
fn = F*n(:);
end

function fsym = cnseq_symmetry_residual(u, q, x, n, tau, uhat)
nd = numel(x);
ncu = nd + 2;
nh = n(:)/sqrt(n(:).'*n(:));
qmat = cnseq_qmat(q, ncu, nd);

mom_hat = uhat(2:(nd+1));
mn = mom_hat(:).'*nh;
dmom_dn_q = qmat(2:(nd+1),:)*nh;
Pt = eye(nd) - nh*nh.';
dtmom_dn_q = Pt*dmom_dn_q;

fsym = cnseq_zeros(ncu, 1, u);
fsym(1) = qmat(1,:)*nh + cnseq_tau_entry(tau,1).*(u(1)-uhat(1));
fsym(2:(nd+1)) = mn*nh + dtmom_dn_q;
fsym(ncu) = qmat(ncu,:)*nh + cnseq_tau_entry(tau,ncu).*(u(ncu)-uhat(ncu));
end

function f = cnseq_flux_axial(u, q, w, v, x, mu)
nd = numel(x);
if nd == 1
    f = cnseq_flux_cart(u, q, w, v, x, mu);
    return;
end

ncu = nd + 2;
r = x(2);
rinv = 1.0/r;
[rho, vel, ~, p, H] = cnseq_state(u, w, mu);
gU = cnseq_grad(q, ncu, nd);
[gvel, gT_dim] = cnseq_grad_primitives_axial(u, gU, w, x, mu);
[tauv, kappa_scale] = cnseq_stress_heat_axial(vel, gvel, w, x, mu);

f = cnseq_zeros(ncu, nd, u);

av = 0.0;
if numel(v) >= 1
    av = v(1);
end

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
    f(ncu,d) = f(ncu,d) - (heat_contrib + kappa_scale*gT_dim(d))*metric;
end
end

function s = cnseq_source_axial(u, q, w, v, x, mu)
nd = numel(x);
ncu = nd + 2;
s = cnseq_zeros(ncu, 1, u);
if nd == 1
    return;
end

r = x(2);
rinv = 1.0/r;
[f, p, ttt, trt] = cnseq_flux_axial_with_stress(u, q, w, v, x, mu);
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

function [f, p, ttt, trt] = cnseq_flux_axial_with_stress(u, q, w, v, x, mu)
f = cnseq_flux_axial(u, q, w, v, x, mu);
[~, vel, ~, p, ~] = cnseq_state(u, w, mu);
nd = numel(x);
ncu = nd + 2;
gU = cnseq_grad(q, ncu, nd);
[gvel, ~] = cnseq_grad_primitives_axial(u, gU, w, x, mu);
[tauv, ~] = cnseq_stress_heat_axial(vel, gvel, w, x, mu);
ttt = tauv(3,3);
if nd == 2
    trt = 0*ttt;
else
    trt = tauv(2,3);
end
end

function [gvel, gT_dim] = cnseq_grad_primitives_axial(u, gU, w, x, mu)
nd = numel(x);
rho = u(1);
rhoE = u(nd+2);
vel = u(2:(nd+1))/rho;
E = rhoE/rho;

nphys = nd;
if nd == 2
    nphys = 3;
end
gvel = cnseq_zeros(nphys, nphys, u);
gT_dim = cnseq_zeros(nd, 1, u);

T_xi = w(9);
T_e = w(10);
eRef = mu(4);
for d = 1:nd
    grho = gU(1,d);
    for i = 1:nd
        gvel(i,d) = (gU(1+i,d) - grho*vel(i))/rho;
    end
    gE = (gU(nd+2,d) - grho*E)/rho;
    ge = gE;
    for i = 1:nd
        ge = ge - vel(i)*gvel(i,d);
    end
    gxi = grho/rho;
    gT_dim(d) = T_xi*gxi + T_e*eRef*ge;
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
    gT_dim(3) = gT_dim(3)*rinv;
end
end

function [tauv, kappa_scale] = cnseq_stress_heat_axial(vel, gvel, w, x, mu) %#ok<INUSD>
nphys = size(gvel, 1);
rhoRef = mu(1);
uRef = mu(2);
LRef = mu(5);
transportFactor = mu(6);
cChem = mu(7);
muPhys = transportFactor*w(3);
kappaPhys = transportFactor*(w(4) + cChem*w(5));
muScale = muPhys/(rhoRef*uRef*LRef);
kappa_scale = kappaPhys/(rhoRef*uRef^3*LRef);

divu = trace(gvel);
tauv = cnseq_zeros(nphys, nphys, vel);
for i = 1:nphys
    for j = 1:nphys
        delta = 0.0;
        if i == j
            delta = 1.0;
        end
        tauv(i,j) = muScale*(gvel(i,j) + gvel(j,i) - (2.0/3.0)*divu*delta);
    end
end
end

function A = cnseq_zeros(m, n, prototype)
if isa(prototype, 'sym')
    A = sym(zeros(m, n));
else
    A = zeros(m, n);
end
end
