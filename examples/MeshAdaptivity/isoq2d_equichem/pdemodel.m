function pde = pdemodel
%PDEMODEL Axisymmetric equilibrium five-species-air Navier--Stokes model.
%
% This is the equilibrium-chemistry counterpart of
% isoq2d_idealgas/pdemodel_axialns.m. The conservative state is
% [rho, rho*uz, rho*ur, rho*E]. Thermodynamic and transport properties are
% supplied by the equilibrium material database in w(1:15).

pde.mass = @mass;
pde.flux = @flux;
pde.source = @source;
pde.fbou = @fbou;
pde.ubou = @ubou;
pde.initu = @initu;
pde.initw = @initw;
pde.materialstate = @materialstate;
pde.avfield = @avfield;
pde.fbouhdg = @fbouhdg;
pde.visscalars = @visscalars;
pde.visvectors = @visvectors;
pde.surfacequantities = @surfacequantities;
end

function m = mass(u, q, w, v, x, t, mu, eta) %#ok<INUSD>
m = ones_like(4,1,u);
end

function state = materialstate(u, q, w, v, x, t, mu, eta) %#ok<INUSD>
rho = u(1);
uz = u(2)/rho;
ur = u(3)/rho;
e = u(4)/rho - 0.5*(uz*uz + ur*ur);
rholm = limiting(rho, 0.0001/mu(1), 20.0/mu(1), 1e2, 0.0001/mu(1));
elm = limiting(e, -150000/mu(4), 20000000/mu(4), 1e2, -150000/mu(4));
state = [log(mu(1)*rholm); mu(4)*elm];
end

function f = flux(u, q, w, v, x, t, mu, eta) %#ok<INUSD>
[rho,vel,p,H,gU,tauv,kappaScale,gT] = flow_data(u,q,w,x,mu);
av = artificial_viscosity(v,mu);
f = zeros_like(4,2,u);

for d = 1:2
    f(1,d) = rho*vel(d) - av*gU(1,d);
    for i = 1:2
        f(1+i,d) = rho*vel(i)*vel(d) - av*gU(1+i,d) - tauv(i,d);
    end
    f(1+d,d) = f(1+d,d) + p;
    viscousWork = vel(1)*tauv(1,d) + vel(2)*tauv(2,d);
    f(4,d) = rho*vel(d)*H - av*gU(4,d) - viscousWork ...
             - kappaScale*gT(d);
end
end

function s = source(u, q, w, v, x, t, mu, eta)
f = flux(u,q,w,v,x,t,mu,eta);
[~,~,p,~,~,tauv] = flow_data(u,q,w,x,mu);
r = x(2);
s = -f(:,2)/r;
s(3) = s(3) + (p-tauv(3,3))/r;
end

function f = avfield(u, q, w, v, x, t, mu, eta) %#ok<INUSD>
rho = u(1);
uz = u(2)/rho;
ur = u(3)/rho;
duz_dz = (q(2)-q(1)*uz)/rho;
dur_dr = (q(7)-q(5)*ur)/rho;
divu = duz_dz + dur_dr + ur/x(2);
sensor = limiting(divu*tanh(mu(end-2)*v(1)),0,mu(end-3),1e3,0);
f = [sensor; w(1)];
end

function fb = fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau) %#ok<INUSD>
[states,qn] = boundary_states(u,q,w,mu,eta,n);
fb = zeros_like(4,6,u);
fb(:,1) = normal_flux(states(:,1),q,w,v,x,mu,n) + tau.*(states(:,1)-uhat);
fb(:,2) = normal_flux(u,q,w,v,x,mu,n) + tau.*(u-uhat);
fb(:,3) = normal_flux(states(:,3),q,w,v,x,mu,n) + tau.*(states(:,3)-uhat);
fb(:,4) = normal_flux(states(:,4),q,w,v,x,mu,n) + tau.*(states(:,4)-uhat);
fb(1,4) = 0;
fb(4,4) = 0;
fb(:,5) = symmetry_residual(u,q,n,tau,uhat);
fb(:,6) = qn + tau.*(u-uhat);
end

function ub = ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau) %#ok<INUSD>
[ub,~] = boundary_states(u,q,w,mu,eta,n);
end

function fb = fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau) %#ok<INUSD>
uinf = eta(1:4);
fout = u-uhat;
fin = uinf-uhat;

% Isothermal no-slip wall. Density is extrapolated, velocity is zero, and
% the wall energy is obtained from the equilibrium temperature derivative.
fiso = 0*u;
fiso(1) = u(1)-uhat(1);
fiso(2:3) = -uhat(2:3);
fiso(4) = uhat(1)*wall_specific_energy(u,w,mu)-uhat(4);

% Symmetry condition, written in the same primitive-gradient form as the
% ideal-gas ISOQ model but using the equilibrium temperature derivatives.
% rho = uhat(1);
% uz = uhat(2)/rho;
% ur = uhat(3)/rho;
% E = uhat(4)/rho;
% dr = q(5);
% duz = (q(6)-dr*uz)/rho;
% dur = (q(7)-dr*ur)/rho;
% dE = (q(8)-dr*E)/rho;
% de = dE-uz*duz-ur*dur;
% dT = w(9)*(dr/rho)+w(10)*mu(4)*de;
% fsym = 0*u;
% fsym(1) = dr + tau_entry(tau,1).*(u(1)-uhat(1));
% fsym(2) = duz + tau_entry(tau,2).*(u(2)-uhat(2));
% fsym(3) = dur - tau_entry(tau,3).*uhat(3);
% fsym(4) = dT + tau_entry(tau,4).*(u(4)-uhat(4));

uslip = remove_normal_momentum(u,n);
fslip = tau.*(uslip-uhat);
qmat = reshape(q(:),[4,2]);
fgrad = qmat*n(:)+tau.*(u-uhat);

fwall = 0*u;
fwall(1) = u(1)-uhat(1);
fwall(2:3) = -uhat(2:3);

% Symmetry condition
fsym = fgrad;
fsym(2:3) = fslip(2:3);

% Inflow, outflow, isothermal, symmetry, slip wall, zero gradient, no slip.
fb = [fin fout fiso fsym fslip fgrad fwall];
end

function u0 = initu(x, mu, eta) %#ok<INUSD>
u0 = eta(1:4);
end

function w0 = initw(x, mu, eta) %#ok<INUSD>
w0 = zeros_like(15,1,x);
w0(1) = 1.0;
w0(2) = 300.0;
w0(3) = 1.0e-5;
w0(4) = 1.0e-2;
w0(5) = 0.0;
w0(6) = 300.0;
w0(7) = 1.0;
w0(8) = 1.0e-3;
w0(9) = 0.0;
w0(10) = 1.0e-3;
w0(11) = 0.767;
w0(12) = 0.233;
end

function s = visscalars(u, q, w, v, x, t, mu, eta) %#ok<INUSD>
rho = u(1);
velocity = u(2:3)/rho;
speed = mu(2)*sqrt(velocity.'*velocity);
mach = speed/w(6);
% density, pressure, temperature, Mach, AV, Y_N, Y_O, Y_NO, Y_N2, Y_O2
s = [mu(1)*rho; w(1); w(2); mach; artificial_viscosity(v,mu); ...
     w(14); w(15); w(13); w(11); w(12)];
end

function s = visvectors(u, q, w, v, x, t, mu, eta) %#ok<INUSD>
s = mu(2)*u(2:3)/u(1);
end

function s = surfacequantities(u, q, w, v, x, t, mu, eta, uhat, n, tau) %#ok<INUSD>
% Wall outputs on pde.ibs:
% s(1) = Cp = (p - p_inf)/(0.5*rho_inf*|u_inf|^2)
% s(2) = Cf = t dot (tau_v n + HDG momentum penalty) /
%        (0.5*rho_inf*|u_inf|^2), with t=[-n_y,n_x].
% s(3) = Cq = (kappa grad(T) dot n + HDG energy penalty) /
%        (rho_inf*|u_inf|^3).
[~,~,p,~,~,tauv,kappaScale,gT] = flow_data(uhat,q,w,x,mu);

rhoInf = eta(1);
velInf = eta(2:3)/rhoInf;
speedInf2 = velInf(:).'*velInf(:);
qdyn = 0.5*rhoInf*speedInf2;
qheatref = rhoInf*speedInf2*sqrt(speedInf2);
pInf = mu(9)/mu(3);

normal = n(:);
tangent = [-normal(2); normal(1)];
traction = tauv(1:2,1:2)*normal;
traction(1) = traction(1) + tau_entry(tau,2).*(u(2)-uhat(2));
traction(2) = traction(2) + tau_entry(tau,3).*(u(3)-uhat(3));
heatFlux = kappaScale*(gT(:).'*normal) + tau_entry(tau,4).*(u(4)-uhat(4));

Cp = (p-pInf)/qdyn;
Cf = (tangent(:).'*traction(:))/qdyn;
Cq = heatFlux/qheatref;
s = [Cp; Cf; Cq];
end

function [rho,vel,p,H,gU,tauv,kappaScale,gT] = flow_data(u,q,w,x,mu)
rho = u(1);
vel = u(2:3)/rho;
p = w(1)/mu(3);
H = (u(4)+p)/rho;
gU = -reshape(q(:),[4,2]);

gvel = zeros_like(3,3,u);
gT = zeros_like(2,1,u);
E = u(4)/rho;
for d = 1:2
    grho = gU(1,d);
    gvel(1,d) = (gU(2,d)-grho*vel(1))/rho;
    gvel(2,d) = (gU(3,d)-grho*vel(2))/rho;
    gE = (gU(4,d)-grho*E)/rho;
    ge = gE-vel(1)*gvel(1,d)-vel(2)*gvel(2,d);
    gT(d) = w(9)*(grho/rho) + w(10)*mu(4)*ge;
end
gvel(3,3) = vel(2)/x(2);

muScale = mu(6)*w(3)/(mu(1)*mu(2)*mu(5));
kappa = mu(6)*(w(4)+mu(7)*w(5));
kappaScale = kappa/(mu(1)*mu(2)^3*mu(5));
divu = gvel(1,1)+gvel(2,2)+gvel(3,3);
tauv = zeros_like(3,3,u);
for i = 1:3
    for j = 1:3
        delta = 0.0;
        if i == j, delta = 1.0; end
        tauv(i,j) = muScale*(gvel(i,j)+gvel(j,i)-(2.0/3.0)*divu*delta);
    end
end
end

function [states,qn] = boundary_states(u,q,w,mu,eta,n)
qin = reshape(q(:),[4,2]);
qn = qin*n(:);
uIn = eta(1:4);
uIso = wall_state(u,w,mu);
uAd = u;
uAd(2:3) = 0;
uSym = remove_normal_momentum(u,n);
% inflow, outflow, isothermal, adiabatic, symmetry/slip, zero-gradient
states = [uIn(:),u(:),uIso(:),uAd(:),uSym(:),u(:)];
end

function uw = wall_state(u,w,mu)
rho = u(1);
uw = u;
uw(2:3) = 0;
uw(4) = rho*wall_specific_energy(u,w,mu);
end

function eWall = wall_specific_energy(u,w,mu)
rho = u(1);
uz = u(2)/rho;
ur = u(3)/rho;
e = u(4)/rho-0.5*(uz*uz+ur*ur);
eWall = e+(mu(8)-w(2))/(w(10)*mu(4));
end

function us = remove_normal_momentum(u,n)
nh = n(:)/sqrt(n(:).'*n(:));
mom = u(2:3);
us = u;
us(2:3) = mom-(mom.'*nh)*nh;
end

function fn = normal_flux(u,q,w,v,x,mu,n)
fn = flux(u,q,w,v,x,0,mu,0)*n(:);
end

function fs = symmetry_residual(u,q,n,tau,uhat)
nh = n(:)/sqrt(n(:).'*n(:));
qmat = reshape(q(:),[4,2]);
momHat = uhat(2:3);
normalMomentum = momHat.'*nh;
tangentProjection = eye(2)-nh*nh.';
fs = zeros_like(4,1,u);
fs(1) = qmat(1,:)*nh + tau_entry(tau,1).*(u(1)-uhat(1));
fs(2:3) = normalMomentum*nh + tangentProjection*(qmat(2:3,:)*nh);
fs(4) = qmat(4,:)*nh + tau_entry(tau,4).*(u(4)-uhat(4));
end

function value = tau_entry(tau,index)
if isscalar(tau)
    value = tau;
else
    value = tau(index);
end
end

function av = artificial_viscosity(v,mu)
av = (mu(end-1)+mu(end)*v(2))*tanh(mu(end-2)*v(1));
end

function A = zeros_like(m,n,prototype)
if isa(prototype,'sym')
    A = sym(zeros(m,n));
else
    A = zeros(m,n);
end
end

function A = ones_like(m,n,prototype)
if isa(prototype,'sym')
    A = sym(ones(m,n));
else
    A = ones(m,n);
end
end
