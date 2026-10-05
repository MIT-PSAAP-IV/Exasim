function pde = pdemodel
%PDEMODEL Axisymmetric finite-rate five-species-air model with backend AV.

pde.mass = @mass;
pde.flux = @flux;
pde.source = @source;
pde.fbou = @fbou;
pde.ubou = @ubou;
pde.fbouhdg = @fbouhdg;
pde.initu = @initu;
pde.initw = @initw;
pde.sourcew = @sourcew;
pde.eos = @eos;
pde.monitor = @monitor;
pde.avfield = @avfield;
pde.visscalars = @visscalars;
pde.visvectors = @visvectors;
pde.surfacequantities = @surfacequantities;
end

function m = mass(u, q, w, v, x, t, mu, eta) %#ok<INUSD>
m = sym(ones(8,1));
end

function f = flux(u, q, w, v, x, t, mu, eta)
f = fluxaxial2d(u,q,w,artificialviscosity(v,mu),x,t,mu,eta);
end

function s = source(u, q, w, v, x, t, mu, eta)
s = sourceaxial2d(u,q,w,artificialviscosity(v,mu),x,t,mu,eta);
end

function fb = fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
fb = fbouaxialnd(u,q,w,artificialviscosity(v,mu),x,t,mu,eta, ...
                 uhat,n,tau,eta,mu,0);
end

function ub = ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
ub = ubouaxialnd(u,q,w,artificialviscosity(v,mu),x,t,mu,eta, ...
                 uhat,n,tau,eta,mu,0);
end

function fb = fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
fb = fbouhdgaxialnd(u,q,w,artificialviscosity(v,mu),x,t,mu,eta,uhat,n,tau);

qmat = reshape(q(:),[8,2]);
fgrad = qmat*n(:)+tau.*(u-uhat);

ns = 5;
nd = 2;
irhou = ns + (1:nd);
m  = u(irhou);
mn = m.' * n;
uslip = u;
uslip(irhou) = m - mn .* n;
fslip = tau .* (uslip - uhat);

% Symmetry condition
fsym = fslip;
fsym(end) = fgrad(end);
fb = [fb fsym];

end

function u0 = initu(x, mu, eta) %#ok<INUSD>
u0 = sym(ones(8,1));
end

function w0 = initw(x, mu, eta) %#ok<INUSD>
w0 = sym(1.0);
end

function f = eos(u, q, w, v, x, t, mu, eta)
f = eosnd(u,q,w,v,x,t,mu,eta);
end

function f = sourcew(u, q, w, v, x, t, mu, eta)
f = eosnd(u,q,w,v,x,t,mu,eta);
end

function m = monitor(u, q, w, v, x, t, mu, eta) %#ok<INUSD>
rhoSpecies = mu(1)*u(1:5);
rho = sum(rhoSpecies);
temperature = mu(4)*w(1);
[~,Mw,~] = thermodynamicsModels();
pressurePhysical = pressure(temperature,rhoSpecies,Mw);

% Positive, dimensionless margins.  A continuation state is accepted only
% when every component is positive and finite at every owned DG node.
m = [(rhoSpecies-mu(13))/mu(1); ...
     (rho-mu(14))/mu(1); ...
     (temperature-mu(15))/mu(4); ...
     (mu(16)-temperature)/mu(4); ...
     (pressurePhysical-mu(17))/mu(3)];
end

function f = avfield(u, q, w, v, x, t, mu, eta) %#ok<INUSD>
ns = 5;
ncu = 8;
rho = sum(u(1:ns));
uz = u(ns+1)/rho;
ur = u(ns+2)/rho;
qrhoZ = sum(q(1:ns));
qrhoR = sum(q(ncu+(1:ns)));
radius = 1.0e-8+lmax(x(2)-1.0e-8,1.0e3);

% q=-grad(u), so these two terms are -duz/dz and -dur/dr.
compression = (q(ns+1)-qrhoZ*uz)/rho ...
            + (q(ncu+ns+2)-qrhoR*ur)/rho ...
            - ur/radius;
sensor = limiting(compression*tanh(mu(end-2)*v(1)), ...
                  0,mu(end-3),1.0e3,0);
rhoSpecies = mu(1)*u(1:ns);
temperature = mu(4)*w(1);
[~,Mw,~] = thermodynamicsModels();
pressurePhysical = pressure(temperature,rhoSpecies,Mw);
f = [sensor; pressurePhysical];
end

function s = visscalars(u, q, w, v, x, t, mu, eta) %#ok<INUSD>
rhoSpecies = mu(1)*u(1:5);
rho = sum(rhoSpecies);
temperature = mu(4)*w(1);
[~,Mw,~] = thermodynamicsModels();
pressurePhysical = pressure(temperature,rhoSpecies,Mw);
velocity = mu(2)*u(6:7)/sum(u(1:5));
soundSpeed = soundspeed(temperature,abs(rhoSpecies));
mach = sqrt(velocity.'*velocity)/soundSpeed;
massFractions = rhoSpecies/rho;

% Pressure remains first for visualization output.
% Remaining fields are rho, T, Mach, Y_N, Y_O, Y_NO, Y_N2, and Y_O2.
s = [pressurePhysical;rho;temperature;mach;massFractions];
end

function velocity = visvectors(u, q, w, v, x, t, mu, eta) %#ok<INUSD>
velocity = mu(2)*u(6:7)/sum(u(1:5));
end

function s = surfacequantities(u, q, w, v, x, t, mu, eta, uhat, n, tau) %#ok<INUSD>
% Pointwise catalytic-wall outputs evaluated by the backend on pde.ibs.
%
% s(1) = Cp = (p - p_inf)/(0.5*rho_inf*|u_inf|^2).
% s(2) = Cf = t dot (tau_v n + HDG momentum penalty) /
%        (0.5*rho_inf*|u_inf|^2), with t=[-n_r,n_z].
% s(3) = Cq = (kappa grad(T) dot n + HDG energy penalty) /
%        (rho_inf*|u_inf|^3). Positive Cq follows the saved outward
%        boundary normal and is consistent with the heat-flux sign used in
%        fluxaxial2d.

ns = 5;
ncu = 8;
[~,Mw,~] = thermodynamicsModels();

rhoInf = sum(eta(1:ns));
uzInf = eta(ns+1)/rhoInf;
urInf = eta(ns+2)/rhoInf;
speedInf2 = uzInf*uzInf + urInf*urInf;
qdyn = 0.5*rhoInf*speedInf2;
qheatref = rhoInf*speedInf2*sqrt(speedInf2);
pInf = pressure(mu(4),mu(1)*eta(1:ns),Mw)/mu(3);

T_dim = mu(4)*w(1);
p = pressure(T_dim,mu(1)*uhat(1:ns),Mw)/mu(3);

rho = sum(uhat(1:ns));
rhoinv = 1.0/rho;
uz = uhat(ns+1)*rhoinv;
ur = uhat(ns+2)*rhoinv;
rho_i_dim = mu(1)*uhat(1:ns);

% Exasim convention for this model is q=-grad(U).  The first block is the
% axial z derivative and the second block is the radial r derivative.
drho_dz_i = -q(1:ns);
drhou_dz  = -q(ns+1);
drhov_dz  = -q(ns+2);
drhoE_dz  = -q(ns+3);

drho_dr_i = -q((ncu+1):(ncu+ns));
drhou_dr  = -q(ncu+ns+1);
drhov_dr  = -q(ncu+ns+2);
drhoE_dr  = -q(ncu+ns+3);

drho_dz = sum(drho_dz_i);
drho_dr = sum(drho_dr_i);

duz_dz = (drhou_dz - drho_dz*uz)*rhoinv;
dur_dz = (drhov_dz - drho_dz*ur)*rhoinv;
duz_dr = (drhou_dr - drho_dr*uz)*rhoinv;
dur_dr = (drhov_dr - drho_dr*ur)*rhoinv;

kinetic = 0.5*(uz*uz + ur*ur);
drhoe_dz = drhoE_dz - (uz*drhou_dz + ur*drhov_dz) + kinetic*drho_dz;
drhoe_dr = drhoE_dr - (uz*drhou_dr + ur*drhov_dr) + kinetic*drho_dr;

[dT_drho_i_dim,dT_drhoe_dim,~,~,mu_d_dim,kappa_dim] = ...
    transportcoefficients(T_dim,rho_i_dim);
dT_drho_i = dT_drho_i_dim*mu(1)/mu(4);
dT_drhoe = dT_drhoe_dim*mu(3)/mu(4);
dT_dz = sum(dT_drho_i.*drho_dz_i) + dT_drhoe*drhoe_dz;
dT_dr = sum(dT_drho_i.*drho_dr_i) + dT_drhoe*drhoe_dr;

mu_d = mu_d_dim/mu(5);
kappa = kappa_dim/mu(6);
Re = mu(11);
Pr = mu(10);
Ec = mu(9);

radius = x(2);
rinv = 1.0/radius;

% Axisymmetric Newtonian stresses from fluxaxial2d.
tzz = mu_d*(2.0/3.0)*(2.0*duz_dz - dur_dr - ur*rinv)/Re;
tzr = mu_d*(duz_dr + dur_dz)/Re;
trr = mu_d*(2.0/3.0)*(2.0*dur_dr - duz_dz - ur*rinv)/Re;

nz = n(1);
nr = n(2);
tz = -nr;
tr = nz;
tractionz = tzz*nz + tzr*nr + tau_entry(tau,ns+1)*(u(ns+1)-uhat(ns+1));
tractionr = tzr*nz + trr*nr + tau_entry(tau,ns+2)*(u(ns+2)-uhat(ns+2));
tauTangential = tz*tractionz + tr*tractionr;

conductiveWallFlux = kappa*(dT_dz*nz + dT_dr*nr)/(Re*Pr*Ec) ...
    + tau_entry(tau,ncu)*(u(ncu)-uhat(ncu));

Cp = (p - pInf)/qdyn;
Cf = tauTangential/qdyn;
Cq = conductiveWallFlux/qheatref;
s = [Cp; Cf; Cq];
end

function va = artificialviscosity(v,mu)
av = physicalav(v,mu);
va = [av;av];
end

function av = physicalav(v,mu)
av = (mu(end-1)+mu(end)*v(2))*tanh(mu(end-2)*v(1));
end

function tauc = tau_entry(tau,idx)
if numel(tau) == 1
    tauc = tau;
else
    tauc = tau(idx);
end
end
