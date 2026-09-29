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
irhou = ns+(1:nd);
momentum = u(irhou);
normalMomentum = momentum.'*n;
uslip = u;
uslip(irhou) = momentum-normalMomentum.*n;
fslip = tau.*(uslip-uhat);

% Axis symmetry.
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

% Positive, dimensionless margins. A continuation state is accepted only
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

% q=-grad(u), so these terms are -duz/dz and -dur/dr.
compression = (q(ns+1)-qrhoZ*uz)/rho ...
            + (q(ncu+ns+2)-qrhoR*ur)/rho ...
            - ur/radius;
f = limiting(compression*tanh(mu(end-2)*v(1)), ...
             0,mu(end-3),1.0e3,0);
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

% Pressure stays first because meshadaptfield=1 selects the sensor scalar.
% Fields: p, rho, T, Mach, AV, Y_N, Y_O, Y_NO, Y_N2, Y_O2.
s = [pressurePhysical;rho;temperature;mach;physicalav(v,mu);massFractions];
end

function velocity = visvectors(u, q, w, v, x, t, mu, eta) %#ok<INUSD>
velocity = mu(2)*u(6:7)/sum(u(1:5));
end

function va = artificialviscosity(v,mu)
av = physicalav(v,mu);
va = [av;av];
end

function av = physicalav(v,mu)
av = (mu(end-1)+mu(end)*v(2))*tanh(mu(end-2)*v(1));
end
