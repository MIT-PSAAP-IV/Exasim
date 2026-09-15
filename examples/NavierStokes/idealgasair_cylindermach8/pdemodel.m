function pde = pdemodel
%PDEMODEL Database ideal-gas-air Navier--Stokes model for Mach-8 cylinder.
%
% Material database coordinates are xi=log(rho_dimensional) and dimensional
% internal energy e.  Database property order:
%   w(1) p, w(2) T, w(3) mu, w(4) kappa_db=cp*mu,
%   w(5) a, w(6) p_xi, w(7) p_e, w(8) T_xi, w(9) T_e.
%
% physicsparam:
%   1 gamma, 2 Re, 3 Pr, 4 Minf,
%   5:8 freestream conservative state,
%   9 Tinf nondim, 10 Tref [K], 11 Twall [K],
%   12 rhoRef [kg/m^3], 13 uRef [m/s], 14 pRef [Pa],
%   15 eRef [J/kg], 16 LRef [m], 17 transportFactor.

pde.mass = @mass;
pde.flux = @flux;
pde.source = @source;
pde.fbou = @fbou;
pde.ubou = @ubou;
pde.initu = @initu;
pde.initw = @initw;
pde.materialstate = @materialstate;
pde.fbouhdg = @fbouhdg;
end

function m = mass(u, q, w, v, x, t, mu, eta) %#ok<INUSD>
m = sym([1.0; 1.0; 1.0; 1.0]);
end

function state = materialstate(u, q, w, v, x, t, mu, eta) %#ok<INUSD>
rho = u(1);
ux = u(2)/rho;
uy = u(3)/rho;
e = u(4)/rho - 0.5*(ux*ux + uy*uy);
state = [log(mu(12)*rho); mu(15)*e];
end

function f = flux(u, q, w, v, x, t, mu, eta) %#ok<INUSD>
gam = mu(1);
gam1 = gam - 1.0;
Pr = mu(3);
Tref = mu(10);
rhoRef = mu(12);
uRef = mu(13);
pRef = mu(14);
eRef = mu(15);
LRef = mu(16);
transportFactor = mu(17);
c23 = 2.0/3.0;

% regularization parameters, matching the analytic ideal-gas model
alpha = 1.0e3;
rmin = 1.0e-2;
pmin = 1.0e-3;

av = v(1);

r = u(1);
ru = u(2);
rv = u(3);
rE = u(4);
rx = q(1);
rux = q(2);
rvx = q(3);
rEx = q(4);
ry = q(5);
ruy = q(6);
rvy = q(7);
rEy = q(8);

% Regularize density and pressure in the same way as the analytic model.
r = rmin + lmax(r-rmin,alpha);
dr = atan(alpha*(r - rmin))/pi + (alpha*(r - rmin))/(pi*(alpha^2*(r - rmin)^2 + 1)) + 1/2;
rx = rx*dr;
ry = ry*dr;

r1 = 1/r;
uv = ru*r1;
vv = rv*r1;
E = rE*r1;
p = w(1)/pRef;
p = pmin + lmax(p-pmin,alpha);
dp = atan(alpha*(p - pmin))/pi + (alpha*(p - pmin))/(pi*(alpha^2*(p - pmin)^2 + 1)) + 1/2;
h = E + p*r1;

fi = [ru, ru*uv+p, rv*uv, ru*h, ...
      rv, ru*vv, rv*vv+p, rv*h];

ux = (rux - rx*uv)*r1;
vx = (rvx - rx*vv)*r1;
Ex = (rEx - rx*E)*r1;
ex = Ex - uv*ux - vv*vx;

uy = (ruy - ry*uv)*r1;
vy = (rvy - ry*vv)*r1;
Ey = (rEy - ry*E)*r1;
ey = Ey - uv*uy - vv*vy;

% Database derivative convention: T_xi and T_e are dimensional derivatives
% with respect to xi=log(rho_dimensional) and e_dimensional.
T_xi = w(8);
T_e = w(9);
Tx = T_xi*(rx/r) + T_e*eRef*ex;
Ty = T_xi*(ry/r) + T_e*eRef*ey;

muPhys = transportFactor*w(3);
kappaPhys = transportFactor*w(4)/Pr;
muScale = muPhys/(rhoRef*uRef*LRef);
kappaScale = kappaPhys/(rhoRef*uRef^3*LRef);

% Match the analytic pressure-gradient regularization for heat flux.
Tx = Tx*dp;
Ty = Ty*dp;

txx = muScale*c23*(2*ux - vy);
txy = muScale*(uy + vx);
tyy = muScale*c23*(2*vy - ux);
fv = [0, txx, txy, uv*txx + vv*txy + kappaScale*Tx, ...
      0, txy, tyy, uv*txy + vv*tyy + kappaScale*Ty];

fl = [av.*rx, av.*rux, av.*rvx, av.*rEx, av.*ry, av.*ruy, av.*rvy, av.*rEy];
f = reshape(fi + fv + fl, [4,2]);
end

function s = source(u, q, w, v, x, t, mu, eta) %#ok<INUSD>
s = [sym(0.0); sym(0.0); sym(0.0); sym(0.0)];
end

function fb = fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau) %#ok<INUSD>
f = flux(uhat, q, w, v, x, t, mu, eta);
fi = f(:,1)*n(1) + f(:,2)*n(2) + tau*(u-uhat);
faw = fi;
faw(1) = 0.0;
faw(end) = 0.0;
ftw = fi;
ftw(1) = 0.0;
fb = [fi faw ftw faw fi fi];
end

function ub = ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau) %#ok<INUSD>
uinf = sym(mu(5:8));
uinf = uinf(:);
u = u(:);
nx = n(1); ny = n(2);

Tinf = mu(9);
Tref = mu(10);
Twall = mu(11);
TisoW = Twall/Tref * Tinf;
utw = u(:);
utw(2:3) = 0;
utw(4) = u(1)*TisoW;

usw = u;
usw(2) = u(2) - nx*(u(2)*nx + u(3)*ny);
usw(3) = u(3) - ny*(u(2)*nx + u(3)*ny);

ub = [uinf uinf utw usw uinf u];
end

function fb = fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau) %#ok<INUSD>
Tinf = mu(9);
Tref = mu(10);
Twall = mu(11);
TisoW = Twall/Tref * Tinf;
uinf = sym(mu(5:8));
uinf = uinf(:);

fout = u - uhat;
fin = uinf - uhat;

fw = 0*u;
fw(1) = u(1) - uhat(1);
fw(2) = 0.0 - uhat(2);
fw(3) = 0.0 - uhat(3);
fw(4) = -uhat(4) + uhat(1)*TisoW;
fb = [fin fin fw fw fin fout];
end

function u0 = initu(x, mu, eta) %#ok<INUSD>
u0 = sym(mu(5:8));
end

function w0 = initw(x, mu, eta) %#ok<INUSD>
w0 = sym(zeros(9,1));
w0(1) = 1.0;
w0(2) = 300.0;
w0(3) = 1.0e-5;
w0(4) = 1.0e-2;
w0(5) = 300.0;
w0(6) = 1.0;
w0(7) = 1.0e-3;
w0(8) = 0.0;
w0(9) = 1.0e-3;
end
