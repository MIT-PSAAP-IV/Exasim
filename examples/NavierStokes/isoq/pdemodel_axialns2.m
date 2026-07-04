function pde = pdemodel
pde.mass = @mass;
pde.flux = @flux;
pde.source = @source;
pde.fbou = @fbou;
pde.ubou = @ubou;
pde.initu = @initu;
% pde.avfield = @avfield;
pde.fbouhdg = @fbouhdg;
end

function m = mass(u, q, w, v, x, t, mu, eta)
m = sym([1.0; 1.0; 1.0; 1.0]); 
end

function f = flux(u, q, w, v, x, t, mu, eta)

gam = mu(1);
gam1 = gam - 1.0;
Re = mu(2);
Pr = mu(3);
Minf = mu(4);
Tref = mu(10);
muRef = 1/Re;
%Tinf = 1/(gam*gam1*Minf^2);
pinf = 1/(gam*Minf^2);
Tinf = pinf/(gam-1);
c23 = 2.0/3.0;

% regularization mueters
alpha = 4.0e3;
rmin = 5.0e-2;
pmin = 2.0e-3;

av = mu(12)*tanh(mu(13)*v(1));

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

% Regularization of rho (cannot be smaller than rmin)
r = rmin + lmax(r-rmin,alpha);
% Density sensor
dr = atan(alpha*(r - rmin))/pi + (alpha*(r - rmin))/(pi*(alpha^2*(r - rmin)^2 + 1)) + 1/2;
%dr=1;
rx = rx*dr;
ry = ry*dr;
r1 = 1/r;
uv = ru*r1;
vv = rv*r1;
E = rE*r1;
q = 0.5*(uv*uv+vv*vv);
p = gam1*(rE-r*q);
% Regularization of pressure p (cannot be smaller than pmin)
p = pmin + lmax(p-pmin,alpha);
% Pressure sensor
dp = atan(alpha*(p - pmin))/pi + (alpha*(p - pmin))/(pi*(alpha^2*(p - pmin)^2 + 1)) + 1/2;
%dp=1;
% Total enthalpy
h = E+p*r1;
% Inviscid fluxes
fi = [ru, ru*uv+p, rv*uv, ru*h, ...
      rv, ru*vv, rv*vv+p, rv*h];
    
ux = (rux - rx*uv)*r1;
vx = (rvx - rx*vv)*r1;
qx = uv*ux + vv*vx;
px = gam1*(rEx - rx*q - r*qx);
px = px*dp;
Tx = 1/gam1*(px*r - p*rx)*r1^2;
uy = (ruy - ry*uv)*r1;
vy = (rvy - ry*vv)*r1;
qy = uv*uy + vv*vy;
py = gam1*(rEy - ry*q - r*qy);
py = py*dp;
Ty = 1/gam1*(py*r - p*ry)*r1^2;
% Adding Artificial viscosities
T = p/(gam1*r);
Tphys = Tref/Tinf * T;
mu = getViscosity(muRef,Tref,Tphys,1);
fc = mu*gam/(Pr);
% Viscous fluxes with artificial viscosities
y = x(2);
txx = (mu)*c23*(2*ux - vy + vv/y);
txy = (mu)*(uy + vx);
tyy = (mu)*c23*(2*vy - ux + vv/y);
fv = [0, txx, txy, uv*txx + vv*txy + (fc)*Tx, ...
      0, txy, tyy, uv*txy + vv*tyy + (fc)*Ty];

fl = [av*rx, av*rux, av*rvx, av*rEx, av*ry, av*ruy, av*rvy, av*rEy];

f = fi + fv + fl;

f = reshape(f,[4,2]);        

end

% function f = avfield(u, q, w, v, x, t, mu, eta)
%     f = getavfield2d(u,q,v,mu);
% end

function f = source(u, q, w, v, x, t, mu, eta)

gam = mu(1);
gam1 = gam - 1.0;
Re = mu(2);
Pr = mu(3);
Minf = mu(4);
Tref = mu(10);
muRef = 1/Re;
Tinf = 1/(gam*gam1*Minf^2);
c23 = 2.0/3.0;

% regularization mueters
alpha = 4.0e3;
rmin = 5.0e-2;
pmin = 2.0e-3;

av = mu(12)*tanh(mu(13)*v(1));

r = u(1);
ru = u(2);
rv = u(3);
rE = u(4);
rx = q(1);
rux = q(2);
rvx = q(3);
%rEx = q(4);
ry = q(5);
ruy = q(6);
rvy = q(7);
rEy = q(8);

% Regularization of rho (cannot be smaller than rmin)
r = rmin + lmax(r-rmin,alpha);
% Density sensor
dr = atan(alpha*(r - rmin))/pi + (alpha*(r - rmin))/(pi*(alpha^2*(r - rmin)^2 + 1)) + 1/2;
%dr=1;
rx = rx*dr;
ry = ry*dr;
r1 = 1/r;
uv = ru*r1;
vv = rv*r1;
E = rE*r1;
q = 0.5*(uv*uv+vv*vv);
p = gam1*(rE-r*q);
% Regularization of pressure p (cannot be smaller than pmin)
p = pmin + lmax(p-pmin,alpha);
% Pressure sensor
dp = atan(alpha*(p - pmin))/pi + (alpha*(p - pmin))/(pi*(alpha^2*(p - pmin)^2 + 1)) + 1/2;
%dp=1;
% Total enthalpy
h = E+p*r1;
    
% Inviscid fluxes
fi = [rv, ru*vv, rv*vv, rv*h];
    
ux = (rux - rx*uv)*r1;
vx = (rvx - rx*vv)*r1;
%qx = uv*ux + vv*vx;
%px = gam1*(rEx - rx*q - r*qx);
%px = px*dp;
%Tx = 1/gam1*(px*r - p*rx)*r1^2;
uy = (ruy - ry*uv)*r1;
vy = (rvy - ry*vv)*r1;
qy = uv*uy + vv*vy;
py = gam1*(rEy - ry*q - r*qy);
py = py*dp;
Ty = 1/gam1*(py*r - p*ry)*r1^2;
% Adding Artificial viscosities
T = p/(gam1*r);
Tphys = Tref/Tinf * T;
mu = getViscosity(muRef,Tref,Tphys,1);
fc = mu*gam/(Pr);
% Viscous fluxes with artificial viscosities
y = x(2);
%txx = (mu)*c23*(2*ux - vy + vv/y);
txy = (mu)*(uy + vx);
tyy = (mu)*c23*(2*vy - ux + vv/y);
ttt = (mu)*c23*(-2*vv/y - ux - vy);
fv = [0, txy, tyy-ttt, uv*txy + vv*tyy + fc*Ty];

fl = [av*ry, av*ruy, av*rvy, av*rEy];

f = -(fi + fv + fl)/y;

f = reshape(f,[4,1]);        


end

function fb = fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
   
    fb = [sym(0.0); sym(0.0); sym(0.0); sym(0.0)];
end

function ub = ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    ub = [sym(0.0); sym(0.0); sym(0.0); sym(0.0)];
end

function fb = fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)

    gam = mu(1);
    gam1 = gam - 1.0;
    Tinf = mu(9);
    Tref = mu(10);
    Twall = mu(11);
    TisoW = Twall/Tref * Tinf;    
    uinf = sym(mu(5:8)); % freestream flow
    uinf = uinf(:);
        
    f_out = u - uhat;
    f_in = uinf - uhat;

    % iso-thermal wall boundary condition    
    f_iso = 0*u;
    f_iso(1) = u(1) - uhat(1); % extrapolate density
    f_iso(2) = 0.0  - uhat(2); % zero velocity
    f_iso(3) = 0.0  - uhat(3); % zero velocity           
    f_iso(4) = -uhat(4) + uhat(1)*TisoW; % set temperature to Twall
      
    % iso-thermal wall boundary condition    
    f3 = 0*u;
    f3(1) = u(1) - uhat(1); % extrapolate density
    f3(2) = 0.0  - uhat(2); % zero velocity
    f3(3) = 0.0  - uhat(3); % zero velocity           
    %f3(4) = -uhat(4) + uhat(1)*v(2);

    % symmetry boundary condition    
    r = uhat(1);
    ru = uhat(2);
    rv = uhat(3);
    rE = uhat(4);
    r1 = 1/r;
    uv = ru*r1;        
    vv = rv*r1;    
    ke = 0.5*(uv*uv+vv*vv);
    p = gam1*(rE-r*ke);

    ry = q(5);
    ruy = q(6);    
    rvy = q(7);
    rEy = q(8);    
    uy = (ruy - ry*uv)*r1;    
    vy = (rvy - ry*vv)*r1;
    qy = uv*uy + vv*vy;
    py = gam1*(rEy - ry*ke - r*qy);
    Ty = 1/gam1*(py*r - p*ry)*r1^2;

    f_sym = 0*u;
    f_sym(1) = ry + tau*(u(1) - uhat(1));  % dr/dy = 0
    f_sym(2) = uy + tau*(u(2) - uhat(2));  % du/dy = 0
    f_sym(3) = vy + tau*(0.0 - uhat(3));   % v = 0
    f_sym(4) = Ty + tau*(u(4) - uhat(4));  % dT/dy = 0
    
    % slip wall condition                
    ru = u(2);
    rv = u(3);
    nx = n(1);
    ny = n(2);   
    run = ru*nx + rv*ny;        
    uinf = u;
    uinf(2) = uinf(2) - nx.*run;
    uinf(3) = uinf(3) - ny.*run;
    f_slip = tau*(uinf - uhat);    
    
    % zero gradient condition
    q = q(:);
    f_grad = q(1:4)*nx + q(5:8)*ny + tau*(u(:) - uhat(:));
        
    % supersonic inflow, supersonic outflow, isothermal, symmetry, slip wall, gradient                 
    fb = [f_in f_out f_iso f_sym f_slip f_grad f3];

end

function u0 = initu(x, mu, eta)
    u0 = sym(mu(5:8)); % freestream flow   
end


