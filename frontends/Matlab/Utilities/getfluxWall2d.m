function f = getfluxWall2d(udg,param)
fv = zeros(size(udg,1),4);
gam = param(1);
gam1 = gam - 1.0;
Re = param(2);
Pr = param(3);
Minf = param(4);
rmin = param(5);
pmin = param(7);
Tref = param(9);
alpha = 1.0e3;
muRef = 1/Re;
Tinf = 1/(gam*gam1*Minf^2);
c23 = 2.0/3.0;
r = udg(:,1);
ru = udg(:,2);
rv = udg(:,3);
rE = udg(:,4);
rx = udg(:,5);
rux = udg(:,6);
rvx = udg(:,7);
rEx = udg(:,8);
ry = udg(:,9);
ruy = udg(:,10);
rvy = udg(:,11);
rEy = udg(:,12);
% Regularization of rho (cannot be smaller than rmin)
r = rmin + lmax(r-rmin,alpha);
r1 = 1./r;
uv = ru.*r1;
vv = rv.*r1;
E = rE.*r1;
q = 0.5*(uv.*uv+vv.*vv);
p = gam1*(rE-r.*q);
% Regularization of pressure p (cannot be smaller than pmin)
p = pmin + lmax(p-pmin,alpha);
% Total enthalpy
%h = E+p*r1;
%fi = [ru, ru*uv+p, rv*uv, ru*h, ...
%        rv, ru*vv, rv*vv+p, rv*h];
ux = (rux - rx.*uv).*r1;
vx = (rvx - rx.*vv).*r1;
qx = uv.*ux + vv.*vx;
px = gam1*(rEx - rx.*q - r.*qx);
%px = px*dp;
Tx = 1/gam1*(px.*r - p.*rx).*r1.^2;
uy = (ruy - ry.*uv).*r1;
vy = (rvy - ry.*vv).*r1;
qy = uv.*uy + vv.*vy;
py = gam1*(rEy - ry.*q - r.*qy);
%py = py*dp;
Ty = 1/gam1*(py.*r - p.*ry).*r1.^2;
% Adding Artificial viscosities
T = p./(gam1*r);
T = Tref/Tinf * T;
mu = getViscosity(muRef,Tref,T,1);
fc = mu*gam/Pr;
% Viscous fluxes with artificial viscosities
txx = (mu)*c23.*(2*ux - vy);
txy = (mu).*(uy + vx);
tyy = (mu)*c23.*(2*vy - ux);
%fv = [0, txx, txy, uv.*txx + vv.*txy + fc*Tx, ...
%      0, txy, tyy, uv.*txy + vv.*tyy + fc*Ty];
fv(:,1)=0; fv(:,2)=txx; fv(:,3)=txy; fv(:,4)=fc.*Tx;
fv(:,5)=0; fv(:,6)=txy; fv(:,7)=tyy; fv(:,8)=fc.*Ty;
f = fv;