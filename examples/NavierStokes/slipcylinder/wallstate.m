function [dutdn, dTdn, lambda] = wallstate(u, q, mu, n)

gam = mu(1);
gam1 = gam - 1.0;
% Re = mu(2);
% Pr = mu(3);
Minf = mu(4);
rinf = mu(5);
Tref = mu(10);
%mu_inf = mu(12);
mu_ref = mu(13);
Tmu_ref = mu(14);
omega = mu(15); 
R = mu(16);
rho_ref = mu(19);
Tinf = 1/(gam*gam1*Minf^2);

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

dr=1;
rx = rx*dr;
ry = ry*dr;
r1 = 1/r;
uv = ru*r1;
vv = rv*r1;
q = 0.5*(uv*uv+vv*vv);
p = gam1*(rE-r*q);
dp = 1;
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

T = p/(gam1*r);
Tphys = Tref/Tinf * T;
rphys = r*rho_ref;

mu_phys = mu_ref * (Tphys  / Tmu_ref)^omega;
lambda = (mu_phys/rphys) * sqrt(pi/(2 * R * Tphys)); 

nx = n(1);
ny = n(2);
tx = -ny;
ty = nx;
dutdn = tx*(ux*nx + uy*ny) + ty*(vx*nx + vy*ny);
dTdn = Tx*nx + Ty*ny;

end


