
function pde = pdemodel
pde.mass = @mass;
pde.flux = @flux;
pde.source = @source;
pde.fbou = @fbou;
pde.fbouhdg = @fbouhdg;
pde.ubou = @ubou;
pde.initu = @initu;
pde.initv = @initv;
pde.avfield = @avfield;
pde.qoivolume = @qoivolume;
end

function m = mass(u, q, w, v, x, t, mu, eta)
m = sym([1.0; 1.0; 1.0; 1.0; 1.0]); 
end

function f = flux(u, q, w, v, x, t, mu, eta)
    gam = mu(1);    
    gam1 = gam - 1.0;
    Re = mu(2);
    Pr = mu(3);
    Minf = mu(4);
    Re1 = 1/Re;
    M2 = Minf^2;
    c23 = 2.0/3.0;
    av = v(1);
    r = u(1);
    ru = u(2);
    rv = u(3);
    rw = u(4);
    rE = u(5);
    rx = q(1);
    rux = q(2);
    rvx = q(3);
    rwx = q(4);
    rEx = q(5);
    ry = q(6);
    ruy = q(7);
    rvy = q(8);
    rwy = q(9);
    rEy = q(10);
    rz = q(11);
    ruz = q(12);
    rvz = q(13);
    rwz = q(14);
    rEz = q(15);
    r1 = 1/r;
    uv = ru*r1;
    vv = rv*r1;
    wv = rw*r1;
    E = rE*r1;
    ke = 0.5*(uv*uv+vv*vv+wv*wv);
    p = gam1*(rE-r*ke);
    T = gam*M2*p*r1;
    mut = 1.4042*T*sqrt(T)/(T + 0.40417);
    fc = mut/(gam1*M2*Re*Pr);
    h = E+p*r1;
    fi = [ru, ru*uv+p, rv*uv, rw*uv, ru*h, ...
          rv, ru*vv, rv*vv+p, rw*vv, rv*h, ...
          rw, ru*wv, rv*wv, rw*wv+p, rw*h];
    ux = (rux - rx*uv)*r1;
    vx = (rvx - rx*vv)*r1;
    wx = (rwx - rx*wv)*r1;
    kex = uv*ux + vv*vx + wv*wx;
    px = gam1*(rEx - rx*ke - r*kex);
    Tx = gam*M2*(px*r - p*rx)*r1^2;
    uy = (ruy - ry*uv)*r1;
    vy = (rvy - ry*vv)*r1;
    wy = (rwy - ry*wv)*r1;
    key = uv*uy + vv*vy + wv*wy;
    py = gam1*(rEy - ry*ke - r*key);
    Ty = gam*M2*(py*r - p*ry)*r1^2;
    uz = (ruz - rz*uv)*r1;
    vz = (rvz - rz*vv)*r1;
    wz = (rwz - rz*wv)*r1;
    kez = uv*uz + vv*vz + wv*wz;
    pz = gam1*(rEz - rz*ke - r*kez);
    Tz = gam*M2*(pz*r - p*rz)*r1^2;
    txx = mut*Re1*c23*(2*ux - vy - wz);
    txy = mut*Re1*(uy + vx);
    txz = mut*Re1*(uz + wx);
    tyy = mut*Re1*c23*(2*vy - ux - wz);
    tyz = mut*Re1*(vz + wy);
    tzz = mut*Re1*c23*(2*wz - ux - vy);
    fv = [0, txx, txy, txz, uv*txx + vv*txy + wv*txz + fc*Tx, ...
          0, txy, tyy, tyz, uv*txy + vv*tyy + wv*tyz + fc*Ty,...
          0, txz, tyz, tzz, uv*txz + vv*tyz + wv*tzz + fc*Tz];
    fl = [av*rx, av*rux, av*rvx, av*rwx, av*rEx, ...
          av*ry, av*ruy, av*rvy, av*rwy, av*rEy, ...
          av*rz, av*ruz, av*rvz, av*rwz, av*rEz];
    f = fi+fv+fl;
    f = reshape(f,[5,3]);    
end

function s = source(u, q, w, v, x, t, mu, eta)
s = [sym(0.0); sym(0.0); sym(0.0); sym(0.0); sym(0.0)];
end

function fb = fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
f = flux(u, q, w, v, x, t, mu, eta);
fb = f(:,1)*n(1) + f(:,2)*n(2) + f(:,3)*n(3) + tau*(u-uhat);
end

function ub = ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
ub = sym([0.0; 0.0; 0.0; 0.0; 0.0]); 
end

function u0 = initu(x, mu, eta)

    L = 1.0;
    x1 = x(1);
    x2 = x(2);
    x3 = x(3);
    gam = mu(1);
    Minf = mu(4);
    M2 = Minf^2;
    p0 = 1/(gam*M2);
    
    uv = sin(x1/L) * cos(x2/L) * cos(x3/L);
    vv = -cos(x1/L) * sin(x2/L) * cos(x3/L);
    wv = 0.0;
    p = p0 + (1/16) * (cos(2*x1/L) + cos(2*x2/L)) * (2.0 + cos(2*x3/L));
    r = gam*M2*p; % T(t=0) = 1 and p = rho*T/(gamma*Mref^2).
    
    u01 = r;
    u02 = r*uv;
    u03 = r*vv;
    u04 = r*wv;
    u05 = p/(gam-1) + 0.5*r*(uv^2+vv^2+wv^2);

    u0 = [u01; u02; u03; u04; u05];
end

function v0 = initv(x, mu, eta)
    v0 = sym(0.0);
end

function myavfield = avfield(u, q, w, v, x, t, mu, eta)
    gam = mu(1);
    hm = mu(6);
    avcoeff = mu(7);
    porder = mu(8);
    myavfield = getavfield3d(u, q, hm, gam, avcoeff, porder);
end

function fb = fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
f = flux(u, q, w, v, x, t, mu, eta);
fb = f(:,1)*n(1) + f(:,2)*n(2) + f(:,3)*n(3) + tau*(u-uhat);
end

function s = qoivolume(u, q, w, v, x, t, mu, eta)
    gam = mu(1);
    Re = mu(2);
    Minf = mu(4);
    rhoRef = mu(5);
    M2 = Minf^2;
    Omega = (2*pi)^3;

    r = u(1);
    ru = u(2);
    rv = u(3);
    rw = u(4);
    rE = u(5);
    rx = q(1);
    rux = q(2);
    rvx = q(3);
    rwx = q(4);
    ry = q(6);
    ruy = q(7);
    rvy = q(8);
    rwy = q(9);
    rz = q(11);
    ruz = q(12);
    rvz = q(13);
    rwz = q(14);

    r1 = 1/r;
    uv = ru*r1;
    vv = rv*r1;
    wv = rw*r1;
    ke = 0.5*(uv*uv + vv*vv + wv*wv);
    p = (gam - 1.0)*(rE - r*ke);
    T = gam*M2*p*r1;
    mut = 1.4042*T*sqrt(T)/(T + 0.40417);

    ux = (rux - rx*uv)*r1;
    vx = (rvx - rx*vv)*r1;
    wx = (rwx - rx*wv)*r1;
    uy = (ruy - ry*uv)*r1;
    vy = (rvy - ry*vv)*r1;
    wy = (rwy - ry*wv)*r1;
    uz = (ruz - rz*uv)*r1;
    vz = (rvz - rz*vv)*r1;
    wz = (rwz - rz*wv)*r1;

    curl2 = (wy - vz)^2 + (uz - wx)^2 + (vx - uy)^2;
    divu = ux + vy + wz;

    s(1) = 0.5*r*(uv*uv + vv*vv + wv*wv)/(rhoRef*Omega);
    s(2) = mut*curl2/(rhoRef*Re*Omega);
    s(3) = (4.0/3.0)*mut*divu^2/(rhoRef*Re*Omega);
end

function avField = getavfield3d(u, q, hm, gam, avcoeff, porder)
    gam1 = gam - 1.0;

    alpha = 1.0e3;
    rmin = 1.0e-3;
    Hmin = 1.0e-4;
    divmax = 50.0;
    sbmax = 4.0;
    sb0 = 0.1;

    r = u(1);
    ru = u(2);
    rv = u(3);
    rw = u(4);
    rE = u(5);

    rx = q(1);
    rux = q(2);
    rvx = q(3);
    rwx = q(4);
    ry = q(6);
    ruy = q(7);
    rvy = q(8);
    rwy = q(9);
    rz = q(11);
    ruz = q(12);
    rvz = q(13);
    rwz = q(14);

    % Regularize density and total enthalpy before forming the acoustic scale.
    r = rmin + lmax(r-rmin, alpha);
    r1 = 1/r;
    uv = ru*r1;
    vv = rv*r1;
    wv = rw*r1;
    E = rE*r1;
    kinetic = 0.5*(uv*uv + vv*vv + wv*wv);
    H = gam*E - gam1*kinetic;
    H = Hmin + lmax(H-Hmin, alpha);
    cstar = sqrt((2.0*gam1*H)/(gam + 1.0));

    ux = (rux - rx*uv)*r1;
    vx = (rvx - rx*vv)*r1;
    wx = (rwx - rx*wv)*r1;
    uy = (ruy - ry*uv)*r1;
    vy = (rvy - ry*vv)*r1;
    wy = (rwy - ry*wv)*r1;
    uz = (ruz - rz*uv)*r1;
    vz = (rvz - rz*vv)*r1;
    wz = (rwz - rz*wv)*r1;

    divu = ux + vy + wz;
    vortx = wy - vz;
    vorty = uz - wx;
    vortz = vx - uy;
    vort = sqrt(vortx*vortx + vorty*vorty + vortz*vortz);

    divu = limiting(divu, 0.0, divmax, alpha, 5.0);
    vort = limiting(vort, 0.0, divmax, alpha, 0.0);
    ducros = divu*divu/(divu*divu + vort*vort + 1.0e-16);

    sb = sqrt(hm/porder)*(divu/cstar)*ducros;
    avField = avcoeff*limiting(sb, 0.0, sbmax, alpha, sb0);
end
