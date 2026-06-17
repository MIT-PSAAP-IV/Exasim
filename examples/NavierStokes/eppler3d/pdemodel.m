function pde = pdemodel
pde.mass = @mass;
pde.flux = @flux;
pde.source = @source;
pde.fbou = @fbou;
pde.fbouhdg = @fbouhdg;
pde.ubou = @ubou;
pde.initu = @initu;
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
    fc = 1/(gam1*M2*Re*Pr);

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
    ke = 0.5*(uv*uv + vv*vv + wv*wv);
    p = gam1*(rE - r*ke);
    h = E + p*r1;

    fi = [ru, ru*uv+p, rv*uv, rw*uv, ru*h, ...
          rv, ru*vv, rv*vv+p, rw*vv, rv*h, ...
          rw, ru*wv, rv*wv, rw*wv+p, rw*h];

    ux = (rux - rx*uv)*r1;
    vx = (rvx - rx*vv)*r1;
    wx = (rwx - rx*wv)*r1;
    qx = uv*ux + vv*vx + wv*wx;
    px = gam1*(rEx - rx*ke - r*qx);
    Tx = gam*M2*(px*r - p*rx)*r1^2;

    uy = (ruy - ry*uv)*r1;
    vy = (rvy - ry*vv)*r1;
    wy = (rwy - ry*wv)*r1;
    qy = uv*uy + vv*vy + wv*wy;
    py = gam1*(rEy - ry*ke - r*qy);
    Ty = gam*M2*(py*r - p*ry)*r1^2;

    uz = (ruz - rz*uv)*r1;
    vz = (rvz - rz*vv)*r1;
    wz = (rwz - rz*wv)*r1;
    qz = uv*uz + vv*vz + wv*wz;
    pz = gam1*(rEz - rz*ke - r*qz);
    Tz = gam*M2*(pz*r - p*rz)*r1^2;

    txx = Re1*c23*(2*ux - vy - wz);
    txy = Re1*(uy + vx);
    txz = Re1*(uz + wx);
    tyy = Re1*c23*(2*vy - ux - wz);
    tyz = Re1*(vz + wy);
    tzz = Re1*c23*(2*wz - ux - vy);

    fv = [0, txx, txy, txz, uv*txx + vv*txy + wv*txz + fc*Tx, ...
          0, txy, tyy, tyz, uv*txy + vv*tyy + wv*tyz + fc*Ty, ...
          0, txz, tyz, tzz, uv*txz + vv*tyz + wv*tzz + fc*Tz];

    f = reshape(fi + fv, [5, 3]);
end

function s = source(u, q, w, v, x, t, mu, eta)
s = [sym(0.0); sym(0.0); sym(0.0); sym(0.0); sym(0.0)];
end

function fb = fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    f = flux(u, q, w, v, x, t, mu, eta);
    fn = f(:,1)*n(1) + f(:,2)*n(2) + f(:,3)*n(3) + tau*(u - uhat);
    f_wall = fn;
    f_wall(1) = 0.0;
    f_wall(2:4) = -uhat(2:4);
    f_slip = slipflux(u, uhat, n, tau);
    fb = [f_wall fn f_slip];
end

function ub = ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    uinf = sym(mu(5:9));
    uinf = uinf(:);
    u_wall = u(:);
    u_wall(2:4) = 0;
    u_slip = slipstate(u, n);
    ub = [u_wall uinf u_slip];
end

function fb = fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    uinf = sym(mu(5:9));
    uinf = uinf(:);

    f_far = uinf - uhat;

    f_wall = 0*u;
    f_wall(1) = u(1) - uhat(1);
    f_wall(2) = 0.0 - uhat(2);
    f_wall(3) = 0.0 - uhat(3);
    f_wall(4) = 0.0 - uhat(4);
    f = flux(uhat, q, w, v, x, t, mu, eta);
    f_wall(5) = f(5,1)*n(1) + f(5,2)*n(2) + f(5,3)*n(3) + tau*(u(5) - uhat(5));

    f_slip = slipflux(u, uhat, n, tau);
    fb = [f_wall f_far f_slip];
end

function f = slipflux(u, uhat, n, tau)
    us = slipstate(u, n);
    f = tau*(us - uhat);
end

function us = slipstate(u, n)
    ru = u(2);
    rv = u(3);
    rw = u(4);
    run = ru*n(1) + rv*n(2) + rw*n(3);
    us = u(:);
    us(2) = us(2) - n(1)*run;
    us(3) = us(3) - n(2)*run;
    us(4) = us(4) - n(3)*run;
end

function u0 = initu(x, mu, eta)
    u0 = sym(mu(5:9));
end
