function [p, txx, txy, txz, tyy, tyz, tzz, Qx, Qy, Qz, qcriterion, vortx, vorty, vortz, divv] = nsfluxcart3d(u, q, gam, Re, Pr)

    gam1 = gam - 1.0;
    Minf = 1.0;
    Re1 = 1/Re;
    M2 = Minf^2;
    c23 = 2.0/3.0;
    fc = 1/(gam1*M2*Re*Pr);

    r = u(:,1,:);
    ru = u(:,2,:);
    rv = u(:,3,:);
    rw = u(:,4,:);
    rE = u(:,5,:);

    rx = q(:,1,:);
    rux = q(:,2,:);
    rvx = q(:,3,:);
    rwx = q(:,4,:);
    rEx = q(:,5,:);

    ry = q(:,6,:);
    ruy = q(:,7,:);
    rvy = q(:,8,:);
    rwy = q(:,9,:);
    rEy = q(:,10,:);

    rz = q(:,11,:);
    ruz = q(:,12,:);
    rvz = q(:,13,:);
    rwz = q(:,14,:);
    rEz = q(:,15,:);

    r1 = 1./r;
    uv = ru.*r1;
    vv = rv.*r1;
    wv = rw.*r1;
    ke = 0.5*(uv.*uv + vv.*vv + wv.*wv);
    p = gam1*(rE - r.*ke);

    ux = (rux - rx.*uv).*r1;
    vx = (rvx - rx.*vv).*r1;
    wx = (rwx - rx.*wv).*r1;
    qx = uv.*ux + vv.*vx + wv.*wx;
    px = gam1*(rEx - rx.*ke - r.*qx);
    Tx = gam*M2*(px.*r - p.*rx).*r1.^2;

    uy = (ruy - ry.*uv).*r1;
    vy = (rvy - ry.*vv).*r1;
    wy = (rwy - ry.*wv).*r1;
    qy = uv.*uy + vv.*vy + wv.*wy;
    py = gam1*(rEy - ry.*ke - r.*qy);
    Ty = gam*M2*(py.*r - p.*ry).*r1.^2;

    uz = (ruz - rz.*uv).*r1;
    vz = (rvz - rz.*vv).*r1;
    wz = (rwz - rz.*wv).*r1;
    qz = uv.*uz + vv.*vz + wv.*wz;
    pz = gam1*(rEz - rz.*ke - r.*qz);
    Tz = gam*M2*(pz.*r - p.*rz).*r1.^2;

    txx = Re1.*c23.*(2.*ux - vy - wz);
    txy = Re1.*(uy + vx);
    txz = Re1.*(uz + wx);
    tyy = Re1.*c23.*(2.*vy - ux - wz);
    tyz = Re1.*(vz + wy);
    tzz = Re1.*c23.*(2.*wz - ux - vy);

    Qx = fc*Tx; 
    Qy = fc*Ty;
    Qz = fc*Tz;

    divv = ux + vy + wz;

    vortx = wy - vz;
    vorty = uz - wx;
    vortz = vx - uy;

    s11 = ux;
    s22 = vy;
    s33 = wz;
    s12 = 0.5*(uy + vx);
    s13 = 0.5*(uz + wx);
    s23 = 0.5*(vz + wy);
    strain2 = s11.*s11 + s22.*s22 + s33.*s33 + ...
              2.0*(s12.*s12 + s13.*s13 + s23.*s23);
    rotation2 = 0.5*(vortx.*vortx + vorty.*vorty + vortz.*vortz);
    qcriterion = 0.5*(rotation2 - strain2);
end
