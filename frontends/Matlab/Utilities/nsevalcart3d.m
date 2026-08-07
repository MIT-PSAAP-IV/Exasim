function [vortx, vorty, vortz, divv, qcriterion] = nsevalcart3d(u, q)
%NSEVALCART3D Evaluate 3D Navier-Stokes velocity-gradient diagnostics.
%
%   [vortx, vorty, vortz, divv, qcriterion] = nsevalcart3d(u, q)
%
% u contains conservative variables with components
%   rho, rho*u, rho*v, rho*w, rho*E.
%
% q contains conservative-variable gradients ordered as
%   x-gradients of all components,
%   y-gradients of all components,
%   z-gradients of all components.
%
% u and q may have size npe x nc x ne or npe x nc x ne x m.

validateInputs(u, q);

r = u(:,1,:,:);
ru = u(:,2,:,:);
rv = u(:,3,:,:);
rw = u(:,4,:,:);

rx = q(:,1,:,:);
rux = q(:,2,:,:);
rvx = q(:,3,:,:);
rwx = q(:,4,:,:);

ry = q(:,6,:,:);
ruy = q(:,7,:,:);
rvy = q(:,8,:,:);
rwy = q(:,9,:,:);

rz = q(:,11,:,:);
ruz = q(:,12,:,:);
rvz = q(:,13,:,:);
rwz = q(:,14,:,:);

r1 = 1./r;
uv = ru.*r1;
vv = rv.*r1;
wv = rw.*r1;

ux = (rux - rx.*uv).*r1;
vx = (rvx - rx.*vv).*r1;
wx = (rwx - rx.*wv).*r1;

uy = (ruy - ry.*uv).*r1;
vy = (rvy - ry.*vv).*r1;
wy = (rwy - ry.*wv).*r1;

uz = (ruz - rz.*uv).*r1;
vz = (rvz - rz.*vv).*r1;
wz = (rwz - rz.*wv).*r1;

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

function validateInputs(u, q)
if ndims(u) ~= 3 && ndims(u) ~= 4
    error('u must have size npe x nc x ne or npe x nc x ne x m.');
end
if ndims(q) ~= 3 && ndims(q) ~= 4
    error('q must have size npe x ncq x ne or npe x ncq x ne x m.');
end
if ndims(u) ~= ndims(q)
    error('u and q must have the same number of dimensions.');
end
if size(u, 2) < 5
    error('u must contain at least 5 conservative components.');
end
if size(q, 2) < 15
    error('q must contain at least 15 conservative-gradient components.');
end
if size(u, 1) ~= size(q, 1) || size(u, 3) ~= size(q, 3)
    error('u and q must have matching point and element dimensions.');
end
if ndims(u) == 4 && size(u, 4) ~= size(q, 4)
    error('u and q must have matching fourth dimensions.');
end
end
