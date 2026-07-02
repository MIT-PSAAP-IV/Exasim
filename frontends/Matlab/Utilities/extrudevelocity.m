function [vx3d, vy3d] = extrudevelocity(vr2d, porder, tt)

%EXTRUDEVELOCITY Extrude radial velocity and rotate it to Cartesian components.
%   [vx3d,vy3d] = extrudevelocity(vr2d,porder,tt) extrudes the 2D radial
%   velocity vr2d through the angular coordinates tt and returns
%   vx3d = vr*cos(theta), vy3d = vr*sin(theta).

if nargin < 3
    error('extrudevelocity requires vr2d, porder, and tt.');
end

if numel(tt) < 2
    error('tt must contain at least two angle coordinates.');
end

nz = length(tt) - 1;
vr3d = extrudesol(vr2d, porder, nz);

[np2d,nc,ne2d] = size(vr2d);
theta = extrudecoord(tt, porder, np2d, nc, ne2d);

vx3d = vr3d.*cos(theta);
vy3d = vr3d.*sin(theta);

end
