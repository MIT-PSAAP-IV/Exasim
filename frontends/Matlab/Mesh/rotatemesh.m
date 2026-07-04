function mesh = rotatemesh(mesh2d,tt)

%ROTATEMESH Rotate a 2D (z,r) mesh through angle coordinates tt.
%   mesh = rotatemesh(mesh2d,tt) first calls extrudemesh(mesh2d,tt), which
%   builds a 3D mesh with coordinates (z,r,theta). It then maps those
%   coordinates to Cartesian coordinates (z,r*cos(theta),r*sin(theta)).

if nargin < 2
    error('rotatemesh requires mesh2d and tt.');
end

if numel(tt) < 2
    error('tt must contain at least two angle coordinates.');
end

mesh = extrudemesh(mesh2d,tt);

z = mesh.p(1,:);
r = mesh.p(2,:);
theta = mesh.p(3,:);
mesh.p = [z; r.*cos(theta); r.*sin(theta)];

z = mesh.dgnodes(:,1,:);
r = mesh.dgnodes(:,2,:);
theta = mesh.dgnodes(:,3,:);
mesh.dgnodes(:,1,:) = z;
mesh.dgnodes(:,2,:) = r.*cos(theta);
mesh.dgnodes(:,3,:) = r.*sin(theta);

end
