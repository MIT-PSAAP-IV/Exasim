function [mesh, xc, xd] = mkmesh_isoq2d4(porder, res)
%MKMESH_ISOQ2D4  Parametrized graded body-fitted 2D (z,r) shell for isoq3d v4.
%   [mesh, xc, xd] = mkmesh_isoq2d4(porder, res) builds the two-block
%   boundary-layer-graded surface mesh between the isoq wall (xl) and the
%   far-field (xu), leaving the near-axis strip open (clipped) so the axis
%   butterfly plug can fill it.  res = [n1 m1 n2 m2] are the surface/normal
%   subdivisions of block 1 (nose) and block 2 (afterbody).  Returns the
%   shell inner-edge slant curve xc (feeds the plug's frustum) and the
%   near-axis dg-node cloud xd (feeds the plug's axial distribution).
%
%   This is the v1 mkmesh_isoq2d generalized to accept res; the wall sits
%   exactly on the isoq profile (no dR standoff).

if nargin < 2 || isempty(res); res = [48 80 36 80]; end
n1 = res(1); m1 = res(2); n2 = res(3); m2 = res(4);

[xl, xu] = isoq();

% clip the near-axis strip: the plug fills r below these
ind = xl(:,2) >= 0.004; xl = xl(ind,:);
ind = xu(:,2) >= 0.006; xu = xu(ind,:);

x1 = -0.04;
x2 = 0.013;
ind = (xu(:,1) <= x1); xu1 = xu(ind,:);
ind = (xu(:,1) >  x1); xu2 = [xu1(end,:); xu(ind,:)];
ind = (xl(:,1) <= x2); xl1 = xl(ind,:);
ind = (xl(:,1) >  x2); xl2 = [xl1(end,:); xl(ind,:)];

mesh1 = surfmesh2d(xl1, xu1, n1, m1, porder, [2.0 1.2], [5 0]);
mesh2 = surfmesh2d(xl2, xu2, n2, m2, porder, [2.0 1.5], [5 0]);

% inner-edge slant curve (block-1 left column) + near-axis dg cloud for plug
xc = mesh1.p(:,1:n1+1:end);
x = mesh1.dgnodes(:,1,:); x = x(:);
y = mesh1.dgnodes(:,2,:); y = y(:);
a = xc(:,1); b = xc(:,end);
ind = y < a(2) + (b(2)-a(2))*(x-a(1))/(b(1)-a(1)) + 1e-6;
xd = [x(ind) y(ind)]';

[mesh1, mesh2] = rightleft2d(mesh1, mesh2);

mesh = mesh1;
[mesh.p, mesh.t] = connectmesh(mesh1.p', mesh1.t', mesh2.p', mesh2.t', 1e-5);
mesh.dgnodes = cat(3, mesh1.dgnodes, mesh2.dgnodes);
mesh.p = mesh.p';
mesh.t = mesh.t';
mesh.telem = mesh.tlocal;

deltay = min(mesh.p(2,:));
L = max(mesh.p(1,:));
mesh.boundaryexpr = {@(p) abs(p(2,:)-deltay)<1e-6, @(p) p(1,:)> L-1e-4, ...
                     @(p) ((p(1,:) < -1e-3) | (p(2,:) > 0.1)), @(p) abs(p(1,:))< 20 + 1e-6};
mesh.boundarycondition = [6, 2, 1, 3];
mesh.f = facenumbering(mesh.p,mesh.t,1,mesh.boundaryexpr,[]);
mesh.periodicboundary = [];
mesh.periodicexpr = {};
end
