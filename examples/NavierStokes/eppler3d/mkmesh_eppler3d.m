function mesh = mkmesh_eppler3d(porder, elemtype, gridNum, nz, span)

if nargin < 1, porder = 2;     end
if nargin < 2, elemtype = 1;   end
if nargin < 3, gridNum = -6;   end
if nargin < 4, nz = 4;         end
if nargin < 5, span = 0.1;     end

mesh2d = mkmesh_epp387(porder, elemtype, gridNum);
zz = linspace(0, span, nz + 1);
mesh = extrudemesh(mesh2d, zz);

mesh.porder = porder;
mesh.elemtype = elemtype;
mesh.nodetype = mesh2d.nodetype;

zmin = min(mesh.p(3,:));
zmax = max(mesh.p(3,:));
tol = 1e-8;

mesh.boundaryexpr = { ...
    @(p) abs(p(3,:) - zmin) < tol, ...
    @(p) abs(p(3,:) - zmax) < tol, ...
    @(p) sqrt((p(1,:) - 0.5).^2 + p(2,:).^2) < 3, ...
    @(p) abs(p(3,:) - zmin) > -inf};
mesh.boundarycondition = [3; 3; 1; 2];
mesh.curvedboundary = [0 0 0 0];
mesh.curvedboundaryexpr = {@(p) 0*p(1,:), @(p) 0*p(1,:), @(p) 0*p(1,:), @(p) 0*p(1,:)};
mesh.periodicboundary = [];
mesh.periodicexpr = {};
mesh.periodicexpr = {1, @(p) p([1 2],:), 2, @(p) p([1 2],:)};
mesh.f = facenumbering(mesh.p, mesh.t, mesh.elemtype, mesh.boundaryexpr, mesh.periodicexpr);

end
