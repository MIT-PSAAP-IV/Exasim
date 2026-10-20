function mesh = mkmesh_cylht(porder)

% Use 28 solid-side faces around the semicircular interface. The fluid mesh
% uses 21, so the coincident interface is geometrically conforming but its
% face partitions are non-matching (4:3 ratio).
mesh = mkmesh_square(5,21,porder,1,1,1,1,1);
mesh.p(1,:) = loginc(mesh.p(1,:), 2);
mesh.dgnodes(:,1,:) = loginc(mesh.dgnodes(:,1,:), 2);
mesh = mkmesh_halfcircle(mesh, 0.85, 1.0, 1.0, pi/2, 3*pi/2);
mesh.porder = porder;
mesh.boundaryexpr = {@(p) sqrt(p(1,:).^2+p(2,:).^2)>1-1e-6, @(p) p(1,:)>-1e-7, @(p) abs(p(1,:))<20};
mesh.periodicexpr = {};
