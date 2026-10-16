function mesh = mkmesh_cylht(porder)

% Use triangular elements and 21 solid-side faces around the semicircular
% interface. The quadrilateral fluid mesh uses 28 faces, so the coincident
% interface has different element families and a non-matching 4:3 partition.
mesh = mkmesh_square(5,11,porder,1,1,1,0,1);
mesh.p(1,:) = loginc(mesh.p(1,:), 2);
mesh.dgnodes(:,1,:) = loginc(mesh.dgnodes(:,1,:), 2);
mesh = mkmesh_halfcircle(mesh, 0.85, 1.0, 1.0, pi/2, 3*pi/2);
mesh.porder = porder;
mesh.boundaryexpr = {@(p) sqrt(p(1,:).^2+p(2,:).^2)>1-1e-6, @(p) p(1,:)>-1e-7, @(p) abs(p(1,:))<20};
mesh.periodicexpr = {};
