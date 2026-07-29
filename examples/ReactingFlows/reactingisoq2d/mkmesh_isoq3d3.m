function mesh = mkmesh_isoq3d3(mesh2d, ns)

r2d = mesh2d.dgnodes(:,2,:);
dR = min(r2d(:));

tt = linspace(0, pi/2, ns+1);
mesh = rotatemesh(mesh2d, tt);

ymin = min(mesh.p(2,:));
zmin = min(mesh.p(3,:));
xmax = max(mesh.p(1,:));
% ymax = max(mesh.p(2,:));
% zmax = max(mesh.p(3,:));
tol = 1e-8;

mesh.boundaryexpr = {@(p) abs(p(2,:)-ymin)<tol, ...
                     @(p) abs(p(3,:)-zmin)<tol, ...
                     @(p) abs(p(1,:)-xmax)<tol, ...
                     @(p) abs(p(2,:).^2 + p(3,:).^2 - dR^2)<tol, ...
                     @(p) -1e-3<p(1,:) && p(1,:)<xmax+1e-3 && sqrt(p(2,:).^2 + p(3,:).^2) < 0.06 + dR, ...
                     @(p) abs(p(1,:))< 20 + 1e-6};
mesh.boundarycondition = [1, 2, 1, 3, 2, 2];
mesh.f = facenumbering(mesh.p,mesh.t,1,mesh.boundaryexpr,[]);
mesh.periodicboundary = [];
mesh.periodicexpr = {};

% figure(1); clf; meshplot(mesh);
% axis on; axis equal; axis tight;

%colors = ['b', 'r', 'g', 'y', 'm'];
colors = lines(12);
figure(2); clf; hold on;
for i = 1:6
  boundaryplot(mesh,i,colors(i,:));
end
