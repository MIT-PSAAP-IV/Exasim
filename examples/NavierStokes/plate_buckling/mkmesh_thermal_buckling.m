function mesh = mkmesh_thermal_buckling(porder, nx1, nxf, ny, bump_amp, bump_loc, bump_width)

arguments
    porder (1,1) double = 2
    nx1 (1,1) double = 36
    nxf (1,1) double = 120
    ny (1,1) double = 100
    bump_amp (1,1) double = 0.0
    bump_loc (1,1) double = 0.22
    bump_width (1,1) double = 0.02
end

[xl, xu, T1, T2] = thermal_buckling_smooth();

x1 = T1(1);
x2 = T2(1);

ind = (xl(:,1) <= x1);
xl1 = xl(ind,:);

ind = (xu(:,1) <= x2);
xu1 = xu(ind,:);

ind = (xl(:,1) > x1) & (xl(:,1) <= 1.25);
xl2 = [xl1(end,:); xl(ind,:)];

ind = (xu(:,1) > x2) & (xu(:,1) <= 1.22);
xu2 = [xu1(end,:); xu(ind,:)];

% Apply Gaussian bump to plate wall
if bump_amp ~= 0
    xl2(:,2) = xl2(:,2) + bump_amp * exp(-((xl2(:,1) - bump_loc) / bump_width).^2);
end

mesh1 = surfmesh2d(xl1, xu1, nx1, ny, porder, [2.0 1.2], [5 0]);
mesh2 = surfmesh2d(xl2, xu2, nxf, ny, porder, [3.5 1.5], [5 0]);

[mesh1, mesh2] = rightleft2d(mesh1, mesh2);

figure(1); clf; meshplot(mesh1);
axis equal; axis tight;

figure(2); clf; meshplot(mesh2);
axis equal; axis tight;

mesh = mesh1;
[mesh.p, mesh.t] = connectmesh(mesh1.p', mesh1.t', mesh2.p', mesh2.t', 1e-5);
mesh.dgnodes = cat(3, mesh1.dgnodes, mesh2.dgnodes);
mesh.p = mesh.p';
mesh.t = mesh.t';

mesh.p(2,:) = mesh.p(2,:) + 1e-4;
mesh.dgnodes(:,2,:) = mesh.dgnodes(:,2,:) + 1e-4;

% Reconstruct full xl/xu curves (accounting for y-shift)
shift = 1e-4;
xl_full = [xl1; xl2(2:end,:)];
xu_full = [xu1; xu2(2:end,:)];
xl_full(:,2) = xl_full(:,2) + shift;
xu_full(:,2) = xu_full(:,2) + shift;

% Boundary detection via facenumbering with curve proximity
xmax = max(mesh.p(1,:));
xmin = min(mesh.p(1,:));
btol = 5e-4;

% Ensure unique x for interp1
[~, ia] = unique(xl_full(:,1));
xl_full_u = xl_full(ia, :);
[~, ia] = unique(xu_full(:,1));
xu_full_u = xu_full(ia, :);

mesh.boundaryexpr = {
    @(p) abs(p(1,:) - xmax) < btol, ...  % 1: outflow (check first, avoids xmax-overlap with farfield)
    @(p) abs(p(1,:) - xmin) < btol, ...  % 2: inflow
    @(p) abs(p(2,:) - interp1(xl_full_u(:,1), xl_full_u(:,2), p(1,:), 'linear', 'extrap')) < btol, ...  % 3: wall
    @(p) abs(p(2,:) - interp1(xu_full_u(:,1), xu_full_u(:,2), p(1,:), 'linear', 'extrap')) < btol, ...  % 4: farfield
};
mesh.f = facenumbering(mesh.p, mesh.t, 1, mesh.boundaryexpr, []);
mesh.boundarycondition = [2, 1, 3, 1];  % outflow, inflow, wall, farfield
mesh.periodicboundary = [];
mesh.periodicexpr = {};

mesh.telem = mesh.tlocal;
figure(3); clf; meshplot(mesh);
axis on; axis equal; axis tight;

% Verify boundary groups
figure(4); clf;
for i = 1:4
    subplot(2,2,i); boundaryplot(mesh,i); axis equal; title(sprintf('Group %d',i));
end

end
