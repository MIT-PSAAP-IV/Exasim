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

[xl1, xl2, xu1, xu2] = thermal_buckling_smooth();

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

shift = 1e-4;
xl_full = [xl1; xl2(2:end,:)];
xu_full = [xu1; xu2(2:end,:)];
xl_full(:,2) = xl_full(:,2) + shift;
xu_full(:,2) = xu_full(:,2) + shift;

xmax = max(mesh.p(1,:));
xmin = min(mesh.p(1,:));
btol = 5e-4;

[~, ia] = unique(xl_full(:,1));
xl_full_u = xl_full(ia, :);
[~, ia] = unique(xu_full(:,1));
xu_full_u = xu_full(ia, :);

mesh.boundaryexpr = {
    @(p) abs(p(1,:) - xmax) < btol, ...
    @(p) abs(p(1,:) - xmin) < btol, ...
    @(p) abs(p(2,:) - interp1(xl_full_u(:,1), xl_full_u(:,2), p(1,:), 'linear', 'extrap')) < btol, ...
    @(p) abs(p(2,:) - interp1(xu_full_u(:,1), xu_full_u(:,2), p(1,:), 'linear', 'extrap')) < btol, ...
};
mesh.f = facenumbering(mesh.p, mesh.t, 1, mesh.boundaryexpr, []);
mesh.boundarycondition = [2, 1, 3, 1];
mesh.periodicboundary = [];
mesh.periodicexpr = {};

mesh.telem = mesh.tlocal;
figure(3); clf; meshplot(mesh);
axis on; axis equal; axis tight;

figure(4); clf;
for i = 1:4
    subplot(2,2,i); boundaryplot(mesh,i); axis equal; title(sprintf('Group %d',i));
end

end


% -------------------------------------------------------------------------
function [xl1, xl2, xu1, xu2] = thermal_buckling_smooth()

Lplate = 0.2;
L1 = 0.0977;
L2 = 0.0515;
theta_plate = 20*pi/180;
Rnose = 0.01;
R  = (Rnose/tan(2*theta_plate) + (Lplate+L1+L2))*sin(theta_plate);
rt = 0.01;
L  = (Lplate+L1+L2);
h_front = 0.06;
h_back = 3.0;
ra = h_front;
N = 400;

% ---- lower curve: nose circle + cone ----
[T1, T2] = circletangentpoints(rt, 0, rt, L, R);
theta = asin(T1(2) / rt);
theta_circ = linspace(0, theta, N);
x_circ = rt * (1 - cos(theta_circ));
r_circ = rt * sin(theta_circ);
x_cone = linspace(T1(1), L, N);
r_cone = T1(2) + (R - T1(2)) * (x_cone - T1(1)) / (L - T1(1));
xl = [[x_circ, x_cone(2:end)]' [r_circ, r_cone(2:end)]'];

% ---- upper curve: arc + tangent line ----
[T1_up, T2_up] = circletangentpoints(0, 0, ra, L, h_back*R);
if T1_up(2) > 0
    pt_up = T1_up;
else
    pt_up = T2_up;
end
theta_up = atan2(pt_up(2), pt_up(1));
theta_a = linspace(pi, theta_up, N);
x_arc = ra * cos(theta_a);
r_arc = ra * sin(theta_a);
x_line = linspace(pt_up(1), L, N);
r_line = pt_up(2) + (h_back*R - pt_up(2)) * (x_line - pt_up(1)) / (L - pt_up(1));
xu = [x_arc(:) r_arc(:); x_line(2:end)' r_line(2:end)'];

% ---- split xl at nose-cone junction ----
x1 = T1(1);
i = xl(:,1) <= x1;
xl1 = xl(i,:);
i = xl(:,1) > x1 & xl(:,1) <= 1.25;
xl2 = [xl1(end,:); xl(i,:)];

% ---- split xu perpendicular to lower curve at the junction ----
P = xl1(end, :);
t = [L - T1(1), R - T1(2)];
t = t / norm(t);
% find xu point closest to the perpendicular line (where (Q-P)·t = 0)
[~, k] = min(abs((xu - P) * t'));
xu1 = xu(1:k, :);
xu_rest = xu(k+1:end, :);
i = xu_rest(:,1) <= 1.22;
xu2 = [xu1(end,:); xu_rest(i,:)];

end
