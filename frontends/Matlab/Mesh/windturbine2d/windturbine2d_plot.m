function windturbine2d_plot(wt)
%WINDTURBINE2D_PLOT Visualize major stages of the turbine workflow.

figure(201); clf;
local_plot_mesh(wt.baseMesh); axis equal tight;
title('Single airfoil C-mesh');

figure(202); clf; hold on;
for i = 1:numel(wt.baseLoops)
    plot(wt.baseLoops(i).vertices(:,1), wt.baseLoops(i).vertices(:,2), '-o');
end
axis equal tight;
title('Extracted single-airfoil boundary loops');

figure(203); clf; hold on;
for i = 1:numel(wt.blades)
    local_plot_mesh(wt.blades{i});
end
local_plot_rotor_and_tangents(wt);
axis equal tight;
title('Replicated blade meshes');

figure(204); clf; hold on;
if isfield(wt.info, 'background') && isfield(wt.info.background, 'farfield')
    nfar = wt.info.background.farfield.pointCount;
else
    nfar = 128;
end
theta = linspace(0, 2*pi, nfar + 1);
plot(wt.opts.Rfar*cos(theta), wt.opts.Rfar*sin(theta), 'k-o', 'MarkerSize', 3);
for i = 1:numel(wt.bladeLoops)
    q = wt.bladeLoops{i}.vertices;
    plot(q(:,1), q(:,2), 'r-');
end
local_plot_rotor_and_tangents(wt);
axis equal tight;
title('Gmsh geometry loops');

if isstruct(wt.background) && isfield(wt.background, 'p') && ~isempty(wt.background)
    figure(205); clf;
    local_plot_mesh(wt.background); axis equal tight;
    title('Background Gmsh mesh');
end

if isstruct(wt.mesh) && isfield(wt.mesh, 'p') && isfield(wt.mesh, 't')
    figure(206); clf;
    local_plot_mesh(wt.mesh); axis equal tight;
    title('Merged Exasim mesh');
end
end

function local_plot_rotor_and_tangents(wt)
theta = linspace(0, 2*pi, 300);
plot(wt.opts.Rrotor*cos(theta), wt.opts.Rrotor*sin(theta), 'b--', 'LineWidth', 1);
if ~isfield(wt.info, 'blades')
    return;
end
scale = 0.5;
for i = 1:numel(wt.info.blades)
    c = wt.info.blades(i).center;
    t = wt.info.blades(i).tangentDirection;
    quiver(c(1), c(2), scale*t(1), scale*t(2), 0, 'r', 'LineWidth', 1.5);
end
end

function local_plot_mesh(mesh)
if isempty(mesh) || ~isfield(mesh, 'p') || ~isfield(mesh, 't')
    return;
end

p = mesh.p;
t = mesh.t;
if size(p, 2) ~= 2 && size(p, 1) == 2
    p = p';
end
if size(t, 1) <= 8 && size(t, 2) > size(t, 1)
    t = t';
end

patch('faces', t, 'vertices', p, 'facecolor', [0.8,1,0.8], ...
    'edgecolor', 'k', 'Linew', 1, 'FaceAlpha', 1, 'EdgeAlpha', 1);
view(2);
end
