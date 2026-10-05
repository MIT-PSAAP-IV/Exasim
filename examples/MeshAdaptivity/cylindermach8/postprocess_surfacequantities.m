function result = postprocess_surfacequantities(pde, wallib)
%POSTPROCESS_SURFACEQUANTITIES Read and plot saved Mach-8 cylinder wall data.
%
%   result = postprocess_surfacequantities(pde)
%   result = postprocess_surfacequantities(pde, wallib)
%
% The plotted Cp, Cf, and Cq are read from Exasim's outbousurf* files via
% readsurfacequantities.  They are not recomputed from in-memory solution
% arrays, so the plotted coordinates are the exact saved wall face
% nodes/Gauss points on the adapted mesh.

if nargin < 2 || isempty(wallib), wallib = 3; end

casepath = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(pde)
    setup = fullfile(casepath, '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m');
    if exist(setup, 'file'), run(setup); end
    pde.datapath = string(fullfile(casepath, 'backend_run'));
    pde.saveSolBouLoc = 1;
    pde.mpiprocs = count_surface_ranks(fullfile(char(pde.datapath), 'dataout', 'out'));
end

fileprefix = fullfile(char(pde.datapath), 'dataout', 'out');
nranks = pde.mpiprocs;
if ~exist([fileprefix 'bouinfo_np' num2str(nranks-1) '.bin'], 'file')
    nranks = count_surface_ranks(fileprefix);
end
if nranks < 1
    error('postprocess_surfacequantities:NoSurfaceFiles', ...
        'No outbouinfo_np*.bin files found with prefix %s.', fileprefix);
end

surf = readsurfacequantities(fileprefix, nranks, pde.saveSolBouLoc);
idx = find([surf.ib] == wallib, 1);
if isempty(idx)
    error('postprocess_surfacequantities:WallBoundaryMissing', ...
        'Boundary ID %d was not found in the saved surface files.', wallib);
end

wall = surf(idx);
tstep = size(wall.values, 4);
values = wall.values(:,:,:,tstep);
xsave = wall.x(:,:,:,tstep);
nsave = wall.n(:,:,:,tstep);

x = reshape(xsave(:,:,1), [], 1);
y = reshape(xsave(:,:,2), [], 1);
nx = reshape(nsave(:,:,1), [], 1);
ny = reshape(nsave(:,:,2), [], 1);
Cp = reshape(values(:,:,1), [], 1);
Cf = reshape(values(:,:,2), [], 1);
Cq = reshape(values(:,:,3), [], 1);
if ~isempty(wall.dA)
    dA = reshape(wall.dA(:,:,tstep), [], 1);
else
    dA = [];
end

[x, y, nx, ny, Cp, Cf, Cq, dA] = average_duplicate_points(x, y, nx, ny, Cp, Cf, Cq, dA);
[xc, yc, radius] = fit_circle(x, y);
theta = mod(atan2(y-yc, x-xc), 2*pi);
[theta, order] = sort(theta);
x = x(order); y = y(order); nx = nx(order); ny = ny(order);
Cp = Cp(order); Cf = Cf(order); Cq = Cq(order);
if ~isempty(dA), dA = dA(order); end

if any(~isfinite([theta(:); x(:); y(:); nx(:); ny(:); Cp(:); Cf(:); Cq(:)]))
    error('postprocess_surfacequantities:NonfiniteSurfaceData', ...
        'Saved wall coordinates, normals, or surface quantities contain NaN/Inf values.');
end
if ~isempty(dA) && any(~isfinite(dA(:)))
    error('postprocess_surfacequantities:NonfiniteSurfaceArea', ...
        'Saved wall quadrature weights/Jacobians contain NaN/Inf values.');
end

rhatx = (x-xc)/radius;
rhaty = (y-yc)/radius;
normalRadialDot = nx.*rhatx + ny.*rhaty;

plotdir = fullfile(char(pde.datapath), 'surfacequantities_plots');
if ~isfolder(plotdir), mkdir(plotdir); end

make_line_plot(theta, Cp, '\theta', 'C_p', ...
    'Pressure coefficient, C_p=(p-p_\infty)/(0.5\rho_\infty |u_\infty|^2)', ...
    fullfile(plotdir, 'Cp_vs_theta.png'));
make_line_plot(theta, Cf, '\theta', 'C_f', ...
    'Skin friction coefficient, C_f=t\cdot(\tau n)/(0.5\rho_\infty |u_\infty|^2)', ...
    fullfile(plotdir, 'Cf_vs_theta.png'));
make_line_plot(theta, Cq, '\theta', 'C_q', ...
    'Heat-flux coefficient, C_q=(-\kappa\nabla T\cdot n)/(\rho_\infty |u_\infty|^3)', ...
    fullfile(plotdir, 'Cq_vs_theta.png'));

make_scatter_plot(x, y, Cp, 'C_p on saved wall points', fullfile(plotdir, 'Cp_wall_points.png'));
make_scatter_plot(x, y, Cf, 'C_f on saved wall points', fullfile(plotdir, 'Cf_wall_points.png'));
make_scatter_plot(x, y, Cq, 'C_q on saved wall points', fullfile(plotdir, 'Cq_wall_points.png'));

result = struct();
result.boundary = wallib;
result.saveSolBouLoc = wall.saveSolBouLoc;
result.tstep = tstep;
result.center = [xc yc];
result.radius = radius;
result.theta = theta;
result.x = [x y];
result.n = [nx ny];
result.dA = dA;
result.Cp = Cp;
result.Cf = Cf;
result.Cq = Cq;
result.plotdir = plotdir;
result.normalRadialDotRange = [min(normalRadialDot) max(normalRadialDot)];

fprintf('Surface quantities read from %s with %d MPI rank file(s).\n', fileprefix, nranks);
fprintf('Boundary ID %d, save step %d, %d unique saved wall points, saveSolBouLoc=%d.\n', ...
    wallib, tstep, numel(theta), wall.saveSolBouLoc);
fprintf('Fitted cylinder center=(%.16g, %.16g), radius=%.16g.\n', xc, yc, radius);
fprintf('normal dot radial range = [%.6e, %.6e].\n', result.normalRadialDotRange(1), result.normalRadialDotRange(2));
fprintf('Cp range = [%.6e, %.6e].\n', min(Cp), max(Cp));
fprintf('Cf range = [%.6e, %.6e].\n', min(Cf), max(Cf));
fprintf('Cq range = [%.6e, %.6e].\n', min(Cq), max(Cq));
fprintf('Plots written to %s.\n', plotdir);
end

function nranks = count_surface_ranks(fileprefix)
files = dir([fileprefix 'bouinfo_np*.bin']);
nranks = numel(files);
end

function [x, y, nx, ny, Cp, Cf, Cq, dA] = average_duplicate_points(x, y, nx, ny, Cp, Cf, Cq, dA)
scale = max([max(x)-min(x), max(y)-min(y), 1.0]);
tol = 1.0e-12*scale;
keys = round([x(:), y(:)]/tol);
[~, ~, ic] = unique(keys, 'rows');
n = max(ic);
count = accumarray(ic, 1, [n 1]);
x = accumarray(ic, x(:), [n 1])./count;
y = accumarray(ic, y(:), [n 1])./count;
nx = accumarray(ic, nx(:), [n 1])./count;
ny = accumarray(ic, ny(:), [n 1])./count;
Cp = accumarray(ic, Cp(:), [n 1])./count;
Cf = accumarray(ic, Cf(:), [n 1])./count;
Cq = accumarray(ic, Cq(:), [n 1])./count;
if ~isempty(dA)
    dA = accumarray(ic, dA(:), [n 1]);
end
end

function [xc, yc, radius] = fit_circle(x, y)
A = [2*x(:), 2*y(:), ones(numel(x),1)];
b = x(:).^2 + y(:).^2;
c = A\b;
xc = c(1);
yc = c(2);
radius = sqrt(max(c(3) + xc*xc + yc*yc, 0.0));
end

function make_line_plot(x, y, xlab, ylab, ttl, filename)
fig = figure('Visible', 'off');
plot(x, y, 'o-', 'LineWidth', 1.0, 'MarkerSize', 4);
grid on;
xlabel(xlab, 'Interpreter', 'tex');
ylabel(ylab, 'Interpreter', 'tex');
title(ttl, 'Interpreter', 'tex');
save_figure(fig, filename);
close(fig);
end

function make_scatter_plot(x, y, c, ttl, filename)
fig = figure('Visible', 'off');
scatter(x, y, 24, c, 'filled');
axis equal tight;
grid on;
colorbar;
xlabel('x');
ylabel('y');
title(ttl, 'Interpreter', 'tex');
save_figure(fig, filename);
close(fig);
end

function save_figure(fig, filename)
if exist('exportgraphics', 'file') == 2
    exportgraphics(fig, filename, 'Resolution', 200);
else
    saveas(fig, filename);
end
end
