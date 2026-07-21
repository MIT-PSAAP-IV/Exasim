% Compare the original clemeshmap wake mapping with clemeshmap2.
%
% Run from the Exasim MATLAB environment after the frontend paths are loaded.
% The script preserves mkmesh_airfoil2d_test.m and writes comparison figures
% to tempdir/clemeshmap2_eppler3d.

foilfile = fullfile(fileparts(mfilename('fullpath')), 'epp387_smoothed');
[xf,yf] = read_foil(foilfile);

porder = 2;
TEC = 1.1;
sps = [TEC, 1, 1, 1, 1, TEC, 1, 1, 1, 1, TEC];
spr = [10, 10, 10, 10, 10, 10, 10]*1.2;
yref = [0.1 0.3];

lw = 0.6;
ll = .01;

nxw = 15;
nflr = 21;
nflf = 21;
nfuf = 23;
nfur = 29;
nr   = 15;

[p0,t0] = clemeshparam6(nxw, nflr, nflf, nfuf, nfur, nr, sps, spr, yref);
[p0,t0,dg0] = clemeshmap(xf, yf, p0, t0, lw, ll, porder);

[p2,t2] = clemeshparam6(nxw, nflr, nflf, nfuf, nfur, nr, sps, spr, yref);
wakeopts = struct('expansionRatio', 2, 'nstations', 11, 'verbose', true);
[p2,t2,dg2,info2] = clemeshmap2(xf, yf, p2, t2, lw, ll, porder, wakeopts);

stats0 = local_block_stats(p0, t0);
stats2 = local_block_stats(p2, t2);
[xw0, ww0] = local_wake_width(p0, max(xf), 11);
[xw2, ww2] = local_wake_width(p2, max(xf), 11);

fprintf('original: component nodes = %d, elements = %d\n', ...
    stats0.nodes, stats0.elements);
fprintf('original: min signed area = %.16e, min abs area = %.16e\n', ...
    stats0.minSignedArea, stats0.minAbsArea);
fprintf('original: negative signed-area elements = %d, near-zero area elements = %d\n', ...
    stats0.negativeSignedAreaElements, stats0.nearZeroAreaElements);
fprintf('original: wake width monotone = %d\n', local_is_monotone(ww0));

fprintf('clemeshmap2: component nodes = %d, elements = %d\n', ...
    stats2.nodes, stats2.elements);
fprintf('clemeshmap2: min signed area = %.16e, min abs area = %.16e\n', ...
    stats2.minSignedArea, stats2.minAbsArea);
fprintf('clemeshmap2: negative signed-area elements = %d, near-zero area elements = %d\n', ...
    stats2.negativeSignedAreaElements, stats2.nearZeroAreaElements);
fprintf('clemeshmap2: wake width monotone = %d\n', local_is_monotone(ww2));

plotdir = fullfile(tempdir, 'clemeshmap2_eppler3d');
if ~exist(plotdir, 'dir')
    mkdir(plotdir);
end

figure(101); clf;
local_plot_blocks(p0, t0); axis equal tight;
title('Original clemeshmap');
saveas(gcf, fullfile(plotdir, 'original_mesh.png'));

figure(102); clf;
local_plot_blocks(p2, t2); axis equal tight;
title('clemeshmap2 wake-expanded mesh');
saveas(gcf, fullfile(plotdir, 'expanded_mesh.png'));

figure(103); clf;
local_plot_blocks(p0, t0); hold on;
local_plot_blocks(p2, t2);
axis equal tight;
xlim([0.75 1.55]);
ylim([-0.18 0.18]);
title('Trailing-edge wake comparison');
saveas(gcf, fullfile(plotdir, 'trailing_edge_zoom.png'));

figure(104); clf;
plot(xw0, ww0, 'o-', xw2, ww2, 's-', 'LineWidth', 1.5);
grid on;
xlabel('x');
ylabel('wake width');
legend('clemeshmap', 'clemeshmap2', 'Location', 'best');
title('Downstream wake width');
saveas(gcf, fullfile(plotdir, 'wake_width.png'));

fprintf('comparison plots written to %s\n', plotdir);

% Return the expanded component mesh in the workspace for interactive inspection.
p = p2; %#ok<NASGU>
t = t2; %#ok<NASGU>
xdg = dg2; %#ok<NASGU>
info = info2; %#ok<NASGU>

function stats = local_block_stats(p, t)
area = [];
for i = 1:numel(p)
    area = [area; local_signed_area(p{i}, t{i})]; %#ok<AGROW>
end
stats.nodes = sum(cellfun(@(q) size(q, 1), p));
stats.elements = sum(cellfun(@(q) size(q, 1), t));
stats.minSignedArea = min(area);
stats.minAbsArea = min(abs(area));
stats.negativeSignedAreaElements = nnz(area < 0);
stats.nearZeroAreaElements = nnz(abs(area) <= 1e-14);
end

function area = local_signed_area(p, t)
x = reshape(p(t',1), size(t,2), [])';
y = reshape(p(t',2), size(t,2), [])';
area = 0.5*sum(x.*y(:,[2:end 1]) - y.*x(:,[2:end 1]), 2);
end

function [stations, width] = local_wake_width(p, xStart, nstations)
xEnd = max(vertcat(p{:}) * [1; 0]);
stations = linspace(xStart, xEnd, nstations);
[xu, yu] = local_boundary_envelope(p{6}, 1, 0);
[xl, yl] = local_boundary_envelope(p{1}, -1, 0);
yu = interp1(xu, yu, stations, 'linear', 'extrap');
yl = interp1(xl, yl, stations, 'linear', 'extrap');
width = yu - yl;
end

function [xo, yo] = local_boundary_envelope(p, side, centerY)
if side > 0
    q = p(p(:,2) >= centerY, :);
else
    q = p(p(:,2) <= centerY, :);
end

xrounded = round(q(:,1) * 1e12) / 1e12;
[xo, ~, ic] = unique(xrounded);
if side > 0
    yo = accumarray(ic, q(:,2), [], @max);
else
    yo = accumarray(ic, q(:,2), [], @min);
end
end

function flag = local_is_monotone(width)
tol = 1e-12 * max(1, max(abs(width)));
flag = all(isfinite(width)) && all(diff(width) >= -tol);
end

function local_plot_blocks(p, t)
hold on;
for i = 1:numel(p)
    simpplot(p{i}, t{i});
end
end
