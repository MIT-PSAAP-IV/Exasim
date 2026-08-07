function computeBLthicknesses(base, mesh2d, master2d, physicparams, wid, delta, m, alpha)
%COMPUTEBLTHICKNESSES Compute and plot Eppler boundary-layer thicknesses.
%
%   computeBLthicknesses(base, mesh2d, master2d, physicparams, wid, delta, m, alpha)
%
% The function reads spanwise-averaged time-mean solution files from
%   base/udgavg/case1/spanwiseudgavg.bin, ..., base/udgavg/case4/spanwiseudgavg.bin
% samples the mean field along inward normals to boundary marker wid, computes
% boundary-layer thickness, displacement thickness, momentum thickness, and
% shape parameter on the lower and upper surfaces, and exports six PNG figures
% to base:
%
%   lower_displacement_thickness.png
%   upper_displacement_thickness.png
%   lower_momentum_thickness.png
%   upper_momentum_thickness.png
%   lower_shape_parameter.png
%   upper_shape_parameter.png

if nargin < 8 || isempty(alpha)
    alpha = 0;
end
validateInputs(base, mesh2d, master2d, physicparams, wid, delta, m, alpha);
base = char(base);

numCases = size(physicparams,1);
smoothPasses = 10;
xyMidChord = [0.0 0.0; 0.8 0.015; 1.0 0.0];

lowerData = cell(numCases, 1);
upperData = cell(numCases, 1);
caseLabels = cell(numCases, 1);

for icase = 1:numCases
    fname = fullfile(base, 'udgavg', sprintf('case%d', icase), 'spanwiseudgavg.bin');
    [~, ~, ~, ~, udgavg] = read_rank(fname);
    udgavg = squeezeTrailingSingleton(udgavg);

    [x, blt, dpt, mot, H, lower] = computeCaseBL(mesh2d, master2d, udgavg, ...
        wid, delta, m, alpha, xyMidChord);

    upper = ~lower;
    lowerData{icase} = makeSurfaceData(x(lower), blt(lower), dpt(lower), ...
        mot(lower), H(lower), smoothPasses, true);
    upperData{icase} = makeSurfaceData(x(upper), blt(upper), dpt(upper), ...
        mot(upper), H(upper), smoothPasses, false);
    caseLabels{icase} = ['Re = ' formatIntegerWithCommas(physicparams(icase, 2))];
end

plotSurfaceFigure(base, 'lower_displacement_thickness', lowerData, 'dpt', ...
    caseLabels, '$\delta^*$', 'Lower surface displacement thickness');
plotSurfaceFigure(base, 'upper_displacement_thickness', upperData, 'dpt', ...
    caseLabels, '$\delta^*$', 'Upper surface displacement thickness');
plotSurfaceFigure(base, 'lower_momentum_thickness', lowerData, 'mot', ...
    caseLabels, '$\theta$', 'Lower surface momentum thickness');
plotSurfaceFigure(base, 'upper_momentum_thickness', upperData, 'mot', ...
    caseLabels, '$\theta$', 'Upper surface momentum thickness');
plotSurfaceFigure(base, 'lower_shape_parameter', lowerData, 'H', ...
    caseLabels, '$H$', 'Lower surface shape parameter');
plotSurfaceFigure(base, 'upper_shape_parameter', upperData, 'H', ...
    caseLabels, '$H$', 'Upper surface shape parameter');

end

function validateInputs(base, mesh2d, master2d, physicparams, wid, delta, m, alpha)
if nargin < 1 || isempty(base)
    error('base must be a nonempty directory path.');
end
if ~isstruct(mesh2d) || ~isfield(mesh2d, 'ne')
    error('mesh2d must be a mesh structure with field ne.');
end
if ~isstruct(master2d)
    error('master2d must be a structure.');
end
if isempty(physicparams) || ~isequal(size(physicparams), [4 9])
    error('physicparams must have size 4 x 9.');
end
if isempty(wid)
    error('wid must contain at least one boundary marker.');
end
if ~isscalar(delta) || ~isnumeric(delta) || ~isfinite(delta) || delta <= 0
    error('delta must be a positive scalar.');
end
if ~isscalar(m) || ~isnumeric(m) || ~isfinite(m) || m < 2 || fix(m) ~= m
    error('m must be an integer greater than or equal to 2.');
end
if ~isscalar(alpha) || ~isnumeric(alpha) || ~isfinite(alpha)
    error('alpha must be a finite scalar.');
end
end

function [x, blt, dpt, mot, H, lower] = computeCaseBL(mesh2d, master2d, udgavg, ...
    wid, delta, m, alpha, xyMidChord)

if ndims(udgavg) ~= 3
    error('spanwiseudgavg must have size npe x nc x ne.');
end
if size(udgavg, 2) < 20
    error('spanwiseudgavg must contain at least 20 components.');
end
if size(udgavg, 3) ~= mesh2d.ne
    error('spanwiseudgavg element count does not match mesh2d.ne.');
end

ind1 = 1;
[ugn, xgn, ~] = surfacefieldalongnormal(mesh2d, master2d, udgavg, ...
    wid, delta, m, alpha, ind1);
[~, ~, vortz] = nsevalcart3d(ugn(:, 1:5, :, :), ugn(:, 6:20, :, :));

x1 = xgn(:, 1, :, :);
x2 = xgn(:, 2, :, :);
s = sqrt((x1 - x1(:,:,:,1)).^2 + (x2 - x2(:,:,:,1)).^2);

nf = size(s, 3);
blt = zeros(nf, 1);
dpt = zeros(nf, 1);
mot = zeros(nf, 1);
H = zeros(nf, 1);
for j = 1:nf
    y = s(1, 1, j, :);
    vort = vortz(1, 1, j, :);
    [blt(j), dpt(j), mot(j), H(j)] = BLthicknesses(y(:), vort(:));
end

x = squeeze(xgn(1, 1, :, 1));
y = squeeze(xgn(1, 2, :, 1));
x = x(:);
y = y(:);
lower = lowerSurfaceMask(x, y, 1, xyMidChord);
lower = lower(:);

end

function data = makeSurfaceData(x, blt, dpt, mot, H, smoothPasses, trimLeadingEdge)
x = x(:);
blt = blt(:);
dpt = dpt(:);
mot = mot(:);
H = H(:);

valid = isfinite(x) & isfinite(blt) & isfinite(dpt) & isfinite(mot) & isfinite(H);
if trimLeadingEdge
    valid = valid & x > 0.022;
end

x = x(valid);
blt = blt(valid);
dpt = dpt(valid);
mot = mot(valid);
H = H(valid);

[x, ind] = sort(x);
data.x = x;
data.blt = smoothing(blt(ind), smoothPasses);
data.dpt = smoothing(dpt(ind), smoothPasses);
data.mot = smoothing(mot(ind), smoothPasses);
data.H = smoothing(H(ind), smoothPasses);
end

function A = squeezeTrailingSingleton(A)
while ndims(A) > 3 && size(A, ndims(A)) == 1
    A = reshape(A, size(A, 1), size(A, 2), size(A, 3));
end
end

function plotSurfaceFigure(base, basename, data, fieldName, caseLabels, ylabelText, plotTitle)
fig = figure('Color', 'w');
clf(fig);
set(fig, 'Units', 'pixels', 'Position', [100 100 900 620]);

ax = axes(fig);
hold(ax, 'on');
colors = lines(numel(data));
for icase = 1:numel(data)
    if isempty(data{icase}.x)
        warning('No data to plot for %s, case %d.', basename, icase);
        continue;
    end
    ind = find(data{icase}.x >= 0.05 & data{icase}.x <= 0.95);
    
    y = data{icase}.(fieldName);
    plot(ax, data{icase}.x(ind), y(ind), ...
         'LineWidth', 2, 'Color', colors(icase, :), ...
         'DisplayName', caseLabels{icase});    
end

grid(ax, 'on');
box(ax, 'on');
xlabel(ax, '$x$', 'Interpreter', 'latex');
ylabel(ax, ylabelText, 'Interpreter', 'latex');
title(ax, plotTitle, 'Interpreter', 'none');
legend(ax, 'Location', 'best', 'Interpreter', 'none');
set(ax, 'FontSize', 22, 'LineWidth', 1.0);
axis(ax, 'tight');

exportgraphics(fig, fullfile(base, [basename '.png']), 'Resolution', 300);
end

function text = formatIntegerWithCommas(value)
digits = sprintf('%.0f', value);
text = regexprep(digits, '\B(?=(\d{3})+(?!\d))', ',');
end
