function [xSurf, ySurf, Cp, Cf, CL, CD] = computeCpCf(base, physicparams)
%COMPUTECPCF Compute airfoil pressure/friction coefficients for Eppler DNS.
%
%   [Cp, Cf, CL, CD] = computeCpCf(base, physicparams)
%
% base is the directory containing outbou/case1, ..., outbou/case4.
% physicparams has size 4 x 9 with rows
%   [gamma, Re, Pr, Minf, rinf, ruinf, rvinf, rwinf, rEinf].
%
% Cp and Cf are returned as 4 x 1 cell arrays. Each cell contains a
% p1 x nface2d array after spanwise averaging the 3D boundary data.

if nargin < 1 || isempty(base)
    base = fullfile('tmp', 'epplerdns');
end
base = char(base);

if nargin < 2 || isempty(physicparams) || ~isequal(size(physicparams), [4 9])
    error('physicparams must have size 4 x 9.');
end

numCases = 4;
Cp = cell(numCases, 1);
Cf = cell(numCases, 1);
CL = zeros(numCases, 1);
CD = zeros(numCases, 1);
Cx = zeros(numCases, 1);
Cy = zeros(numCases, 1);
xSurf = cell(numCases, 1);
ySurf = cell(numCases, 1);

for icase = 1:numCases
    caseDir = fullfile(base, 'outbou', sprintf('case%d', icase));
    [~, ~, ~, ~, xf] = read_rank(fullfile(caseDir, 'outbouxdg.bin'));
    [~, ~, ~, ~, nlf] = read_rank(fullfile(caseDir, 'outboundg.bin'));
    [~, ~, ~, ~, uhf] = read_rank(fullfile(caseDir, 'outbouuhmean.bin'));
    [~, ~, ~, ~, udgf] = read_rank(fullfile(caseDir, 'outbouudgmean.bin'));

    params = physicparams(icase, :);
    [Cp{icase}, Cf{icase}, xSurf{icase}, ySurf{icase}, Cx(icase), Cy(icase)] = ...
        computeCaseCpCf(xf, nlf, uhf, udgf, params);

    alpha = atan2(params(7), params(6));
    CD(icase) =  Cx(icase)*cos(alpha) + Cy(icase)*sin(alpha);
    CL(icase) = -Cx(icase)*sin(alpha) + Cy(icase)*cos(alpha);
end

plotCoefficientFigure(base, 'Cp', Cp, xSurf, ySurf, physicparams, true);
plotCoefficientFigure(base, 'Cf', Cf, xSurf, ySurf, physicparams, false);

end

function [cp, cf, x, y, Cx, Cy] = computeCaseCpCf(xf, nlf, uhf, udgf, params)
xf = squeezeTrailingSingleton(xf);
nlf = squeezeTrailingSingleton(nlf);
uhf = squeezeTrailingSingleton(uhf);
udgf = squeezeTrailingSingleton(udgf);

npf3d = size(xf, 1);
p1 = round(sqrt(npf3d));
if p1*p1 ~= npf3d
    error('Cannot infer p1 from %d boundary points per face.', npf3d);
end

nf = size(xf, 2);
xm = squeeze(mean(xf, 1));
if size(xm, 2) ~= 3
    error('Boundary coordinate data must have three physical coordinates.');
end

z = xm(:, 3);
tol = 1.0e-10*max(1, max(abs(z)));
nz = numel(unique(round(z/tol)*tol));
if nz <= 0 || mod(nf, nz) ~= 0
    error('Cannot infer spanwise face count from boundary coordinates.');
end
nx = nf / nz;

[~, ind] = sortrows(xm(:, [3 1]));
xf = permute(xf, [1 3 2]);
nlf = permute(nlf, [1 3 2]);
uhf = permute(uhf, [1 3 2]);
udgf = permute(udgf, [1 3 2]);

xf = xf(:, :, ind);
nlf = nlf(:, :, ind);
uhf = uhf(:, :, ind);
udgf = udgf(:, :, ind);

xa = squeeze(mean(reshape(xf, [p1 p1 3 nx nz]), [2 5]));
x = squeeze(xa(:, 1, :));
y = squeeze(xa(:, 2, :));

na = squeeze(mean(reshape(nlf, [p1 p1 3 nx nz]), [2 5]));
n1 = squeeze(na(:, 1, :));
n2 = squeeze(na(:, 2, :));

ncu = size(uhf, 2);
uha = squeeze(mean(reshape(uhf, [p1 p1 ncu nx nz]), [2 5]));

nc = size(udgf, 2);
udga = squeeze(mean(reshape(udgf, [p1 p1 nc nx nz]), [2 5]));

gamma = params(1);
Re = params(2);
Pr = params(3);
Minf = params(4);
pinf = 1/(gamma*Minf^2);

[p, txx, txy, ~, tyy] = nsfluxcart3d(uha(:, 1:5, :), udga(:, 6:end, :), gamma, Re, Pr);
p = squeeze(p);
txx = squeeze(txx);
txy = squeeze(txy);
tyy = squeeze(tyy);

cp = -2*(p - pinf);
t1 = txx.*n1 + txy.*n2;
t2 = txy.*n1 + tyy.*n2;
cf = -2*(t1.*n2 - t2.*n1);

master = makeLineFaceMaster(p1 - 1);
[Cx, Cy] = surfaceForceCoefficients(cp, cf, x, y, master);
end

function A = squeezeTrailingSingleton(A)
while ndims(A) > 3 && size(A, ndims(A)) == 1
    A = reshape(A, size(A, 1), size(A, 2), size(A, 3));
end
end

function master = makeLineFaceMaster(porder)
pde.nd = 2;
pde.porder = porder;
pde.pgauss = max(2*porder, 1);
pde.elemtype = 1;
pde.nodetype = 1;
master = Master(pde);
end

function plotCoefficientFigure(base, name, coeff, xSurf, ySurf, physicparams, plotExperimental)
fig = figure('Color', 'w');
clf(fig);
set(fig, 'Units', 'pixels', 'Position', [100 100 1100 900]);

axpos = [
    0.070 0.555 0.410 0.375
    0.555 0.555 0.410 0.375
    0.070 0.095 0.410 0.375
    0.555 0.095 0.410 0.375
];

if strcmp(name, 'Cp')
    ylabelText = '$-C_p$';
elseif strcmp(name, 'Cf')
    ylabelText = '$C_f$';
else
    ylabelText = name;
end

for icase = 1:4
    ax = subplot('Position', axpos(icase, :));
    plotAirfoilCoefficient(xSurf{icase}, ySurf{icase}, coeff{icase});
    if plotExperimental
        plotExperimentalCp(physicparams(icase, 2));
    end
    title(['Re = ' formatIntegerWithCommas(physicparams(icase, 2))], ...
          'Interpreter', 'none');
    if icase > 2
        xlabel('$x$', 'Interpreter', 'latex');
    else
        xlabel('');
        ax.XTickLabel = [];
    end
    if mod(icase, 2) == 1
        ylabel(ylabelText, 'Interpreter', 'latex');
    else
        ylabel('');
    end
    box on;
    set(ax, 'FontSize', 18, 'LineWidth', 1.0);
end

exportgraphics(fig, fullfile(base, [name '.png']), 'Resolution', 300);
end

function plotAirfoilCoefficient(x, y, value)
xyMidChord = [0.0 0.0; 0.8 0.015; 1.0 0.0];
lowerMask = lowerSurfaceMask(x, y, 1, xyMidChord);
upper = find(sum(lowerMask, 1) == 0);
lower = find(sum(lowerMask, 1) > 0);

hold on;
if ~isempty(upper)
    plot(x(:, upper), value(:, upper), 'r-', 'LineWidth', 2);
end
if ~isempty(lower)
    plot(x(:, lower), value(:, lower), 'b-', 'LineWidth', 2);
end
grid on;
box on;
axis tight;
end

function plotExperimentalCp(Re)
baseDir = fileparts(mfilename('fullpath'));
[upperFile, lowerFile] = experimentalCpTableFiles(baseDir, Re);

if isfile(upperFile) && isfile(lowerFile)
    upperData = load(upperFile);
    lowerData = load(lowerFile);
    plot(upperData.x(:), -upperData.Cp(:), 'ks', ...
         'MarkerSize', 6, 'LineWidth', 1.5, 'MarkerFaceColor', 'none');
    plot(lowerData.x(:), -lowerData.Cp(:), 'ks', ...
         'MarkerSize', 6, 'LineWidth', 1.5, 'MarkerFaceColor', 'none');
    return;
end

fallbackFile = fullfile(baseDir, ...
    sprintf('eppler_Re%d_alpha6_cp_digitized.mat', round(Re)));
if isfile(fallbackFile)
    expData = load(fallbackFile);
    plot(expData.x_upper(:), -expData.Cp_upper(:), 'ks', ...
         'MarkerSize', 6, 'LineWidth', 1.5, 'MarkerFaceColor', 'none');
    plot(expData.x_lower(:), -expData.Cp_lower(:), 'ks', ...
         'MarkerSize', 6, 'LineWidth', 1.5, 'MarkerFaceColor', 'none');
else
    warning('Experimental Cp table files not found for Re = %.0f.', Re);
end
end

function [upperFile, lowerFile] = experimentalCpTableFiles(baseDir, Re)
switch round(Re)
    case 100000
        stem = 'eppler_Re100000_alpha6';
    case 200000
        stem = 'eppler_Re200000_alpha602';
    case 300000
        stem = 'eppler_Re300000_alpha6';
    case 460000
        stem = 'eppler_Re460000_alpha602';
    otherwise
        stem = sprintf('eppler_Re%d_alpha6', round(Re));
end

upperFile = fullfile(baseDir, [stem '_upper_table.mat']);
lowerFile = fullfile(baseDir, [stem '_lower_table.mat']);
end

function text = formatIntegerWithCommas(value)
digits = sprintf('%.0f', value);
text = regexprep(digits, '\B(?=(\d{3})+(?!\d))', ',');
end
