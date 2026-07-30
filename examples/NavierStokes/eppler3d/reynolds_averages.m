function reynolds_averages(base, mesh2d, physicparams)
%REYNOLDS_AVERAGES Plot Eppler DNS Reynolds/Favre average fields.
%
%   reynolds_averages(base, mesh2d, physicparams)
%
% base is the directory containing sra/case1, ..., sra/case4.
% Each case directory is expected to contain sol2davg_step_0.bin with
% Reynolds averages stored as npe x 30 x ne. The file may contain either
% raw payload values or a three-double header [npe, 30, ne] followed by the
% payload. mesh2d is the 2D mesh used by scaplot. physicparams must have
% size 4 x 9 and is used only for case labels.

if nargin < 1 || isempty(base)
    base = "/Users/cuongnguyen/Documents/Exasim/tmp/epplerdns";
end
base = char(base);

if nargin < 3 || isempty(physicparams) || ~isequal(size(physicparams), [4 9])
    error('physicparams must have size 4 x 9.');
end
if ~isfield(mesh2d, 'dgnodes')
    error('mesh2d.dgnodes is required.');
end

npe = size(mesh2d.dgnodes, 1);
ne = size(mesh2d.dgnodes, 3);
ncReavg = 30;
numCases = 4;

reavg = cell(numCases, 1);
faavg = cell(numCases, 1);
pMeanOverPinf = cell(numCases, 1);
pRmsOverPinf = cell(numCases, 1);
caseLabels = cell(numCases, 1);
for icase = 1:numCases
    fname = fullfile(base, 'sra', sprintf('case%d', icase), 'sol2davg_step_0.bin');
    reavg{icase} = readReynoldsAverageFile(fname, npe, ncReavg, ne);
    faavg{icase} = FavreAverages(reavg{icase});
    pinf = 1/(physicparams(icase, 1)*physicparams(icase, 4)^2);
    pMeanOverPinf{icase} = reavg{icase}(:, 9, :)/pinf;
    pRmsOverPinf{icase} = faavg{icase}(:, 44, :)/pinf;
    caseLabels{icase} = ['Re = ' formatIntegerWithCommas(physicparams(icase, 2))];
end

axisBox = [-0.05 1.5 -0.2 0.3];
nref = 2;

plotCaseFigure(base, 'u_mean', reavg, 6, mesh2d, caseLabels, axisBox, nref);
plotCaseFigure(base, 'p_mean_over_pinf', pMeanOverPinf, 1, mesh2d, caseLabels, axisBox, nref);
plotCaseFigure(base, 'u_rms', faavg, 40, mesh2d, caseLabels, axisBox, nref);
plotCaseFigure(base, 'p_rms_over_pinf', pRmsOverPinf, 1, mesh2d, caseLabels, axisBox, nref);
plotCaseFigure(base, 'turbulent_kinetic_energy', faavg, 33, mesh2d, caseLabels, axisBox, nref);

stressComponents = {
    27, 'tau11', 'Reynolds stress tau11';
    28, 'tau22', 'Reynolds stress tau22';
    29, 'tau33', 'Reynolds stress tau33';
    30, 'tau12', 'Reynolds stress tau12';
    31, 'tau13', 'Reynolds stress tau13';
    32, 'tau23', 'Reynolds stress tau23'
};
for k = 1:size(stressComponents, 1)
    plotCaseFigure(base, stressComponents{k, 2}, ...
                   faavg, stressComponents{k, 1}, mesh2d, caseLabels, axisBox, nref);
end

end

function reavg = readReynoldsAverageFile(filename, npe, nc, neExpected)
fid = fopen(filename, 'r');
if fid < 0
    error('Cannot open Reynolds-average file: %s', filename);
end
cleanup = onCleanup(@() fclose(fid));

data = fread(fid, 'double');
if isempty(data)
    error('Reynolds-average file is empty: %s', filename);
end

if numel(data) >= 3 && isHeader(data(1:3), npe, nc)
    ne = data(3);
    payload = data(4:end);
    if numel(payload) ~= npe * nc * ne
        error('Header payload size mismatch in %s.', filename);
    end
else
    payload = data;
    if mod(numel(payload), npe * nc) ~= 0
        error('Payload size is incompatible with npe=%d and nc=%d in %s.', ...
              npe, nc, filename);
    end
    ne = numel(payload) / (npe * nc);
end

if ne ~= neExpected
    error('Element count mismatch in %s: file has %d, mesh has %d.', ...
          filename, ne, neExpected);
end

reavg = reshape(payload, [npe, nc, ne]);
end

function tf = isHeader(header, npe, nc)
tf = numel(header) == 3 && ...
     all(isfinite(header)) && ...
     all(header == floor(header)) && ...
     header(1) == npe && header(2) == nc && header(3) > 0;
end

function plotCaseFigure(base, basename, data, component, mesh2d, caseLabels, axisBox, nref)
fig = figure('Color', 'w');
clf(fig);
set(fig, 'Units', 'pixels', 'Position', [100 100 1600 560]);

axpos = [
    0.045 0.565 0.415 0.370
    0.535 0.565 0.415 0.370
    0.045 0.095 0.415 0.370
    0.535 0.095 0.415 0.370
];
cbgap = 0.006;
cbwidth = 0.012;

for icase = 1:4
    ax = axes(fig, 'Position', axpos(icase, :));
    field = data{icase}(:, component, :);
    scaplot(mesh2d, field, [], nref);
    axis(axisBox);
    colormap('jet');
    cb = colorbar;
    cb.Position = [axpos(icase,1) + axpos(icase,3) + cbgap, ...
                   axpos(icase,2), cbwidth, axpos(icase,4)];
    title(caseLabels{icase}, 'Interpreter', 'none');
    if icase > 2
        xlabel('$x$', 'Interpreter', 'latex');
    else
        xlabel('');
        ax.XTickLabel = [];
    end
    if mod(icase, 2) == 1
        ylabel('$y$', 'Interpreter', 'latex');
    else
        ylabel('');
        ax.YTickLabel = [];
    end
    set(ax, 'FontSize', 22);
end

outfile = fullfile(base, [basename '.png']);
exportgraphics(fig, outfile, 'Resolution', 300);
end

function text = formatIntegerWithCommas(value)
digits = sprintf('%.0f', value);
text = regexprep(digits, '\B(?=(\d{3})+(?!\d))', ',');
end
