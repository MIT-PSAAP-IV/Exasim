function qcritvort(base, pde, mesh, dmd, mesh2d, master2d, physicparams)
%QCRITVORT Write q-criterion VTU files and plot vorticity amplitude.
%
%   qcritvort(base, pde, mesh, dmd, mesh2d, master2d, physicparams)
%
% base is the outudg directory containing case1, case2, ... subdirectories.
% For each case, this function reads
%
%   base/case<i>/outudg_t4000_np<rank>.bin
%
% writes
%
%   base/case<i>/qcrit.vtu
%
% and plots the vorticity amplitude on the two-dimensional mesh from the
% selected spanwise plane.

validateInputs(base, pde, mesh, dmd, mesh2d, master2d, physicparams);
base = char(base);

numCases = size(physicparams, 1);
axisBox = [-0.05 1.5 -0.2 0.3];
vortRange = [0 200];
nref = 2;
spanIndex = 16;

vortAmp = cell(numCases, 1);
caseLabels = cell(numCases, 1);
for icase = 1:numCases
    caseDir = fullfile(base, sprintf('case%d', icase));
    fileout = fullfile(caseDir, 'qcrit.vtu');
    filein = fullfile(caseDir, 'outudg_t4000');

    pdeCase = pde;
    pdeCase.physicsparam = physicparams(icase, :);
    UDG = paraviewqcrit(fileout, filein, pdeCase, mesh, dmd);
    vortAmp{icase} = computeVorticityAmplitude(UDG, mesh2d, master2d, spanIndex);
    caseLabels{icase} = ['Re = ' formatIntegerWithCommas(physicparams(icase, 2))];
end

plotVorticityFigure(base, vortAmp, mesh2d, caseLabels, axisBox, vortRange, nref);

end

function validateInputs(base, pde, mesh, dmd, mesh2d, master2d, physicparams)
if nargin < 7
    error('qcritvort requires base, pde, mesh, dmd, mesh2d, master2d, and physicparams.');
end
if isempty(base)
    error('base must be a nonempty outudg directory path.');
end
if ~isstruct(pde)
    error('pde must be a structure.');
end
if ~isstruct(mesh)
    error('mesh must be a structure.');
end
if isempty(dmd) || ~iscell(dmd)
    error('dmd must be a nonempty cell array.');
end
if ~isstruct(mesh2d) || ~isfield(mesh2d, 'ne')
    error('mesh2d must be a mesh structure with field ne.');
end
if ~isstruct(master2d) || ~isfield(master2d, 'npe') || ~isfield(master2d, 'npf')
    error('master2d must contain fields npe and npf.');
end
if isempty(physicparams) || size(physicparams, 2) < 4
    error('physicparams must have at least four columns: gamma, Re, Pr, Ma.');
end
end

function a = computeVorticityAmplitude(UDG, mesh2d, master2d, spanIndex)
nc = 20;
npe2d = master2d.npe;
npf2d = master2d.npf;
ne2d = mesh2d.ne;

denom = npe2d*npf2d*nc*ne2d;
if mod(numel(UDG), denom) ~= 0
    error(['UDG size is incompatible with reshape size ' ...
           '[master2d.npe, master2d.npf, 20, mesh2d.ne, nz].']);
end
nz = numel(UDG)/denom;
if nz < spanIndex
    error('Requested spanwise index %d exceeds nz = %d.', spanIndex, nz);
end

UDG = reshape(UDG, [npe2d, npf2d, nc, ne2d, nz]);
UDG2d = squeeze(UDG(:, 2, :, :, spanIndex));

[vortx, vorty, vortz] = nsevalcart3d(UDG2d(:, 1:5, :), UDG2d(:, 6:20, :));
a = sqrt(vortx.^2 + vorty.^2 + vortz.^2);
end

function plotVorticityFigure(base, vortAmp, mesh2d, caseLabels, axisBox, vortRange, nref)
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

for icase = 1:numel(vortAmp)
    ax = axes(fig, 'Position', axpos(icase, :));
    scaplot(mesh2d, vortAmp{icase}, vortRange, nref);
    colormap('jet');
    colorbarHandle = colorbar;
    colorbarHandle.Position = [axpos(icase, 1) + axpos(icase, 3) + cbgap, ...
                               axpos(icase, 2), cbwidth, axpos(icase, 4)];
    axis(axisBox);
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

exportgraphics(fig, fullfile(base, 'vorticity_amplitude.png'), 'Resolution', 300);
end

function text = formatIntegerWithCommas(value)
digits = sprintf('%.0f', value);
text = regexprep(digits, '\B(?=(\d{3})+(?!\d))', ',');
end
