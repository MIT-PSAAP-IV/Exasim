function UDG = paraviewqcrit(fileout, filein, pde, mesh, dmd)
%PARAVIEWQCRIT Write q-criterion, x-velocity, and pressure to a VTU file.
%
%   paraviewqcrit(fileout, filein, pde, mesh, dmd)
%
% fileout is the output VTU filename, with or without the .vtu extension.
% filein is the base solution filename read by getsolution, without the
% _np<rank>.bin suffix. The output fields are:
%
%   qcrit = qcriterion clipped to [-200, 200]
%   u     = rho*u/rho
%   p     = Euler pressure

validateInputs(fileout, filein, pde, mesh, dmd);

npe = inferNpe(pde, mesh);
[gamma, mach] = inferGasParameters(pde);

UDG = getsolution(string(filein), dmd, npe);
if size(UDG, 2) < 20
    error('UDG must contain at least 20 components to compute qcriterion.');
end

qcrit = qcriterion(UDG);
qcrit = min(max(qcrit, -200), 200);

visfield = zeros(size(UDG, 1), 3, size(UDG, 3), 'like', UDG);
visfield(:, 1, :) = qcrit;
visfield(:, 2, :) = UDG(:, 2, :)./UDG(:, 1, :);
visfield(:, 3, :) = eulereval3d(UDG, 'p', gamma, mach);

app = pde;
app.visscalars = {'qcrit', 1, 'u', 2, 'p', 3};
app.visvectors = {};
app.visfilename = outputBaseName(fileout);
if ~isfield(app, 'viselem')
    app.viselem = [];
end
if ~isfield(app, 'visdt')
    app.visdt = [];
end

vis(visfield, app, mesh);

end

function validateInputs(fileout, filein, pde, mesh, dmd)
if nargin < 5
    error('paraviewqcrit requires fileout, filein, pde, mesh, and dmd.');
end
if isempty(fileout)
    error('fileout must be a nonempty VTU filename.');
end
if isempty(filein)
    error('filein must be a nonempty solution-file base name.');
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
requiredPdeFields = {'porder', 'nd', 'elemtype'};
for i = 1:numel(requiredPdeFields)
    if ~isfield(pde, requiredPdeFields{i})
        error('pde.%s is required by vis.', requiredPdeFields{i});
    end
end
end

function npe = inferNpe(pde, mesh)
if isfield(pde, 'npe') && ~isempty(pde.npe)
    npe = pde.npe;
elseif isfield(mesh, 'dgnodes') && ~isempty(mesh.dgnodes)
    npe = size(mesh.dgnodes, 1);
else
    error('Cannot infer npe. Provide pde.npe or mesh.dgnodes.');
end
end

function [gamma, mach] = inferGasParameters(pde)
if isfield(pde, 'gamma') && ~isempty(pde.gamma)
    gamma = pde.gamma;
elseif isfield(pde, 'gam') && ~isempty(pde.gam)
    gamma = pde.gam;
elseif isfield(pde, 'physicsparam') && numel(pde.physicsparam) >= 1
    gamma = pde.physicsparam(1);
else
    error('Cannot infer gamma. Provide pde.gamma, pde.gam, or pde.physicsparam(1).');
end

if isfield(pde, 'mach') && ~isempty(pde.mach)
    mach = pde.mach;
elseif isfield(pde, 'Ma') && ~isempty(pde.Ma)
    mach = pde.Ma;
elseif isfield(pde, 'Minf') && ~isempty(pde.Minf)
    mach = pde.Minf;
elseif isfield(pde, 'physicsparam') && numel(pde.physicsparam) >= 4
    mach = pde.physicsparam(4);
else
    mach = 1.0;
end
end

function visbase = outputBaseName(fileout)
[outdir, name, ext] = fileparts(char(fileout));
if isempty(name)
    error('fileout must include an output filename.');
end
if ~isempty(ext) && ~strcmpi(ext, '.vtu')
    error('fileout must have extension .vtu or no extension.');
end
if ~isempty(outdir) && ~isfolder(outdir)
    mkdir(outdir);
end
visbase = string(fullfile(outdir, name));
end
