function [vgn, vgt, xgn, nlg] = velocityprofiles(base, mesh2d, master2d, wid, delta, m, alpha)
%VELOCITYPROFILES Compute normal and tangential velocity profiles.
%
%   [vgn, vgt, xgn, nlg] =
%       velocityprofiles(base, mesh2d, master2d, wid, delta, m)
%
% The function reads mean velocity fields from base, forms
% udg = cat(2, u_mean, v_mean), samples the velocity along inward normals
% with surfacefieldalongnormal, and computes signed normal/tangential
% velocity components with normaltangentvelocity.
%
% Expected primary input files are:
%   fullfile(base, 'u_mean.bin')
%   fullfile(base, 'v_mean.bin')
%
% Each file may contain either raw npe x ne payload data or a binary double
% header [npe, 1, ne] followed by the payload. If these files are absent,
% the function falls back to fullfile(base, 'sol2davg_step_0.bin') and uses
% Reynolds-average components 6 and 7.

validateInputs(base, mesh2d, master2d, wid, delta, m);

[uMean, vMean] = readMeanVelocityFields(base, mesh2d);
udg = cat(2, uMean, vMean);

[ugn, xgn, nlg] = surfacefieldalongnormal(mesh2d, master2d, udg, wid, delta, m, alpha);
[vgn, vgt] = normaltangentvelocity(ugn, nlg);

end

function validateInputs(base, mesh2d, master2d, wid, delta, m)
if ~(ischar(base) || isstring(base))
    error('base must be a character vector or string scalar.');
end
if ~isstruct(mesh2d)
    error('mesh2d must be a structure.');
end
if ~isstruct(master2d)
    error('master2d must be a structure.');
end
if ~isfield(mesh2d, 'dgnodes')
    error('mesh2d.dgnodes is required.');
end
if size(mesh2d.dgnodes, 2) ~= 2
    error('velocityprofiles expects a 2D mesh.');
end
if isempty(wid)
    error('wid must contain at least one boundary marker.');
end
if ~isscalar(delta) || ~isnumeric(delta) || ~isfinite(delta) || delta < 0
    error('delta must be a finite nonnegative scalar.');
end
if ~isscalar(m) || ~isnumeric(m) || ~isfinite(m) || m < 1 || fix(m) ~= m
    error('m must be a positive integer.');
end
end

function [uMean, vMean] = readMeanVelocityFields(base, mesh2d)
base = char(base);
npe = size(mesh2d.dgnodes, 1);
ne = size(mesh2d.dgnodes, 3);

uFile = findExistingFile(base, {'u_mean.bin', 'u_mean', 'u_mean_step_0.bin'});
vFile = findExistingFile(base, {'v_mean.bin', 'v_mean', 'v_mean_step_0.bin'});

if ~isempty(uFile) && ~isempty(vFile)
    uMean = readScalarFieldFile(uFile, npe, ne);
    vMean = readScalarFieldFile(vFile, npe, ne);
    return;
end

reavgFile = fullfile(base, 'sol2davg_step_0.bin');
if exist(reavgFile, 'file') == 2
    reavg = readReynoldsAverageFile(reavgFile, npe, 30, ne);
    uMean = reavg(:, 6, :);
    vMean = reavg(:, 7, :);
    return;
end

error('Cannot find u_mean/v_mean files or sol2davg_step_0.bin in base: %s', base);
end

function filename = findExistingFile(base, names)
filename = '';
for i = 1:numel(names)
    candidate = fullfile(base, names{i});
    if exist(candidate, 'file') == 2
        filename = candidate;
        return;
    end
end
end

function field = readScalarFieldFile(filename, npe, neExpected)
data = readDoubleFile(filename);
if isempty(data)
    error('Scalar field file is empty: %s', filename);
end

if hasHeader(data, npe)
    nc = data(2);
    ne = data(3);
    payload = data(4:end);
    if nc ~= 1
        error('Scalar field file %s has nc=%d; expected nc=1.', filename, nc);
    end
else
    ne = neExpected;
    payload = data;
end

if ne ~= neExpected
    error('Element count mismatch in %s: file has %d, mesh has %d.', ...
          filename, ne, neExpected);
end
if numel(payload) ~= npe*ne
    error('Payload size mismatch in %s: expected %d doubles, found %d.', ...
          filename, npe*ne, numel(payload));
end

field = reshape(payload, [npe, 1, ne]);
end

function reavg = readReynoldsAverageFile(filename, npe, nc, neExpected)
data = readDoubleFile(filename);
if isempty(data)
    error('Reynolds-average file is empty: %s', filename);
end

if hasHeader(data, npe)
    ncf = data(2);
    ne = data(3);
    payload = data(4:end);
    if ncf ~= nc
        error('Component count mismatch in %s: file has %d, expected %d.', ...
              filename, ncf, nc);
    end
else
    ne = neExpected;
    payload = data;
end

if ne ~= neExpected
    error('Element count mismatch in %s: file has %d, mesh has %d.', ...
          filename, ne, neExpected);
end
if numel(payload) ~= npe*nc*ne
    error('Payload size mismatch in %s: expected %d doubles, found %d.', ...
          filename, npe*nc*ne, numel(payload));
end

reavg = reshape(payload, [npe, nc, ne]);
end

function data = readDoubleFile(filename)
fid = fopen(filename, 'r');
if fid < 0
    error('Cannot open file: %s', filename);
end
cleanup = onCleanup(@() fclose(fid));
data = fread(fid, 'double');
end

function tf = hasHeader(data, npe)
tf = numel(data) >= 3 && ...
     all(isfinite(data(1:3))) && ...
     all(data(1:3) == floor(data(1:3))) && ...
     data(1) == npe && data(2) > 0 && data(3) > 0;
end
