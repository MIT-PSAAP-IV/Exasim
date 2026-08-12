function [time, qoi] = plotqoi(base, qoidesc, latexmode)
%PLOTQOI Read and plot Exasim QoI histories sorted by mesh resolution.
%
%   [time, qoi] = plotqoi(base, qoidesc, latexmode)
%
%   base is a directory containing QoI text files. The expected filename
%   convention is outqoi..._n<N>_...txt, where N is the one-dimensional mesh
%   resolution used for labels such as N^3.
%
%   Each QoI text file has one header line. Remaining rows are numeric, with
%   column 1 equal to time and columns 2:end equal to QoIs.
%
%   qoidesc is a string array or cell array of character vectors containing
%   one y-axis label per QoI. latexmode explicitly controls whether these
%   labels use MATLAB's LaTeX interpreter. Legend entries always use LaTeX.

if nargin ~= 3
    error('plotqoi:InvalidInput', ...
          'Expected exactly three inputs: base, qoidesc, latexmode.');
end

if ~(ischar(base) || (isstring(base) && isscalar(base)))
    error('plotqoi:InvalidBase', ...
          'base must be a character vector or scalar string.');
end
base = char(base);

if ~isfolder(base)
    error('plotqoi:BaseNotFound', ...
          'QoI directory does not exist: %s', base);
end

latexmode = parseLatexMode(latexmode);
qoidesc = parseQoIDescription(qoidesc);

[files, resolutions] = findQoIFiles(base);
nfiles = numel(files);
time = cell(nfiles, 1);
qoi = cell(nfiles, 1);

nqoi = [];
for i = 1:nfiles
    data = readQoIData(files{i});
    if size(data, 2) < 2
        error('plotqoi:InvalidDataColumns', ...
              'QoI file must contain at least two numerical columns: %s', files{i});
    end

    currentNqoi = size(data, 2) - 1;
    if isempty(nqoi)
        nqoi = currentNqoi;
    elseif currentNqoi ~= nqoi
        error('plotqoi:InconsistentQoICount', ...
              'File %s has %d QoIs, expected %d.', files{i}, currentNqoi, nqoi);
    end

    time{i} = data(:, 1);
    qoi{i} = data(:, 2:end);
end

if numel(qoidesc) ~= nqoi
    error('plotqoi:QoIDescriptionMismatch', ...
          'qoidesc has %d entries, but the files contain %d QoIs.', ...
          numel(qoidesc), nqoi);
end

labels = cell(nfiles, 1);
for i = 1:nfiles
    labels{i} = sprintf('$%d^3$', resolutions(i));
end

if latexmode
    ylabelInterpreter = 'latex';
else
    ylabelInterpreter = 'none';
end

for j = 1:nqoi
    figure;
    hold on;
    for i = 1:nfiles
        plot(time{i}, qoi{i}(:, j), 'LineWidth', 1.5);
    end
    xlabel('Dimensionless time');
    ylabel(qoidesc{j}, 'Interpreter', ylabelInterpreter);
    legend(labels, 'Interpreter', 'latex', 'Location', 'best');
    box on;
    grid on;
    set(gca, 'FontSize', 16);
end

end

function latexmode = parseLatexMode(latexmode)
if islogical(latexmode) && isscalar(latexmode)
    return;
end

if isnumeric(latexmode) && isscalar(latexmode) && ...
        (latexmode == 0 || latexmode == 1)
    latexmode = logical(latexmode);
    return;
end

error('plotqoi:InvalidLatexMode', ...
      'latexmode must be a logical scalar or scalar numeric 0/1.');
end

function qoidesc = parseQoIDescription(qoidesc)
if isstring(qoidesc)
    qoidesc = cellstr(qoidesc(:));
elseif iscell(qoidesc)
    qoidesc = qoidesc(:);
elseif ischar(qoidesc)
    qoidesc = {qoidesc};
else
    error('plotqoi:InvalidQoIDescription', ...
          'qoidesc must be a string array or cell array of character vectors.');
end

for i = 1:numel(qoidesc)
    if isstring(qoidesc{i}) && isscalar(qoidesc{i})
        qoidesc{i} = char(qoidesc{i});
    end
    if ~ischar(qoidesc{i})
        error('plotqoi:InvalidQoIDescription', ...
              'Each qoidesc entry must be a character vector or scalar string.');
    end
end
end

function [files, resolutions] = findQoIFiles(base)
listing = dir(fullfile(base, '*.txt'));

files = {};
resolutions = [];
for i = 1:numel(listing)
    if listing(i).isdir
        continue;
    end

    name = listing(i).name;
    resolution = extractMeshResolution(name);
    if isempty(resolution)
        continue;
    end

    files{end+1, 1} = fullfile(base, name); %#ok<AGROW>
    resolutions(end+1, 1) = resolution; %#ok<AGROW>
end

if isempty(files)
    error('plotqoi:NoQoIFiles', ...
          'No QoI text files matching outqoi..._n<N>_...txt were found in %s.', base);
end

[uniqueResolutions, ~, groups] = unique(resolutions);
counts = accumarray(groups, 1);
duplicate = uniqueResolutions(counts > 1);
if ~isempty(duplicate)
    error('plotqoi:DuplicateMeshResolution', ...
          'Multiple QoI files were found for mesh resolution n%d in %s.', ...
          duplicate(1), base);
end

[resolutions, order] = sort(resolutions);
files = files(order);
end

function resolution = extractMeshResolution(filename)
if isempty(regexp(filename, '^outqoi.*\.txt$', 'once'))
    resolution = [];
    return;
end

tokens = regexp(filename, '(?:^|_)n(\d+)(?:_|\.|$)', 'tokens');
if isempty(tokens)
    resolution = [];
    return;
end

if numel(tokens) ~= 1
    error('plotqoi:AmbiguousMeshResolution', ...
          'Could not extract a unique mesh resolution from filename: %s', filename);
end

resolution = str2double(tokens{1}{1});
if ~isfinite(resolution) || resolution <= 0 || resolution ~= floor(resolution)
    error('plotqoi:InvalidMeshResolution', ...
          'Invalid mesh resolution in filename: %s', filename);
end
end

function data = readQoIData(filename)
try
    data = readmatrix(filename, 'FileType', 'text', 'NumHeaderLines', 1);
catch
    data = dlmread(filename, '', 1, 0);
end

if isempty(data) || ~isnumeric(data)
    error('plotqoi:EmptyData', ...
          'No numerical QoI data were found in file: %s', filename);
end

if any(~isfinite(data(:)))
    error('plotqoi:InvalidNumericData', ...
          'QoI file contains NaN or Inf values: %s', filename);
end
end
